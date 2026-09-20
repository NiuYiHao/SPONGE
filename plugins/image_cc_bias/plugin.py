import torch
import numpy as np
import os
import json
from prips import Sponge
from cryodyna.gmm.gmm import Gaussian, Grid, batch_projection

Sponge.set_backend("pytorch")

model = np.load(os.path.join(os.environ["IMAGE_CC_OUT"], "input", "model.npz"))
bb_ids = torch.as_tensor(model["bb_ids"], dtype=torch.long, device="cuda")

with open(os.path.join(os.environ["IMAGE_CC_OUT"], "input", "settings.json")) as file:
    settings = json.load(file)
masses = np.loadtxt(os.path.join(settings["base"], "sponge", "system_mass.txt"), skiprows=1)
bb_masses = torch.as_tensor(masses[model["bb_ids"]], dtype=torch.float32, device="cuda")


observables = {}

def Calculate_Force():
    # 取BB坐标
    coordinates = Sponge.dd.crd
    atom_ids = Sponge.dd.atom_local.long()
    n_local = int(Sponge.dd.atom_numbers)
    coordinates = coordinates.detach().clone().requires_grad_(True)
    global_to_local = torch.empty(n_local, dtype=torch.long, device=coordinates.device)
    global_to_local[atom_ids[:n_local]] = torch.arange(n_local, device=coordinates.device)
    bb_coordinates = coordinates[global_to_local[bb_ids]]

    # 周期边界拼接
    box = torch.as_tensor(model["box"], dtype=bb_coordinates.dtype, device=bb_coordinates.device)
    displacement = bb_coordinates[1:] - bb_coordinates[:-1]
    displacement = displacement - box * torch.round(displacement / box)
    bb_coordinates = torch.cat([bb_coordinates[:1], bb_coordinates[:1] + torch.cumsum(displacement, dim=0)], dim=0)

    # 质心 align 到图片中心
    center = (bb_coordinates * bb_masses[:, None]).sum(dim=0, keepdim=True) / bb_masses.sum()
    bb_coordinates = bb_coordinates - center

    # 计算候选rot
    pose_coordinates = bb_coordinates.detach()
    generator = torch.Generator(device=bb_coordinates.device).manual_seed(20260919)
    quaternions = torch.randn(4096, 4, generator=generator, dtype=bb_coordinates.dtype, device=bb_coordinates.device)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    w, x, y, z = quaternions.unbind(dim=-1)
    row_x = torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], dim=-1)
    row_y = torch.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], dim=-1)
    row_z = torch.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], dim=-1)
    rotations = torch.stack([row_x, row_y, row_z], dim=-2)
    rotated_coordinates = pose_coordinates @ rotations.transpose(-1, -2)
    
    # 处理投影所需的GMM参数
    image_size = int(settings["side"])
    pixel_size = float(settings["apix_A"])
    pixel_axis = (torch.arange(image_size, device=bb_coordinates.device, dtype=bb_coordinates.dtype) - image_size // 2) * pixel_size
    sigma = float(settings["sigma_A"])
    amplitudes = torch.as_tensor(model["amplitude"], dtype=bb_coordinates.dtype, device=bb_coordinates.device)

    # 根据候选rot生成候选投影图
    gaussian = Gaussian(pose_coordinates[None], torch.full_like(amplitudes[None], sigma), amplitudes[None])
    grid = Grid(coords=pixel_axis, shape=(image_size,))
    projections = torch.cat([batch_projection(gaussian, rotations[i:i+32], grid) for i in range(0, len(rotations), 32)], dim=0)
    target = torch.as_tensor(model["target"], dtype=bb_coordinates.dtype, device=bb_coordinates.device)
    
    # 计算CTF并应用于投影图
    frequency = torch.fft.fftfreq(image_size, d=pixel_size, device=bb_coordinates.device, dtype=bb_coordinates.dtype)
    fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
    defocus_u, defocus_v, astigmatism_angle, voltage_kv, cs_mm, amplitude_contrast, phase_shift = map(float, model["ctf"])
    voltage = voltage_kv * 1000
    wavelength = 12.2639 / (voltage + 0.97845e-6 * voltage**2)**0.5
    frequency_squared = fx**2 + fy**2
    frequency_angle = torch.atan2(fy, fx)
    defocus = (defocus_u + defocus_v + (defocus_u - defocus_v) * torch.cos(2 * (frequency_angle - astigmatism_angle * torch.pi / 180))) / 2
    phase = 2 * torch.pi * (-0.5 * defocus * wavelength * frequency_squared + 0.25 * cs_mm * 1e7 * wavelength**3 * frequency_squared**2) - phase_shift * torch.pi / 180
    ctf = -((1 - amplitude_contrast**2)**0.5 * torch.sin(phase) - amplitude_contrast * torch.cos(phase))
    projections = torch.fft.ifft2(torch.fft.fft2(projections) * ctf).real

    # 对模拟图和真实图施加高斯低通滤波
    lowpass = torch.exp(-2 * torch.pi**2 * float(settings["lowpass_sigma_A"])**2 * frequency_squared)
    projections = torch.fft.ifft2(torch.fft.fft2(projections) * lowpass).real
    target = torch.fft.ifft2(torch.fft.fft2(target) * lowpass).real

    # 根据投影图+后续trans搜索rot+trans的组合(pose)
    projections = projections - projections.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean()
    correlation = torch.fft.ifft2(torch.fft.fft2(target) * torch.fft.fft2(projections).conj()).real
    normalization = (projections.square().sum(dim=(-2, -1)) * target.square().sum()).sqrt().clamp_min(1e-20)
    correlation = correlation / normalization[:, None, None]
    shift_index = torch.arange(image_size, device=bb_coordinates.device)
    shifts = torch.where(shift_index <= image_size // 2, shift_index, shift_index - image_size) * pixel_size
    shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")
    allowed_shifts = (shift_x.abs() <= 20.0) & (shift_y.abs() <= 20.0)
    correlation = correlation.masked_fill(~allowed_shifts, -torch.inf)
    best_cc, best_shift_index = correlation.flatten(start_dim=1).max(dim=1)
    best_rotation_index = best_cc.argmax()
    rotation = rotations[best_rotation_index]
    shift_flat = best_shift_index[best_rotation_index]
    translation = torch.stack([shift_x.flatten()[shift_flat], shift_y.flatten()[shift_flat]])

    # 用最佳 pose 重新投影，保留坐标梯度
    gaussian.mus = bb_coordinates[None]
    image = batch_projection(gaussian, rotation[None], grid)[0]
    translation_phase = -2 * torch.pi * (fx * translation[0] + fy * translation[1])
    shift_factor = torch.polar(torch.ones_like(translation_phase), translation_phase)
    image = torch.fft.ifft2(torch.fft.fft2(image) * ctf * lowpass * shift_factor).real
    image = image - image.mean()

    # 计算 CC 与偏置势能
    score = (image * target).sum() / (image.square().sum() * target.square().sum()).sqrt().clamp_min(1e-20)
    bias_strength = float(os.environ.get("IMAGE_CC_K", "2000"))
    bias_energy = bias_strength * (1 - score)

    # 计算力并累加回 SPONGE
    bias_force = -torch.autograd.grad(bias_energy, coordinates)[0]
    assert torch.isfinite(bias_force).all(), "图像偏置力出现非有限值"
    Sponge.dd.frc[:n_local].add_(bias_force[:n_local])
    observables.update(step=int(Sponge.md_info.sys.steps), cc=score.item(), bias=bias_energy.item())


def Mdout_Print():
    if observables:
        Sponge.controller.printf("Force step {step}: CC={cc:.6f}, bias={bias:.6f} kcal/mol\n".format(**observables))

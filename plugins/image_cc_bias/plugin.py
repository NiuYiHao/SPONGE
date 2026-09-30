"""Image-driven flexible fitting and PRIPS bias in a self-contained plugin.

CLI: benchmark --system {1AKE,FUS} --out RUN; report --workspace WORK --out FIG.
SPONGE: py points to this file, LPEM_RUN supplies input.npz/settings.json.
Coordinates use Å, forces kcal/mol/Å. The report separates selected examples
from 45 held-out 1AKE targets and all 10 FUS targets.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,shutil,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

AKE=Path('/home/nyh/tools/Cryodyna/tutorial_data_1ake/pdbs')
FUS=Path('/media/nyh/Elements/FUS-simulateddata')
GRAY='#656B70';BLUE='#2379A8';GREEN='#398A76';INK='#20272D'
DO_AUDIT=False



def tensor(x, device='cuda', dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)

def align(x, reference):
    x = x - x.mean(-2, keepdim=True)
    y = reference - reference.mean(-2, keepdim=True)
    u, _, vh = torch.linalg.svd(x.transpose(-1, -2) @ y)
    d = torch.ones_like(u[..., 0, :]); d[..., -1] = torch.linalg.det(u @ vh)
    return x @ ((u * d.unsqueeze(-2)) @ vh)

def qmatrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(*q.shape[:-1],3,3)

def norm_image(x):
    x = x - x.mean((-2,-1), keepdim=True)
    return x / x.square().sum((-2,-1),keepdim=True).sqrt().clamp_min(1e-12)

def cc(a,b):
    return (norm_image(a)*norm_image(b)).sum((-2,-1))

class Projector:
    def __init__(self, side=64, apix=2., sigma=3., device='cuda', dtype=torch.float32):
        self.side, self.apix, self.sigma = side, apix, sigma
        self.line = (torch.arange(side,device=device,dtype=dtype)-(side-1)/2)*apix

    def render(self,x,r,shift):
        c = x @ r.transpose(-1,-2)
        cxy = c[...,:2] + shift[...,None,:]
        dx = (self.line[:,None] - cxy[...,0].unsqueeze(-2))/self.sigma
        dy = (self.line[:,None] - cxy[...,1].unsqueeze(-2))/self.sigma
        return torch.exp(-.5*dy.square()) @ torch.exp(-.5*dx.square()).transpose(-1,-2)

def blur(x, sigma):
    f=torch.fft.fftfreq(x.shape[-1],device=x.device,dtype=x.dtype)
    fy,fx=torch.meshgrid(f,f,indexing='ij')
    return torch.fft.ifft2(torch.fft.fft2(x)*torch.exp(-2*math.pi**2*sigma**2*(fx*fx+fy*fy))).real

def resize(x,side):
    return F.interpolate(x[:,None],size=(side,side),mode='area')[:,0]

def pose_search(x, target, p, nrot=1024, topk=4, steps=60, seed=731):
    """Global SO(3) + FFT translation, followed by continuous pose refinement.

    Every image receives its own pose. The same solver serves every template.
    """
    gen=torch.Generator(device=x.device).manual_seed(seed)
    q=torch.randn(nrot,4,generator=gen,device=x.device,dtype=x.dtype)
    q[0]=tensor([1,0,0,0],x.device,x.dtype)
    target=norm_image(target)
    ft=torch.fft.fft2(target)
    vals=[]; inds=[]; shifts=[]
    s=p.side; ix=torch.arange(s,device=x.device)
    signed=torch.where(ix<=s//2,ix,ix-s).to(x.dtype)*p.apix
    yy,xx=torch.meshgrid(signed,signed,indexing='ij')
    mask=(xx.abs()<s*p.apix*.2)&(yy.abs()<s*p.apix*.2)
    # Keep the maximum over translations for each image/orientation.
    with torch.no_grad():
        for start in range(0,nrot,64):
            pred=norm_image(p.render(x,qmatrix(q[start:start+64]),torch.zeros(64 if start+64<=nrot else nrot-start,2,device=x.device,dtype=x.dtype)))
            fp=torch.fft.fft2(pred)
            blockv=[]; blocks=[]
            for j in range(0,len(target),8):
                corr=torch.fft.ifft2(ft[j:j+8,None]*fp.conj()[None]).real
                corr.masked_fill_(~mask,-torch.inf)
                v,k=corr.flatten(-2).max(-1)
                blockv.append(v); blocks.append(k)
            vals.append(torch.cat(blockv));shifts.append(torch.cat(blocks))
        scores=torch.cat(vals,1); shiftids=torch.cat(shifts,1)
        _,best=scores.topk(topk,dim=1)
        qb=q[best].clone()
        si=shiftids.gather(1,best)
        tr=torch.stack([xx.flatten()[si],yy.flatten()[si]],-1)
    qb=qb.reshape(-1,4).requires_grad_();tr=tr.reshape(-1,2).requires_grad_()
    obs=target[:,None].expand(-1,topk,-1,-1).reshape(-1,s,s)
    opt=torch.optim.Adam([{'params':[qb],'lr':.025},{'params':[tr],'lr':p.apix*.08}])
    for _ in range(steps):
        opt.zero_grad()
        # Chunking retains the graph while bounding activations.
        for j in range(0,len(qb),32):
            loss=(1-cc(p.render(x,qmatrix(qb[j:j+32]),tr[j:j+32]),obs[j:j+32])).sum()
            loss.backward()
        opt.step()
    with torch.no_grad():
        final=torch.cat([cc(p.render(x,qmatrix(qb[j:j+32]),tr[j:j+32]),obs[j:j+32]) for j in range(0,len(qb),32)]).reshape(-1,topk)
        best=final.argmax(1); ar=torch.arange(len(target),device=x.device)
        return qb.reshape(-1,topk,4)[ar,best].detach(),tr.reshape(-1,topk,2)[ar,best].detach(),final[ar,best]

def geometry(x,ref):
    # Local distances preserve bond and short-range structure; no target enters.
    loss=x.new_zeros(())
    for offset,weight in [(1,1.),(2,.3),(3,.1)]:
        d=(x[offset:]-x[:-offset]).norm(dim=-1)
        d0=(ref[offset:]-ref[:-offset]).norm(dim=-1)
        loss=loss+weight*(d-d0).square().mean()
    return loss

def fit_ensemble(library,targets,p,nrot=1024,pose_steps=60,fit_steps=300,fit_indices=None,coordinate_steps=300,residual_modes=0):
    """Shared structure from multiple views; withheld structure never accepted.

    Fits convex ensemble coordinates, then small smooth residual displacements.
    Poses remain independently optimized for every image.
    """
    library=align(library,library[0])
    fit_indices=torch.arange(len(targets),device=library.device) if fit_indices is None else fit_indices
    allq=[];allt=[];alls=[]
    for x in library:
        q,t,s=pose_search(x,targets,p,nrot=nrot,steps=pose_steps)
        allq.append(q);allt.append(t);alls.append(s)
    score=torch.stack(alls)
    best=int(score[:,fit_indices].mean(1).argmax())
    ref=library[best].clone()
    q=allq[best].clone().requires_grad_();tr=allt[best].clone().requires_grad_()
    # Affine ensemble coefficients allow an exact discrete-template start.
    weights=torch.zeros(len(library),device=library.device);weights[best]=1
    weights.requires_grad_()
    opt=torch.optim.Adam([{'params':[weights],'lr':.025},{'params':[q],'lr':.004},{'params':[tr],'lr':.035}])
    trace=[]; bestloss=float('inf'); checkpoint=None
    for step in range(fit_steps):
        opt.zero_grad()
        x=torch.einsum('k,knc->nc',weights,library)
        lossimage=x.new_zeros(())
        # Full batch objective accumulated through chunks.
        for ids in fit_indices.split(24):
            lossimage=lossimage+(1-cc(p.render(x,qmatrix(q[ids]),tr[ids]),targets[ids])).sum()/len(fit_indices)
        penalty=.003*geometry(x,ref)+.0003*(x-ref).square().mean()
        loss=lossimage+penalty
        if float(loss.detach())<bestloss:
            bestloss=float(loss.detach());checkpoint=(x.detach().clone(),q.detach().clone(),tr.detach().clone(),weights.detach().clone())
        loss.backward()
        # Remove the normal component before Adam; all feasible changes sum to 0.
        # Without this, Adam's common-mode sign can pin a simplex vertex.
        weights.grad.sub_(weights.grad.mean())
        opt.step()
        # Euclidean projection onto the probability simplex, never a truth metric.
        with torch.no_grad():
            u=weights.sort(descending=True).values
            css=u.cumsum(0)-1
            idx=torch.arange(1,len(u)+1,device=u.device)
            rho=(u-css/idx>0).sum()-1
            weights.copy_((weights-css[rho]/(rho+1)).clamp_min(0))
        if step%25==0 or step==fit_steps-1:
            trace.append({'step':step,'cc':1-float(lossimage.detach()),'regularizer':float(penalty.detach()),'objective':float(loss.detach())})
    x,q,tr,w=checkpoint
    # Residue coordinates are explicitly flexible beyond the template span.
    # Local geometry and a weak displacement prior constrain image-only updates.
    origin=x.clone();q=q.clone().requires_grad_();tr=tr.clone().requires_grad_()
    if residual_modes:
        dist=torch.cdist(ref,ref)
        adj=(dist<10).to(ref.dtype);adj.fill_diagonal_(0)
        # Sequence edges guarantee a connected graph for extended chains.
        ar=torch.arange(len(ref)-1,device=ref.device);adj[ar,ar+1]=1;adj[ar+1,ar]=1
        lap=torch.diag(adj.sum(1))-adj
        _,evec=torch.linalg.eigh(lap)
        basis=evec[:,1:residual_modes+1]*len(ref)**.5
        displacement=torch.zeros(residual_modes,3,device=ref.device,requires_grad=True)
    else:
        basis=None;displacement=torch.zeros_like(origin,requires_grad=True)
    opt=torch.optim.Adam([{'params':[displacement],'lr':.035 if residual_modes else .07},{'params':[q],'lr':.0015},{'params':[tr],'lr':.025}])
    bestloss=float('inf');fine_checkpoint=None
    for step in range(coordinate_steps):
        opt.zero_grad()
        x=origin+(basis@displacement if basis is not None else displacement)
        li=x.new_zeros(())
        for ii in fit_indices.split(24):
            li=li+(1-cc(p.render(x,qmatrix(q[ii]),tr[ii]),targets[ii])).sum()/len(fit_indices)
        reg=.02*geometry(x,ref)+.00003*(x-origin).square().mean()
        if residual_modes:
            contact=(dist<10)&(dist>0)
            reg=reg+.003*(torch.cdist(x,x)[contact]-dist[contact]).square().mean()
        loss=li+reg
        if float(loss.detach())<bestloss:
            bestloss=float(loss.detach());fine_checkpoint=(x.detach().clone(),q.detach().clone(),tr.detach().clone())
        loss.backward();opt.step()
        if step%25==0 or step==coordinate_steps-1:
            trace.append({'stage':'residue','step':step,'cc':1-float(li.detach()),'regularizer':float(reg.detach()),'objective':float(loss.detach())})
    if fine_checkpoint is not None:x,q,tr=fine_checkpoint
    else:x,q,tr=origin,q.detach(),tr.detach()
    # Equal pose-only refinement budget for held-out views and training views.
    def refine_pose(shape,qs,ts,steps):
        qs=qs.clone().requires_grad_();ts=ts.clone().requires_grad_()
        optimizer=torch.optim.Adam([{'params':[qs],'lr':.003},{'params':[ts],'lr':.025}])
        for _ in range(steps):
            optimizer.zero_grad()
            for ids in torch.arange(len(targets),device=shape.device).split(24):
                (1-cc(p.render(shape,qmatrix(qs[ids]),ts[ids]),targets[ids])).sum().backward()
            optimizer.step()
        with torch.no_grad():
            pred=torch.cat([p.render(shape,qmatrix(qs[ids]),ts[ids]) for ids in torch.arange(len(targets),device=shape.device).split(24)])
        return pred,qs.detach(),ts.detach()
    bp,bq,bt=refine_pose(ref,allq[best],allt[best],fit_steps+coordinate_steps+60)
    fp,fq,ft=refine_pose(x,q,tr,60)
    return dict(baseline=ref,flexible=x,baseline_projection=bp,flexible_projection=fp,
                baseline_q=bq,flexible_q=fq,baseline_shift=bt,flexible_shift=ft,
                selected=best,weights=w,library_cc=score,trace=trace)

def metrics(x,truth):
    x=align(x,truth)
    rmsd=(x-truth).square().sum(-1).mean().sqrt()
    d=torch.cdist(x,x);dt=torch.cdist(truth,truth)
    mask=torch.triu(torch.ones_like(d,dtype=torch.bool),diagonal=4)
    drmsd=(d[mask]-dt[mask]).square().mean().sqrt()
    near=mask&(dt<15)
    lddt=torch.stack([((d-dt).abs()[near]<tol).float().mean() for tol in [.5,1.,2.,4.]]).mean()
    bonds=(x[1:]-x[:-1]).norm(dim=-1)
    clash=(d[mask]<3.).float().mean()
    return dict(rmsd_A=float(rmsd),drmsd_A=float(drmsd),lddt=float(lddt),
                bond_rmse_A=float((bonds-3.8).square().mean().sqrt()),clash_fraction=float(clash))

def read_ca(path):
    records = [s for s in Path(path).read_text().splitlines()
               if s.startswith('ATOM') and s[12:16].strip() == 'CA' and s[16] in ' A']
    xyz = np.array([[float(s[30:38]), float(s[38:46]), float(s[46:54])] for s in records])
    ids = [(s[21], s[22:27].strip(), s[17:20]) for s in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate CA identities: {path}')
    return xyz - xyz.mean(0), ids

def structures(system):
    if system == '1AKE':
        paths = [AKE / f'1akeA_{i}.pdb' for i in range(1, 51)]
    else:
        root = FUS / 'clustering_k5_k10_k30/k10/cluster_representatives_k10_allatom'
        paths = [next(root.glob(f'cluster_{i}_frame_*.pdb')) for i in range(1, 11)]
    parsed = [read_ca(p) for p in paths]
    assert all(ids == parsed[0][1] for _, ids in parsed), 'Residue mapping differs'
    return np.stack([x for x, _ in parsed]), paths, parsed[0][1]

def fus_images(group):
    paths = sorted((FUS / str(group)).glob('*.png'))
    assert len(paths) == 216
    return np.stack([np.asarray(Image.open(p), dtype=np.float32) for p in paths]), paths

def write_pdb(path, xyz, ids):
    with open(path, 'w') as f:
        for i, (x, key) in enumerate(zip(xyz, ids), 1):
            chain, residue, name = key
            f.write(f'ATOM  {i:5d}  CA  {name:3s} {chain or "A"}{int(residue):4d}    '
                    f'{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00           C\n')
        f.write('END\n')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def align_np(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    xc=x-x.mean(0);yc=y-y.mean(0)
    u,_,v=np.linalg.svd(xc.T@yc)
    d=np.ones(3);d[-1]=np.linalg.det(u@v)
    return xc@((u*d)@v)+y.mean(0)

def pearson(a,b):
    x=np.asarray(a,dtype=float).ravel();y=np.asarray(b,dtype=float).ravel()
    x-=x.mean();y-=y.mean()
    return float(x@y/np.linalg.norm(x)/np.linalg.norm(y))

def measure(x,truth):
    a=align_np(x,truth);error=np.linalg.norm(a-truth,axis=1)
    i,j=np.triu_indices(len(a),4)
    delta=np.linalg.norm(a[i]-a[j],axis=1)-np.linalg.norm(truth[i]-truth[j],axis=1)
    return a,error,float(np.sqrt(np.mean(error**2))),float(np.sqrt(np.mean(delta**2))),len(i)

def prepare_cases():
    cases=[];image_manifest=[];struct_manifest=[];metric_rows=[];error_rows=[]
    for system,run,target in [('1AKE','ake_single_view_v1',19),('FUS','fus_supplied_graph_v2',7)]:
        case=ROOT/'runs'/run/f'{target:02d}'
        s=json.loads((case/'summary.json').read_text());d=dict(np.load(case/'result.npz'))
        xyz,paths,ids=structures(system)
        assert np.max(np.abs(xyz[target-1]-d['truth']))<1e-5
        n=len(ids);truth=d['truth'].astype(float)
        assert np.sqrt(np.mean((align_np(d['baseline'],xyz[s['template']-1])-xyz[s['template']-1])**2))<1e-4
        rows=list(csv.DictReader((case/'image_scores.csv').open()))
        if system=='FUS':
            eligible=[(i,r) for i,r in enumerate(rows) if r['split']=='test' and float(r['delta_cc'])>0]
            view=max(eligible,key=lambda z:float(z[1]['flexible_cc']))[0]
            selection=f'108 evaluation views; {len(eligible)} with positive CC gain; choose highest flexible CC among these'
            native=list(csv.DictReader((case/'native_image_scores.csv').open()))[view]
        else:
            view=0;selection='the sole controlled image for target 19';native=None
        for i,row in enumerate(rows):
            image_manifest.append(dict(system=system,target=target,index_zero_based=i,source=row['image'],
                                       role='structure_fit' if row['split']=='train' else 'pose_registered_evaluation',
                                       displayed=i==view,saved_array=str(case/'result.npz')+f'::target_images[{i}]'))
        candidate_ids=d['template_ids'].tolist()
        for k in candidate_ids:
            struct_manifest.append(dict(system=system,target=target,role='candidate_template',id=k,
                                        selected=k==s['template'],path=str(paths[k-1]),ca_count=n,
                                        ensemble_weight=s['weights'][str(k)],sha256=sha256(paths[k-1])))
        for role,p in [('reference_truth',paths[target-1]),('initial_selected',case/'baseline.pdb'),('flexible_result',case/'flexible.pdb')]:
            struct_manifest.append(dict(system=system,target=target,role=role,id='',selected='',path=str(p),ca_count=n,ensemble_weight='',sha256=sha256(p)))
        result={'system':system,'run':run,'case':str(case),'target':target,'template':s['template'],
                'n_ca':n,'candidate_ids':candidate_ids,'truth_path':str(paths[target-1]),
                'initial_source':str(paths[s['template']-1]),'initial_saved':str(case/'baseline.pdb'),
                'flexible_saved':str(case/'flexible.pdb'),'source_image':rows[view]['image'],
                'display_view_zero_based':view,'selection_rule':selection,'summary':s,'scores':{},'native_single':native}
        result['exported_images']={}
        for key in ['target_images','baseline_projection','flexible_projection']:
            arr=d[key][view].astype(float);z=(arr-arr.mean())/arr.std()
            imagefile=OUT/f'{system}_{target}_{key}.png'
            plt.imsave(imagefile,z,cmap='gray_r',vmin=-2,vmax=4)
            result['exported_images'][key]=str(imagefile)
        if system=='FUS':
            rawfile=OUT/f'FUS_7_{Path(rows[view]["image"]).stem}_raw.png'
            if not rawfile.exists():
                shutil.copy2(rows[view]['image'],rawfile)
            assert sha256(rawfile)==sha256(rows[view]['image'])
            result['exported_images']['raw_png']=str(rawfile)
        for method in ['baseline','flexible']:
            a,e,rmsd,drmsd,pairs=measure(d[method],truth)
            score=pearson(d[f'{method}_projection'][view],d['target_images'][view])
            csv_key='template_cc' if method=='baseline' else 'flexible_cc'
            assert abs(score-float(rows[view][csv_key]))<1e-6
            assert abs(rmsd-s[method]['rmsd_A'])<2e-5
            assert abs(drmsd-s[method]['drmsd_A'])<2e-5
            result['scores'][method]={'single_image_cc':score,'rmsd_A':rmsd,'drmsd_A':drmsd,'distance_pairs':pairs,
                                      'cc_comparison':f'{method}_projection[{view}] vs target_images[{view}]',
                                      'structure_comparison':f'{method} vs truth ({paths[target-1].name})'}
            d[f'{method}_aligned']=a;d[f'{method}_error']=e
            metric_rows.append({'system':system,'target':target,'method':method,**result['scores'][method]})
            for i,(key,err) in enumerate(zip(ids,e),1):
                error_rows.append(dict(system=system,target=target,method=method,ca_index=i,chain=key[0],residue_id=key[1],resname=key[2],error_A=err))
        result['data']=d;cases.append(result)
    for name,rows in [('images',image_manifest),('structures',struct_manifest),('metrics',metric_rows),('residue_errors',error_rows)]:
        with (OUT/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'selection.json').write_text(json.dumps([{k:v for k,v in c.items() if k!='data'} for c in cases],ensure_ascii=False,indent=2))
    return cases

def write_notes(cases):
    lines=['# All-in-one：比较对象与来源','',
           '本图复用已有保存结果，重新核对结构身份，并独立重算单图 CC、RMSD、dRMSD。原逐残基图的两个结构案例保持一致。','',
           '## 谁和谁比较','',
           '| 指标 | 模板匹配 | 柔性结果 |',
           '|---|---|---|',
           '| CC | 模板在其估计 pose 下的投影 vs 同一观测工作图 | 柔性结构在其估计 pose 下的投影 vs 同一观测工作图 |',
           '| RMSD | 选中模板 vs 参考真值；proper Kabsch 后全部 Cα 的均方根距离 | 最终柔性结构 vs 同一参考真值；相同 Cα 映射与对齐定义 |',
           '| dRMSD | 模板内部的 Cα 对距离 vs 参考内部相同 Cα 对距离 | 柔性结构内部的 Cα 对距离 vs 参考内部相同 Cα 对距离 |',
           '| 逐残基 error | 对齐后每个模板 Cα 到参考同名 Cα 的欧氏距离 | 对齐后每个柔性 Cα 到参考同名 Cα 的欧氏距离 |','',
           'dRMSD 采用上三角残基对，序号差至少 4。RMSD 与 error 使用三维坐标；骨架面板为对齐后坐标的同一固定斜视图。',
           '', '图像显示采用逐图 z-score、共同范围 −2 至 4、gray_r；计算 CC 使用保存的数值数组。',
           '']
    for c in cases:
        d=c['data'];n=c['n_ca'];case=Path(c['case']);v=c['display_view_zero_based']
        lines += [f'## {c["system"]}：目标 {c["target"]}，选中模板 {c["template"]}','',
                  f'- 参考真值：[源 PDB]({c["truth_path"]})。',
                  f'- 初始模板：[源 PDB]({c["initial_source"]})；[保存坐标]({c["initial_saved"]})。',
                  f'- 柔性结果：[flexible.pdb]({c["flexible_saved"]})。',
                  f'- 候选模板编号：{c["candidate_ids"]}，每份具有相同的 {n} 个 Cα 身份映射。',
                  f'- 图中工作图数组：`{case}/result.npz::target_images[{v}]`。',
                  f'- 图中预测数组：同一文件的 `baseline_projection[{v}]` 和 `flexible_projection[{v}]`。',
                  f'- dRMSD 包含 {c["scores"]["baseline"]["distance_pairs"]:,} 个距离对。','']
        if c['system']=='1AKE':
            lines+=['图像由 `1akeA_19.pdb` 生成，64×64，2 Å/像素，Gaussian σ=3 Å，单张干净受控投影。目标坐标和真实 pose 用于生成与事后评价；拟合输入为观测图及候选库。',
                    '模板 13 起步；组合权重为模板 1：0.004785443、13：0.481193900、25：0.514020622；然后优化残基层位移。','']
        else:
            lines += [f'展示的原始图片：[ {Path(c["source_image"]).name} ]({c["source_image"]})。其零基索引为 {v}，属于结构拟合留出组。',
                      '结构拟合输入：`7/70000.png, 70002.png, …, 70214.png`，共 108 张。',
                      '配准后评价输入：`7/70001.png, 70003.png, …, 70215.png`，共 108 张。完整绝对路径见 `images.csv`。',
                      '每张评价图的 pose 都由该图单独精修；组平均 CC 汇总全部 108 张评价图。',
                      f'展示规则：{c["selection_rule"]}。该选图属于事后案例展示。',
                      '源图 100×100；反转灰度、σ=2 原始像素 Gaussian 平滑，再缩放至 64×64。估计原始像素为 3 Å，工作像素 4.6875 Å，Gaussian 投影 σ=9 Å。',
                      '模板 2 的组合权重为 1；随后优化 16 个接触图低频形变模式。','']
        lines+=['| 指标 | 初始模板 | 柔性结果 |','|---|---:|---:|']
        for key,label in [('single_image_cc','图中单张 CC'),('rmsd_A','RMSD / Å'),('drmsd_A','dRMSD / Å')]:
            lines.append(f'| {label} | {c["scores"]["baseline"][key]:.7f} | {c["scores"]["flexible"][key]:.7f} |')
        if c['system']=='FUS':
            for key,label in [('test_cc','108 张评价工作图平均 CC'),('native_test_cc','108 张原始 PNG 平均 CC')]:
                lines.append(f'| {label} | {c["summary"]["baseline"][key]:.7f} | {c["summary"]["flexible"][key]:.7f} |')
        lines+=['']
    lines+=['## 比较预算与证据范围','',
            '全部模板初次搜索：1024 个 SO(3) 候选、FFT 平移搜索、100 步 pose 精修。1AKE 柔性阶段为 500 步模板组合与 300 步残基位移；刚性模板追加 860 步 pose-only，柔性结果追加 60 步 pose-only。FUS 对应为 500+500 步联合优化，刚性模板追加 1060 步 pose-only，柔性结果追加 60 步 pose-only。两条路线按各自记录的预算报告。',
            '这些结构来自图像驱动、几何正则拟合。1AKE 为已知前向模型的受控案例，FUS 的尺度与成像参数含估计。SPONGE 短程积分另有独立记录。',
            '案例选择沿用原图：1AKE 19 在其留出目标中 RMSD 降幅较大；FUS 7 来自固定参数组 3–10 中三项共同改善且 RMSD 降幅最大的案例。',
            '', '## 文件','',
            '- `all_in_one.png/pdf/svg/tiff`：整张组合图。',
           '- `selection.json`：对象、来源、选图规则及重新计算的分数。',
           '- `1AKE_19_*.png`、`FUS_7_*.png`：图中各张观测/预测的单独显示文件。FUS 原始 PNG 副本的校验值与源文件一致。',
            '- `images.csv`：217 张输入图的角色与展示标识。',
            '- `structures.csv`：全部候选库、选中模板、真值、柔性结果及文件校验值。',
            '- `metrics.csv`：两案例的精确分数与比较对象。',
            '- `residue_errors.csv`：每个 Cα 的链名、残基号、残基名和误差。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')

def draw_examples():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=prepare_cases();write_notes(cases)
    W,H=7.2,7.45;fig=plt.figure(figsize=(W,H),facecolor='white');groups=[]
    def text(x,y,s,**kw):
        return fig.text(x/W,y/H,s,color=kw.pop('color',INK),**kw)
    def axes(x,y,w,h,label):
        return fig.add_axes([x/W,y/H,w/W,h/H],label=label)
    def arrow(x1,y1,x2,y2,color=GRAY):
        fig.add_artist(FancyArrowPatch((x1/W,y1/H),(x2/W,y2/H),transform=fig.transFigure,
                                      arrowstyle='-|>',mutation_scale=8,lw=.8,color=color))
    centers=[.86,2.31,3.76]
    for k,c in enumerate(cases):
        top=7.12-3.55*k;d=c['data'];name=c['system'];v=c['display_view_zero_based']
        text(.18,top,f'{"a" if k==0 else "b"}  {name}',fontsize=12,weight='bold')
        text(1.53,top+.01,f'{c["n_ca"]} Cα',fontsize=8)
        text(5.00,top+.01,'● 模板',fontsize=7,color=GRAY)
        text(6.04,top+.01,'● 柔性',fontsize=7,color=BLUE)
        az=.7;el=.45
        rz=np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
        rx=np.array([[1,0,0],[0,np.cos(el),-np.sin(el)],[0,np.sin(el),np.cos(el)]])
        coords=[d['truth']@rz@rx,d['baseline_aligned']@rz@rx,d['flexible_aligned']@rz@rx]
        merged=np.concatenate(coords);lo=merged[:,:2].min(0);hi=merged[:,:2].max(0)
        mid=(lo+hi)/2;span=max(hi-lo)*1.1
        struct_axes=[];image_axes=[]
        for j,(p,color) in enumerate(zip(coords,[GREEN,GRAY,BLUE])):
            cx=centers[j]
            title=f'真值 {c["target"]}' if j==0 else f'初始 {c["template"]}' if j==1 else '拟合结果'
            text(cx,top-.34,title,ha='center',fontsize=8,color=color,weight='bold')
            a=axes(cx-.62,top-1.71,1.24,1.24,f'{name}_structure_{j}')
            if j:
                a.plot(coords[0][:,0],coords[0][:,1],color=GREEN,lw=.9,alpha=.32,zorder=1)
            seg=np.stack([p[:-1,:2],p[1:,:2]],axis=1)
            depth=(p[:-1,2]+p[1:,2])/2;order=np.argsort(depth)
            a.add_collection(LineCollection(seg[order],colors=color,linewidths=.85 if k==0 else .6,
                                            capstyle='round',joinstyle='round',zorder=2))
            a.set_xlim(mid[0]-span/2,mid[0]+span/2);a.set_ylim(mid[1]-span/2,mid[1]+span/2)
            a.set_axis_off();struct_axes.append(a)
            arrow(cx,top-1.76,cx,top-1.97,color)
        arrow(2.965,top-1.03,3.105,top-1.03,BLUE)
        groups.append([a.get_label() for a in struct_axes])
        for j,(key,color) in enumerate(zip(['target_images','baseline_projection','flexible_projection'],[GREEN,GRAY,BLUE])):
            cx=centers[j];a=axes(cx-.56,top-3.16,1.12,1.12,f'{name}_image_{j}')
            z=d[key][v].astype(float);z=(z-z.mean())/z.std()
            a.imshow(z,cmap='gray_r',vmin=-2,vmax=4,interpolation='nearest');a.set_axis_off();image_axes.append(a)
            if j==0:label='观测 19 · 64×64' if k==0 else f'观测 {Path(c["source_image"]).name}'
            else:label='CC '+format(c['scores']['baseline' if j==1 else 'flexible']['single_image_cc'],'.7f')
            text(cx,top-3.35,label,ha='center',fontsize=6.5,color=color)
        groups.append([a.get_label() for a in image_axes])
        for m,(key,label) in enumerate([('rmsd_A','RMSD (Å) ↓'),('drmsd_A','dRMSD (Å) ↓')]):
            y=top-.44-m*.76
            text(5.02,y,label,fontsize=7)
            a=axes(5.03,y-.45,1.38,.35,f'{name}_{key}')
            b=c['scores']['baseline'][key];f=c['scores']['flexible'][key]
            a.barh([1,0],[b,f],height=.53,color=[GRAY,BLUE])
            a.set_xlim(0,max(b,f)*1.04);a.set_ylim(-.5,1.5);a.set_axis_off()
            for val,yy,col in [(b,y-.19,GRAY),(f,y-.40,BLUE)]:
                text(6.58,yy,f'{val:.3f}',fontsize=8,color=col,weight='bold' if col==BLUE else 'normal')
        a=axes(5.04,top-3.03,1.94,1.04,f'{name}_error')
        for method,color in [('baseline',GRAY),('flexible',BLUE)]:
            a.plot(np.arange(1,c['n_ca']+1),d[method+'_error'],color=color,lw=.65)
        a.set_xlabel('Cα',fontsize=6.5,labelpad=1);a.set_ylabel('Error (Å)',fontsize=6.5,labelpad=2)
        a.tick_params(labelsize=6,length=2,width=.5,pad=2)
    fig.canvas.draw()
    audit_alignment(fig,json_out=str(OUT/'all_in_one.alignment.json'),
        overlay_svg=str(OUT/'all_in_one.alignment.svg'),row_groups=groups,column_groups=[],strict=True)
    fig.savefig(OUT/'all_in_one.pdf');fig.savefig(OUT/'all_in_one.svg')
    fig.savefig(OUT/'all_in_one.png',dpi=400)
    fig.savefig(OUT/'all_in_one.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


def draw_statistics(out, do_audit=False):
    base=out
    raw=json.loads((base/'input_summaries.json').read_text())
    systems={s:[r for r in raw if r['system']==s and not r['seen_in_library']] for s in ['1AKE','FUS']}
    assert len(raw)==60 and len(systems['1AKE'])==45 and len(systems['FUS'])==10
    assert {r['target'] for r in raw if r['system']=='1AKE' and r['seen_in_library']}=={1,13,25,37,50}
    results=[];rows=[];targets=[]
    for system,records in systems.items():
        keys=sorted(set.intersection(*(set(r[m]) for r in records for m in ['baseline','flexible'])))
        for key in keys:
            for method in ['baseline','flexible']:
                x=np.array([r[method][key] for r in records],dtype=float)
                assert np.isfinite(x).all()
                rows.append(dict(system=system,metric=key,method=method,n=len(x),mean=x.mean(),sample_sd=x.std(ddof=1)))
        cc_key='cc' if system=='1AKE' else 'test_cc'
        joint=sum(all((r['flexible'][key]-r['baseline'][key])*direction>1e-5
                      for key,direction in [(cc_key,1),('rmsd_A',-1),('drmsd_A',-1)]) for r in records)
        results.append(dict(system=system,n=len(records),joint_success=joint,cc_field=cc_key))
        for r in records:
            for method in ['baseline','flexible']:
                for key,value in r[method].items():
                    targets.append(dict(system=system,target=r['target'],template=r['template'],method=method,metric=key,value=value))
    for filename,data in [('statistics.csv',rows),('per_target.csv',targets)]:
        with (out/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'statistics.json').write_text(json.dumps({'groups':results,'metrics':rows},indent=2))
    W,H=13.5,7.75
    fig=plt.figure(figsize=(W,H),facecolor='white');groups=[]
    plate=fig.add_axes([.02/W,.15/H,7.2/W,7.45/H],label='selected_examples')
    plate.imshow(plt.imread(base/'example_plate.png'));plate.set_axis_off()
    def text(x,y,s,**kw):
        fig.text(x/W,y/H,s,color=kw.pop('color',INK),**kw)
    text(7.7,7.43,'● 模板',color=GRAY,fontsize=9);text(8.63,7.43,'● 柔性',color=BLUE,fontsize=9)
    text(10.10,7.43,'细线：逐目标   粗点：均值 ± SD',fontsize=8)
    specs=[('cc','CC ↑'),('rmsd_A','RMSD (Å) ↓'),('drmsd_A','dRMSD (Å) ↓'),('lddt','lDDT-like ↑'),
           ('standardized_mse','MSE (all) ↓'),('bond_rmse_A','Bond RMSE (Å) ↓'),('clash_fraction','Clash fraction ↓')]
    for si,(system,records) in enumerate(systems.items()):
        header=7.04-3.56*si
        text(7.7,header,f'{"c" if si==0 else "d"}  {system}  ·  n = {len(records)}',fontsize=12,weight='bold')
        for row in range(2):
            current=[]
            for col in range(4):
                idx=row*4+col;x=7.78+col*1.39;y=header-1.33-row*1.48
                if idx==7:
                    text(x+.5,y+.63,'CC / RMSD / dRMSD',fontsize=7,ha='center')
                    text(x+.5,y+.30,f'{results[si]["joint_success"]} / {len(records)}',fontsize=20,weight='bold',ha='center',color=BLUE)
                    text(x+.5,y+.06,'共同改善',fontsize=8,ha='center')
                    continue
                key,title=specs[idx]
                actual='test_cc' if key=='cc' and system=='FUS' else key
                b=np.array([r['baseline'][actual] for r in records]);f=np.array([r['flexible'][actual] for r in records])
                a=fig.add_axes([x/W,y/H,1.03/W,.86/H],label=f'{system}_{key}')
                a.plot(np.array([[0,1]]*len(b)).T,np.array([b,f]),color='#CBD0D4',alpha=.5,lw=.5,zorder=1)
                jitter=np.linspace(-.08,.08,len(b))
                for pos,values,color in [(0,b,GRAY),(1,f,BLUE)]:
                    a.scatter(pos+jitter,values,s=4,color=color,alpha=.4,zorder=2)
                    a.errorbar([pos],[values.mean()],yerr=[values.std(ddof=1)],fmt='o',markersize=4,
                               color=color,ecolor=color,capsize=3,elinewidth=1.0,zorder=3)
                a.set_xlim(-.35,1.35);a.set_xticks([]);a.tick_params(length=2,width=.5,pad=2)
                a.yaxis.set_major_locator(MaxNLocator(3));a.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                if np.ptp(np.r_[b,f])==0:
                    a.set_ylim(-.025,.025);a.set_yticks([0])
                fmt=(lambda v:f'{v:.7f}') if key=='cc' else (lambda v:f'{v:.3g}')
                text(x+.515,y+1.17,title,ha='center',fontsize=7.5)
                text(x+.515,y+.97,f'{fmt(b.mean())} → {fmt(f.mean())}',ha='center',fontsize=6.6)
                current.append(a.get_label())
            groups.append(current)
    text(7.7,.12,'1AKE：45 个留出构象；FUS：10 个目标，CC 为每目标 108 张评价图的均值。',fontsize=7)
    fig.canvas.draw()
    if do_audit:
        sys.path.insert(0,'/home/nyh/.codex/skills/nature-figure/scripts')
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=str(out/'all_in_one.alignment.json'),
            overlay_svg=str(out/'all_in_one.alignment.svg'),row_groups=groups,column_groups=[],strict=True)
    fig.savefig(out/'all_in_one.pdf');fig.savefig(out/'all_in_one.svg')
    fig.savefig(out/'all_in_one.png',dpi=300)
    fig.savefig(out/'all_in_one.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    print(json.dumps(results,indent=2))

def audit_alignment(fig, **kwargs):
    if DO_AUDIT:
        sys.path.insert(0, '/home/nyh/.codex/skills/nature-figure/scripts')
        from audit_panel_alignment import require_matplotlib_panel_alignment
        return require_matplotlib_panel_alignment(fig, **kwargs)


def benchmark_run(args):
    """Fit images first, then evaluate saved structures against reference PDBs."""
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    raw,paths,ids=structures(args.system)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    config.update(plugin_sha256=sha256(__file__),torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(),
                  structures=[{'path':str(p),'sha256':sha256(p)} for p in paths],
                  truth_access='acquisition and post-fit evaluation',
                  split='1AKE anchors 1,13,25,37,50; FUS leave-one-structure-out')
    saved=out/'config.json'
    if saved.exists():
        old=json.loads(saved.read_text())
        keys=['system','side','apix','sigma','views','seed','nrot','pose_steps','fit_steps','coordinate_steps','residual_modes','blur_pixels','structures','plugin_sha256']
        if any(old.get(k)!=config.get(k) for k in keys):
            raise ValueError('Choose a fresh output directory for changed parameters, source, or input data.')
    saved.write_text(json.dumps(config,indent=2))
    indices=[int(v)-1 for v in args.targets.split(',')] if args.targets else list(range(len(raw)))
    if any(i<0 or i>=len(raw) for i in indices):raise ValueError('Target index outside the structure library')
    for i in indices:
        case=out/f'{i+1:02d}';case.mkdir(exist_ok=True)
        if (case/'summary.json').exists():continue
        start=time.time();trainids=[0,12,24,36,49] if args.system=='1AKE' else [j for j in range(len(raw)) if j!=i]
        library=tensor(raw[trainids]);p=Projector(args.side,args.apix,args.sigma)
        if args.system=='1AKE':
            gen=torch.Generator(device='cuda').manual_seed(args.seed)
            q=torch.randn(args.views,4,device='cuda',generator=gen)
            shift=torch.randn(args.views,2,device='cuda',generator=gen)*2
            with torch.no_grad():target=p.render(tensor(raw[i]),qmatrix(q),shift)
            imagepaths=[f'controlled_view_{v}' for v in range(len(target))];fitids=None
            acquisition={'known_apix_A':args.apix,'known_sigma_A':args.sigma,'noise':'clean','seed':args.seed}
            np.savez(case/'acquisition_truth.npz',q=q.cpu().numpy(),shift_A=shift.cpu().numpy())
        else:
            original,imagepaths=fus_images(i+1)
            target=resize(blur(-tensor(original),args.blur_pixels),args.side)
            fitids=torch.arange(0,len(target),2,device='cuda')
            acquisition={'estimated_apix_A':args.apix,'native_apix_A':args.apix*args.side/100,
                         'assumed_sigma_A':args.sigma,'working_side':args.side,'native_side':100,
                         'blur_native_pixels':args.blur_pixels,'contrast':'dark protein',
                         'physical_calibration':'estimated; generator metadata pending',
                         'train_views':'even indices','test_views':'odd indices',
                         'images':[{'path':str(v),'sha256':sha256(v)} for v in imagepaths]}
        (case/'acquisition.json').write_text(json.dumps(acquisition,indent=2))
        print(f'{args.system} target {i+1}: {len(trainids)} templates, {len(target)} views',flush=True)
        result=fit_ensemble(library,target,p,args.nrot,args.pose_steps,args.fit_steps,fitids,args.coordinate_steps,args.residual_modes)
        truth=tensor(raw[i]);bm=metrics(result['baseline'],truth);fm=metrics(result['flexible'],truth)
        bs=cc(result['baseline_projection'],target).cpu().numpy();fs=cc(result['flexible_projection'],target).cpu().numpy()
        s=dict(system=args.system,regime='clean' if args.system=='1AKE' else 'supplied',target=i+1,
               template=trainids[result['selected']]+1,seen_in_library=i in trainids,n_views=len(target),
               baseline=bm|{'cc':float(bs.mean()),'standardized_mse':float(2*(1-bs.mean()))},
               flexible=fm|{'cc':float(fs.mean()),'standardized_mse':float(2*(1-fs.mean()))},
               weights={str(j+1):float(w) for j,w in zip(trainids,result['weights'])})
        if args.system=='FUS':
            native_scores={};nativep=Projector(100,args.apix*args.side/100,args.sigma)
            for method,scores in [('baseline',bs),('flexible',fs)]:
                s[method].update(train_cc=float(scores[::2].mean()),test_cc=float(scores[1::2].mean()))
                values=[]
                with torch.no_grad():
                    for lo in range(0,len(target),24):
                        pred=nativep.render(result[method],qmatrix(result[method+'_q'][lo:lo+24]),result[method+'_shift'][lo:lo+24])
                        values.extend(cc(pred,-tensor(original[lo:lo+24])).cpu().tolist())
                native_scores[method]=values
                s[method].update(native_cc=float(np.mean(values)),native_test_cc=float(np.mean(values[1::2])))
            with (case/'native_image_scores.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['image','template_cc','flexible_cc'])
                w.writerows(zip(map(str,imagepaths),native_scores['baseline'],native_scores['flexible']))
        scorekey='cc' if args.system=='1AKE' else 'test_cc'
        s['joint_success']=bool(fm['rmsd_A']<bm['rmsd_A']-1e-5 and fs.mean()>bs.mean()+1e-5 and fm['drmsd_A']<bm['drmsd_A']-1e-5)
        s['joint_test_success']=bool(s['flexible'][scorekey]>s['baseline'][scorekey]+1e-5 and fm['rmsd_A']<bm['rmsd_A']-1e-5 and fm['drmsd_A']<bm['drmsd_A']-1e-5)
        s['seconds']=time.time()-start
        serial={key:value.cpu().numpy() for key,value in result.items() if isinstance(value,torch.Tensor)}
        serial.update(target_images=target.cpu().numpy(),truth=raw[i],template_ids=np.array(trainids)+1)
        np.savez_compressed(case/'result.npz',**serial)
        for method in ['baseline','flexible']:write_pdb(case/f'{method}.pdb',serial[method],ids)
        (case/'trace.json').write_text(json.dumps(result['trace'],indent=2))
        with (case/'image_scores.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['image','split','template_cc','flexible_cc','delta_cc'])
            w.writerows((str(path),'train' if fitids is None or j%2==0 else 'test',b,v,v-b) for j,(path,b,v) in enumerate(zip(imagepaths,bs,fs)))
        (case/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s),flush=True)
    rows=[json.loads(p.read_text()) for p in sorted(out.glob('[0-9][0-9]/summary.json'))]
    (out/'summary.json').write_text(json.dumps(rows,indent=2))


def _plotting_setup():
    global plt,LineCollection,FancyArrowPatch,MaxNLocator,ScalarFormatter
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.ticker import MaxNLocator,ScalarFormatter
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP','DejaVu Sans'],
                         'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.5,
                         'axes.spines.top':False,'axes.spines.right':False})


def report_run(args):
    global ROOT,OUT,DO_AUDIT
    ROOT=Path(args.workspace).resolve();OUT=Path(args.out).resolve();DO_AUDIT=args.audit
    OUT.mkdir(parents=True,exist_ok=True);_plotting_setup()
    records=[];verification=[]
    for name in ['ake_single_view_v1','fus_supplied_graph_v2']:
        for path in sorted((ROOT/'runs'/name).glob('[0-9][0-9]/summary.json')):
            s=json.loads(path.read_text());d=np.load(path.parent/'result.npz')
            for method in ['baseline','flexible']:
                fresh=metrics(tensor(d[method]),tensor(d['truth']))
                delta=max(abs(s[method][key]-value) for key,value in fresh.items())
                assert delta<1e-4,(path,method,delta)
                scores=cc(tensor(d[method+'_projection']),tensor(d['target_images'])).cpu().numpy()
                assert abs(float(scores.mean())-s[method]['cc'])<1e-6
                if s['system']=='FUS':assert abs(float(scores[1::2].mean())-s[method]['test_cc'])<1e-6
                verification.append({'system':s['system'],'target':s['target'],'method':method,'max_structure_delta':delta})
            s['_source']=str(path);s['_sha256']=sha256(path);records.append(s)
    (OUT/'input_summaries.json').write_text(json.dumps(records,indent=2))
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    report_out=OUT;OUT=report_out/'examples';OUT.mkdir(exist_ok=True)
    draw_examples()
    shutil.copyfile(OUT/'all_in_one.png',report_out/'example_plate.png')
    OUT=report_out;draw_statistics(OUT,args.audit)


def replay_run(args):
    """Recompute statistics and the combined figure from the document's payload."""
    import base64
    from html.parser import HTMLParser

    class PayloadParser(HTMLParser):
        def __init__(self):
            super().__init__();self.active=False;self.parts=[]

        def handle_starttag(self,tag,attrs):
            if tag=='script' and dict(attrs).get('id')=='lpem-replay':self.active=True

        def handle_data(self,data):
            if self.active:self.parts.append(data)

        def handle_endtag(self,tag):
            if tag=='script':self.active=False

    parser=PayloadParser();parser.feed(Path(args.document).read_text())
    payload=json.loads(''.join(parser.parts))
    image=base64.b64decode(payload['example_plate_png'],validate=True)
    if hashlib.sha256(image).hexdigest()!=payload['example_sha256']:
        raise ValueError('Embedded example image checksum differs from the document manifest')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'example_plate.png').write_bytes(image)
    (out/'input_summaries.json').write_text(json.dumps(payload['input_summaries'],indent=2))
    _plotting_setup();draw_statistics(out,args.audit)


def cli():
    global AKE,FUS
    parser=argparse.ArgumentParser(description='Image-driven Cα flexible fitting, PRIPS bias, and reproducible evidence figure.')
    parser.add_argument('--ake-root',default=str(AKE));parser.add_argument('--fus-root',default=str(FUS))
    sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('benchmark');b.add_argument('--system',choices=['1AKE','FUS'],required=True);b.add_argument('--out',required=True)
    b.add_argument('--targets',default='');b.add_argument('--side',type=int,default=64)
    b.add_argument('--apix',type=float);b.add_argument('--sigma',type=float)
    b.add_argument('--views',type=int,default=1);b.add_argument('--nrot',type=int,default=1024)
    b.add_argument('--pose-steps',type=int,default=100);b.add_argument('--fit-steps',type=int,default=500)
    b.add_argument('--coordinate-steps',type=int);b.add_argument('--residual-modes',type=int)
    b.add_argument('--seed',type=int,default=731);b.add_argument('--blur-pixels',type=float,default=2.)
    r=sub.add_parser('report');r.add_argument('--workspace',required=True);r.add_argument('--out',required=True);r.add_argument('--audit',action='store_true')
    r=sub.add_parser('replay');r.add_argument('--document',required=True);r.add_argument('--out',required=True);r.add_argument('--audit',action='store_true')
    args=parser.parse_args();AKE=Path(args.ake_root);FUS=Path(args.fus_root)
    if args.command=='benchmark':
        defaults=(2.,3.,300,0) if args.system=='1AKE' else (4.6875,9.,500,16)
        for key,value in zip(['apix','sigma','coordinate_steps','residual_modes'],defaults):
            if getattr(args,key) is None:setattr(args,key,value)
        if args.fit_steps<=0 or args.coordinate_steps<0 or args.nrot<4:parser.error('Use positive fit steps, nonnegative coordinate steps, and at least four rotations.')
        if args.residual_modes<0:parser.error('Residual modes must be nonnegative.')
        benchmark_run(args)
    elif args.command=='report':report_run(args)
    else:replay_run(args)


def _initialize_sponge():
    global Sponge,OUT,d,cfg,ids,box,ref,goal,target,rotation,shift,quaternion,p,handle,writer,frames,observations
    from prips import Sponge
    Sponge.set_backend('pytorch');OUT=Path(os.environ['LPEM_RUN'])
    d=np.load(OUT/'input.npz');cfg=json.loads((OUT/'settings.json').read_text())
    ids=tensor(d['bb_ids'],dtype=torch.long);box=tensor(d['box']);ref=tensor(d['reference']);goal=tensor(d['inferred_target'])
    target=tensor(d['image']);rotation=tensor(d['rotation']);shift=tensor(d['shift']);quaternion=tensor(d['quaternion'])
    p=Projector(cfg['side'],cfg['apix_A'],cfg['sigma_A']);handle=(OUT/'observables.csv').open('w');writer=csv.writer(handle)
    writer.writerow(['step','cc','image_energy_kcal_mol','restraint_energy_kcal_mol','force_max'])
    frames=[];observations={}


def Calculate_Force():
    global rotation,shift,quaternion
    n=int(Sponge.dd.atom_numbers)
    atom_ids=Sponge.dd.atom_local[:n].long()
    assert n==len(Sponge.dd.crd[:n])
    assert int(atom_ids.min())==0 and int(atom_ids.max())==n-1 and len(torch.unique(atom_ids))==n
    mapping=torch.empty(n,device='cuda',dtype=torch.long)
    mapping[atom_ids]=torch.arange(n,device='cuda')
    coords=Sponge.dd.crd[:n].detach().clone().requires_grad_()
    x=coords[mapping[ids]]
    delta=x[1:]-x[:-1];delta=delta-box*torch.round(delta/box)
    x=torch.cat([x[:1],x[:1]+delta.cumsum(0)])
    aligned=align(x,ref)
    step=int(Sponge.md_info.sys.steps)
    if cfg.get('pose_interval',0)>0 and step>0 and step%cfg['pose_interval']==0:
        qr=quaternion.detach().clone().requires_grad_();tr=shift.detach().clone().requires_grad_()
        opt=torch.optim.Adam([{'params':[qr],'lr':.015},{'params':[tr],'lr':.06}])
        best=float('-inf');bestpose=None
        for _ in range(100):
            opt.zero_grad();s=cc(p.render(aligned.detach(),qmatrix(qr),tr),target)
            if float(s.detach())>best:best=float(s.detach());bestpose=(qr.detach().clone(),tr.detach().clone())
            (1-s).backward();opt.step()
        quaternion,shift=bestpose;rotation=qmatrix(quaternion)
    score=cc(p.render(aligned,rotation,shift),target)
    u=cfg['image_k']*(1-score)
    restraint=.5*cfg['restraint_k']*(aligned-goal).square().sum()
    force=-torch.autograd.grad(u+restraint,coords)[0]
    assert torch.isfinite(force).all()
    before=Sponge.dd.frc[:n].detach().clone() if len(frames)==0 else None
    Sponge.dd.frc[:n].add_(force)
    if before is not None:
        applied=Sponge.dd.frc[:n]-before
        np.savez(OUT/'force_injection.npz',expected=force.detach().cpu().numpy(),actual=applied.detach().cpu().numpy())
    if step%cfg['stride']==0 or step==cfg['steps']-1:
        frames.append(aligned.detach().cpu().numpy())
        np.save(OUT/'backbone_A.npy',np.stack(frames))
        with (OUT/'poses.jsonl').open('a') as f:f.write(json.dumps({'step':step,'rotation':rotation.cpu().tolist(),'shift':shift.cpu().tolist()})+'\n')
        writer.writerow([step,float(score),float(u),float(restraint),float(force.abs().max())]);handle.flush()
    observations.update(step=step,cc=float(score))

def Mdout_Print():
    if observations:Sponge.controller.printf('LPEM step {step} CC {cc:.6f}\n'.format(**observations))

if __name__ == '__main__':
    cli()
elif __name__ == 'sponge_pyplugin' or 'LPEM_RUN' in os.environ:
    _initialize_sponge()

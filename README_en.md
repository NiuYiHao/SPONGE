# SPONGE

! Note: The current repository version is the SPONGE2 `v2.0.0-beta.2` prerelease

[中文](README.md)

## LPEM-MDFF: 1AKE / FUS RMSD examples and plugin

| Content | Entry point |
|---|---|
| Result previews, setup and reproduction commands | [Flexible fitting examples](plugins/image_cc_bias/README.md) |
| 1AKE RMSD plot reproduction | [examples/1ake/reproduce_rmsd.py](plugins/image_cc_bias/examples/1ake/reproduce_rmsd.py) |
| FUS RMSD plot reproduction | [examples/fus/reproduce_rmsd.py](plugins/image_cc_bias/examples/fus/reproduce_rmsd.py) |
| Fitting algorithm and SPONGE/PRIPS plugin | [plugins/image_cc_bias/plugin.py](plugins/image_cc_bias/plugin.py) |

The examples redraw per-target RMSD plots on a CPU from archived measured results included in this repository. The example guide also provides commands for fitting again from PDB / PNG inputs.

### From a fresh clone to RMSD result plots

Use Git and Python 3.11 with `venv` / `ensurepip`. Run the following commands in a Linux/macOS terminal to download the develop branch, create an isolated environment, generate both plots, and verify the outputs:

```bash
git clone --branch develop --single-branch https://github.com/NiuYiHao/SPONGE.git
cd SPONGE
python3 -I -m venv .venv
source .venv/bin/activate
python -I -m pip --isolated install --index-url https://pypi.org/simple -r plugins/image_cc_bias/examples/requirements.txt
python -I -m pip check
python -I plugins/image_cc_bias/examples/1ake/reproduce_rmsd.py --out outputs/1ake
python -I plugins/image_cc_bias/examples/fus/reproduce_rmsd.py --out outputs/fus
python -I plugins/image_cc_bias/examples/verify_rmsd.py --out-root outputs
```

The last command prints `PASS 1AKE` and `PASS FUS`. Open `outputs/1ake/1ake_rmsd.png` and `outputs/fus/fus_rmsd.png`. PDF, SVG, CSV and JSON outputs accompany each image. On Ubuntu/Debian, `sudo apt-get install python3-venv` supplies the system Python venv component.

Development and commits target `develop`.

**Simulation Package tOward Next GEneration of molecular modelling**

```
 ____  ____   ___  _   _  ____ _____
/ ___||  _ \ / _ \| \ | |/ ___| ____|
\___ \| |_) | | | |  \| | |  _|  _|
 ___) |  __/| |_| | |\  | |_| | |___
|____/|_|    \___/|_| \_|\____|_____|
```

A high-performance simulation engine for next-generation molecular modelling.

Official website: https://spongemm.cn

When you need to understand or work with SPONGE:

- If you are a human user, start with the [Getting Started guide](docs/getting-started.md)
  and the [Input File Reference](docs/input-reference/README.md).
- If you are an AI assistant:
  - For using SPONGE, install this project's `skills` into your global skills directory, for example `~/.claude/skills` or `~/.agents/skills`.
  - For developing SPONGE, link your project-level skills directory to this project's `skills`, for example `.claude/skills` or `.agents/skills`.

## Citation

> Huang, Y.-P., Xia, Y., Yang, L., Wei, J., Yang, Y.I. and Gao, Y.Q. (2022),
> SPONGE: A GPU-Accelerated Molecular Dynamics Package with Enhanced Sampling and AI-Driven Algorithms.
> _Chin. J. Chem._, 40: 160-168.
> https://doi.org/10.1002/cjoc.202100456

## License

[SPONGE Licensing Agreement (English, controlling)](LICENSE) · [Chinese translation](LICENSE.zh-CN.md) · [Third-Party Notices](THIRD_PARTY_NOTICES)

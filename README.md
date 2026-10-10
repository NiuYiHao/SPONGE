# SPONGE

! 注意：当前仓库版本为 SPONGE2 `v2.0.0-beta.2` 预发布版本

[English](README_en.md)

## LPEM-MDFF：1AKE / FUS RMSD 示例与 plugin

| 内容 | 直接入口 |
|---|---|
| 示例结果图、环境与复现命令 | [柔性拟合示例说明](plugins/image_cc_bias/README.md) |
| 1AKE RMSD 图复现脚本 | [examples/1ake/reproduce_rmsd.py](plugins/image_cc_bias/examples/1ake/reproduce_rmsd.py) |
| FUS RMSD 图复现脚本 | [examples/fus/reproduce_rmsd.py](plugins/image_cc_bias/examples/fus/reproduce_rmsd.py) |
| 拟合算法与 SPONGE/PRIPS plugin | [plugins/image_cc_bias/plugin.py](plugins/image_cc_bias/plugin.py) |

两个示例使用仓库内保存的实测记录，在 CPU 上重画逐目标 RMSD 图；说明页另列从 PDB / PNG 重新拟合的命令。

**Simulation Package tOward Next GEneration of molecular modelling**

```
 ____  ____   ___  _   _  ____ _____
/ ___||  _ \ / _ \| \ | |/ ___| ____|
\___ \| |_) | | | |  \| | |  _|  _|
 ___) |  __/| |_| | |\  | |_| | |___
|____/|_|    \___/|_| \_|\____|_____|
```

面向下一代分子模拟的高性能模拟引擎

官网: https://spongemm.cn

当你需要了解或操作 SPONGE 时：

- 如果你是人类用户：请从[快速入门](docs/getting-started.md)和
  [输入参数参考](docs/input-reference/README.md)开始。
- 如果你是 AI 助手：
  - 若用于使用 SPONGE：将本项目的 `skills` 安装到你的全局 skills 目录，例如 `~/.claude/skills` 或 `~/.agents/skills`。
  - 若用于开发 SPONGE：将你的项目级 skills 目录链接到本项目的 `skills`，例如 `.claude/skills` 或 `.agents/skills`。

## 引用

> Huang, Y.-P., Xia, Y., Yang, L., Wei, J., Yang, Y.I. and Gao, Y.Q. (2022),
> SPONGE: A GPU-Accelerated Molecular Dynamics Package with Enhanced Sampling and AI-Driven Algorithms.
> _Chin. J. Chem._, 40: 160-168.
> https://doi.org/10.1002/cjoc.202100456

## 许可证

[SPONGE Licensing Agreement (English, controlling)](LICENSE) · [中文译本](LICENSE.zh-CN.md) · [第三方声明](THIRD_PARTY_NOTICES)

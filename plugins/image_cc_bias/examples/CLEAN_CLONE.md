# 全新 clone 验收：1AKE / FUS RMSD 图

验收日期：2026-10-10。目标分支：`develop`。验收范围为 GitHub 全新 clone、独立环境安装、按 README 重画保存结果并核验两张 RMSD 图。

## 来源与环境

- 验收 clone 的提交：`550b8492bfd4c1df51b9545e5c4ba5b7a94f1a4d`。
- Python 3.11.12；独立 `.venv`；`include-system-site-packages = false`。
- NumPy 1.26.4、Matplotlib 3.10.3；依赖通过 pip 从 PyPI 安装。
- `python -I` 开启隔离模式；NumPy / Matplotlib 均从新 `.venv` 加载。
- `pip check` 输出 `No broken requirements found.`。
- 两份独立 clone 生成的两张 PNG 分别具有相同 SHA256，文件内容一致。

## 从头执行的命令

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

Python 3.11 解释器及其 venv / ensurepip 组件是运行前置条件。

## 实际输出

```text
No broken requirements found.
PASS 1AKE: 45 targets; case 19 RMSD 0.920656 → 0.101559 Å
PASS FUS: 10 targets; case 7 RMSD 15.854187 → 14.751323 Å
Saved verification: outputs/verification.json
```

每个体系生成 PNG、PDF、SVG、CSV、JSON 五种输出。逐目标 CSV 与仓库参考值一致，JSON 的案例与均值在 `1e-12` 相对/绝对容差内一致，源数据 SHA256 一致。两张 PNG 均为 4322×2480 px。

PDF 字体与文字碰撞检查均通过：最小字体 7 pt；每图 0 fail、0 warn。两图的标签、图例和全部目标已经过视觉核验。

机器可读结果及实际依赖版本见 [clean_clone.json](clean_clone.json)。

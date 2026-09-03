#!/usr/bin/env bash
#create conda env pykt space if not exists, and evaluate a sample model. follow /Users/l/klee_code/git_repos/pykt_space/quick_start.md.txt
set -euo pipefail
cd "$(dirname "$0")"

CONDA_HOME=${CONDA_HOME:-/Users/l/miniconda3}
ENV_NAME=pykt
PY="$CONDA_HOME/envs/$ENV_NAME/bin/python"

# 官方文档是 conda create --name=pykt python=3.7.5，但本机是 osx-arm64，conda 上 python 最低只有 3.8，故用 3.9
if [ ! -x "$PY" ]; then
    "$CONDA_HOME/bin/conda" create -y --name=$ENV_NAME python=3.9
fi

# source activate pykt  —— 脚本里直接用环境内的解释器，避免依赖 conda 的 shell 钩子
"$PY" -c 'import pykt' 2>/dev/null || \
    "$PY" -m pip install -U pykt-toolkit -i https://pypi.python.org/simple

# pykt 的预处理依赖 pandas 1.x 的 groupby 行为（pandas 2 下 uid 会被写成 "(50121,)" 导致解析失败）
"$PY" -c 'import pandas, sys; sys.exit(0 if pandas.__version__.startswith("1.") else 1)' 2>/dev/null || \
    "$PY" -m pip install "numpy==1.26.4" "pandas==1.5.3"

# i downloaded data ASSISTments2015 into /Users/l/klee_code/git_repos/pykt_space/quick_start.md.txt
# 实际的原始数据文件是本目录下的 2015_100_skill_builders_main_problems.csv
# 下面这一步 = quick_start 里的 data_preprocess.py + wandb_dkt_train.py + wandb_predict.py，
# 但全部通过 pykt 库 API 完成，不需要 clone pykt-toolkit 仓库拿 examples/
# 默认只取前 3000 个用户跑 demo；跑全量数据加 --max_users 0
"$PY" run_pykt_demo.py "$@"

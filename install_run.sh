#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# make this also work under remote server (root path is still ~/klee_code)
# conda 的安装位置各机器不同，按 环境变量 -> 已激活的 conda -> PATH 上的 conda -> 常见安装位置 依次找
find_conda_home() {
    [ -x "${CONDA_HOME:-}/bin/conda" ] && { echo "$CONDA_HOME"; return; }
    [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ] && { echo "${CONDA_EXE%/bin/conda}"; return; }
    local base
    base="$(conda info --base 2>/dev/null || true)"
    [ -x "$base/bin/conda" ] && { echo "$base"; return; }
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        [ -x "$d/bin/conda" ] && { echo "$d"; return; }
    done
}
CONDA_HOME="$(find_conda_home)"
if [ -z "$CONDA_HOME" ]; then
    echo "[错误] 找不到 conda。装一个，或者手动指定：CONDA_HOME=/path/to/miniconda3 $0" >&2
    exit 1
fi
ENV_NAME="${ENV_NAME:-pykt}"
PY="$CONDA_HOME/envs/$ENV_NAME/bin/python"
echo "[env] conda=$CONDA_HOME, env=$ENV_NAME"
# 输出重定向/管道（tee 到日志、tmux）时 python 默认块缓冲，进度会卡住不刷出来
export PYTHONUNBUFFERED=1

# 官方文档是 conda create --name=pykt python=3.7.5，但本机是 osx-arm64，conda 上 python 最低只有 3.8，故用 3.9
# 用 conda-forge 而不是默认频道：defaults 需要先 conda tos accept 接受 Anaconda 服务条款，
# 没接受过的机器（比如 ecnu 那台）会直接 CondaToSNonInteractiveError
if [ ! -x "$PY" ]; then
    "$CONDA_HOME/bin/conda" create -y --name=$ENV_NAME -c conda-forge --override-channels python=3.9
fi

# source activate pykt  —— 脚本里直接用环境内的解释器，避免依赖 conda 的 shell 钩子
"$PY" -c 'import pykt' 2>/dev/null || \
    "$PY" -m pip install -U pykt-toolkit -i https://pypi.python.org/simple

# pykt 的预处理依赖 pandas 1.x 的 groupby 行为（pandas 2 下 uid 会被写成 "(50121,)" 导致解析失败）
"$PY" -c 'import pandas, sys; sys.exit(0 if pandas.__version__.startswith("1.") else 1)' 2>/dev/null || \
    "$PY" -m pip install "numpy==1.26.4" "pandas==1.5.3"

# pykt_gh（GitHub 版 pykt 整包）比 PyPI 版多用了这几个包：dimkt 的数据加载用 tqdm，
# extrakt/fluckt 等用 einops，cskt 等在模块顶层 import matplotlib
"$PY" -c 'import tqdm, einops, matplotlib' 2>/dev/null || \
    "$PY" -m pip install tqdm einops matplotlib

# i downloaded data ASSISTments2015 into /Users/l/klee_code/git_repos/pykt_space/quick_start.md.txt
# 实际的原始数据文件是本目录下的 2015_100_skill_builders_main_problems.csv
# 下面这一步 = quick_start 里的 data_preprocess.py + wandb_dkt_train.py + wandb_predict.py，
# 但全部通过 pykt 库 API 完成，不需要 clone pykt-toolkit 仓库拿 examples/
# 默认只取前 3000 个用户跑 demo；跑全量数据加 --max_users 0
"$PY" run_pykt_demo.py "$@"

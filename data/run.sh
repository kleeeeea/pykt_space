#!/usr/bin/env bash
#refer to /Users/l/klee_code/git_repos/iclr/dualkt/pykt_space/data/EduData-master, download datasets in /Users/l/klee_code/git_repos/iclr/dualkt/iclr (1).pdf
#
# 论文（iclr (1).pdf 第 6 页）用的 5 个数据集：ASSIST2009 / ASSIST2017 / NIPS34 / XES3G5M / Algebra2005
#
# 下载方式分两种（每个数据集在 DATASETS 里第二列标明）：
#   edudata:<名字>  走 EduData（data/EduData-master），它自带 URL 表并负责解压
#   url:<地址>      直接下。两种情况需要：EduData 清单里没有（XES3G5M），
#                   或 EduData 的条目是整个目录、会连带下很多用不到的文件（Algebra2005：
#                   KDD_Cup_2010/ 下有 5 个 zip，我们只要 algebra_2005_2006）
#
# 用法：
#   ./run.sh                    # 下全部（已存在的跳过）
#   ./run.sh assist2009 nips34  # 只下指定的
#   FORCE=1 ./run.sh assist2009 # 重下并覆盖
set -euo pipefail
cd "$(dirname "$0")"

# 名字|来源|落地目录（目录存在就跳过）
DATASETS=(
    "assist2009|edudata:assistment-2009-2010-skill|2009_skill_builder_data_corrected"
    "assist2017|edudata:assistment-2017|anonymized_full_release_competition_dataset"
    "nips34|edudata:NIPS-2020|NIPS2020"
    "algebra2005|url:http://base.ustc.edu.cn/data/KDD_Cup_2010/algebra_2005_2006.zip|algebra_2005_2006"
    # XES3G5M 在 Google Drive 上，需要 gdown；README 里写明"下载即表示接受其 license"
    "xes3g5m|gdrive:1eFiIYyh5O2V90RA0brammGH6EpHvPDQe|XES3G5M"
)

# 复用项目里那套 conda 环境（和 ../install_run.sh 同名同逻辑），EduData 就装在里面
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
ENV_NAME="${ENV_NAME:-pykt}"
PY="${PY:-$CONDA_HOME/envs/$ENV_NAME/bin/python}"
[ -x "$PY" ] || { echo "[错误] 找不到 python：$PY（先跑 ../install_run.sh 建环境）" >&2; exit 1; }
echo "[env] python=$PY"

ensure_edudata() {
    "$PY" -c 'import EduData' 2>/dev/null || {
        echo "[install] EduData（从本目录的 EduData-master 装，可编辑模式）"
        "$PY" -m pip install -q -e EduData-master
    }
}

fetch() {
    local name="$1" src="$2" dest="$3"
    if [ -e "$dest" ] && [ -z "${FORCE:-}" ]; then
        echo "[skip] $name -> $dest 已存在（要重下加 FORCE=1）"
        return
    fi
    case "$src" in
        edudata:*)
            ensure_edudata
            echo "[edudata] $name <- ${src#edudata:}"
            "$PY" -c "
from EduData import get_data
get_data('${src#edudata:}', '.')"
            ;;
        url:*)
            local url="${src#url:}" zip="${dest}.zip"
            echo "[curl] $name <- $url"
            curl -fL --retry 3 -C - -o "$zip" "$url"
            mkdir -p "$dest" && unzip -q -o "$zip" -d "$dest" && rm -f "$zip"
            ;;
        gdrive:*)
            "$PY" -c 'import gdown' 2>/dev/null || "$PY" -m pip install -q gdown
            echo "[gdown] $name <- Google Drive ${src#gdrive:}"
            echo "        注意：其 README 声明「下载即表示接受该数据集的 license」"
            local zip="${dest}.zip"
            "$PY" -m gdown "${src#gdrive:}" -O "$zip"
            mkdir -p "$dest" && unzip -q -o "$zip" -d "$dest" && rm -f "$zip"
            ;;
        *)
            echo "[错误] 不认识的来源：$src" >&2; return 1 ;;
    esac
    echo "[ok] $name -> $dest"
}

wanted=("$@")
for entry in "${DATASETS[@]}"; do
    IFS='|' read -r name src dest <<< "$entry"
    if [ ${#wanted[@]} -gt 0 ]; then
        printf '%s\n' "${wanted[@]}" | grep -qx "$name" || continue
    fi
    fetch "$name" "$src" "$dest"
done
echo "全部完成。当前 data/ 下："
ls -d */ 2>/dev/null

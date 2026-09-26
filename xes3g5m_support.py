"""XES3G5M 接入：它和其它数据集不一样，作者给的已经是 pykt 格式的切分结果。

其它数据集是原始作答日志，要跑 pykt 的 process_raw_data + split_datasets；
XES3G5M（data/run.sh 下的那份）里 kc_level/ 已经有：
    train_valid_sequences.csv  每行一个学生的一段序列，带 fold(0~4) 和 selectmasks，可直接用
    test.csv                   每行一个学生的完整序列，逗号连接，没有 selectmasks
                               —— 这是 pykt 说的 "test_original_file"，还要切成
                                  test_sequences.csv（截断到 maxlen）和 test_window_sequences.csv（滑窗）
所以这里不做预处理，只做三件事：
    1. 按 uid 采样（--max_users），因为切分好的文件没法再按原始日志采样
    2. 用 pykt 自己的 generate_sequences / generate_window_sequences 把 test.csv 切出那两个文件
    3. 写 data_config.json 里 xes3g5m 那一条

单独调试：
    prepare_xes3g5m(src_dir, dpath, config_file, max_users=3000, maxlen=200)
"""
import json
import os

import pandas as pd

# 这份数据每步只有一个 KC（没有 "1_2" 这种多 KC 写法），所以 max_concepts=1
MAX_CONCEPTS = 1


def _sample_uids(df, max_users, stratify_fold=True):
    """取前 max_users 个学生；max_users<=0 表示全要。

    stratify_fold：按 fold 分层取。train_valid_sequences.csv 是按 fold 排好序存的
    （前 6669 行全是 fold 0，最后是 fold 4），每个学生只属于一个 fold。直接按文件顺序
    取前 N 个 uid 会几乎全落在 fold 0 上——那样 --fold 0 当验证集时，训练集（fold 1~4）
    几乎是空的，AUC 只有 0.55。所以要在每个 fold 里各取一份。
    """
    if max_users <= 0:
        return df
    if not stratify_fold or "fold" not in df.columns or df["fold"].nunique() <= 1:
        keep = set(df["uid"].drop_duplicates().head(max_users))
        return df[df["uid"].isin(keep)]

    folds = sorted(df["fold"].unique())
    per_fold = max(1, round(max_users / len(folds)))
    keep = set()
    for f in folds:
        keep |= set(df.loc[df["fold"] == f, "uid"].drop_duplicates().head(per_fold))
    return df[df["uid"].isin(keep)]


def _max_id(series):
    """一列逗号连接的 id，返回最大值（忽略 padding -1）。"""
    return max(int(x) for s in series for x in str(s).split(",") if x not in ("-1", ""))


def prepare_xes3g5m(src_dir, dpath, config_file, max_users, maxlen=200, min_seq_len=3,
                    max_test_users=0):
    """把 kc_level 下的文件采样后放到 work/data/xes3g5m/，并写好 data_config。返回 dconfig。

    max_test_users：测试集单独的学生数上限（0 = 跟 max_users 一样）。分开控制是因为
    test_window_sequences 是按步长 1 滑窗生成的，XES3G5M 学生序列很长（平均 370+ 步），
    3000 个学生会切出上百万行、3GB+ 的文件，评估慢且吃内存。
    """
    import time
    from pykt.preprocess.split_datasets import generate_sequences, generate_window_sequences

    os.makedirs(dpath, exist_ok=True)
    train_src = os.path.join(src_dir, "train_valid_sequences.csv")
    test_src = os.path.join(src_dir, "test.csv")
    for p in (train_src, test_src):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} 不在，先跑 data/run.sh xes3g5m")

    # ---- 训练/验证：已经切好，采样后原样落地 ----
    t0 = time.time()
    print(f"[xes3g5m] 读 {train_src}（{os.path.getsize(train_src) / 1e6:.0f}MB）...")
    train = _sample_uids(pd.read_csv(train_src), max_users)
    train.to_csv(os.path.join(dpath, "train_valid_sequences.csv"), index=False)
    print(f"[xes3g5m] train_valid: {train['uid'].nunique()} 个学生 / {len(train)} 段序列"
          f"（{time.time() - t0:.1f}s）")

    # ---- 测试：先采样，再用 pykt 的函数切成 test_sequences / test_window_sequences ----
    t0 = time.time()
    print(f"[xes3g5m] 读 {test_src}（{os.path.getsize(test_src) / 1e6:.0f}MB）...")
    test = _sample_uids(pd.read_csv(test_src), max_test_users or max_users).copy()
    test["fold"] = -1  # pykt 约定测试集 fold = -1
    test.to_csv(os.path.join(dpath, "test.csv"), index=False)
    steps = sum(len(str(s).split(",")) for s in test["responses"])
    print(f"[xes3g5m] test: {test['uid'].nunique()} 个学生 / {steps} 步交互，开始切分"
          f"（window 按步长 1 滑窗，行数约等于交互数，{time.time() - t0:.1f}s）")
    # effective_keys：所有列都要跟着一起切。fold 必须带上——它在 pykt 的 ONE_KEYS 里，
    # 按标量逐行写出，切出来的 test_sequences.csv 少了它，KTDataset 按 fold 过滤时会 KeyError
    # （cidxs 是 pykt 用来对回原始行的索引，这份文件里已经有了）
    effective_keys = list(test.columns)
    t0 = time.time()
    seqs = generate_sequences(test, effective_keys, min_seq_len, maxlen)
    p = os.path.join(dpath, "test_sequences.csv")
    seqs.to_csv(p, index=False)
    print(f"[xes3g5m] test_sequences: {len(seqs)} 行，{os.path.getsize(p) / 1e6:.0f}MB"
          f"（{time.time() - t0:.1f}s）")

    t0 = time.time()
    wseqs = generate_window_sequences(test, effective_keys, maxlen)
    p = os.path.join(dpath, "test_window_sequences.csv")
    wseqs.to_csv(p, index=False)
    print(f"[xes3g5m] test_window_sequences: {len(wseqs)} 行，{os.path.getsize(p) / 1e6:.0f}MB"
          f"（{time.time() - t0:.1f}s）")

    # ---- data_config ----
    # num_q / num_c 取采样后数据里的最大 id + 1（embedding 表按它开大小，够用即可）
    num_q = max(_max_id(train["questions"]), _max_id(test["questions"])) + 1
    num_c = max(_max_id(train["concepts"]), _max_id(test["concepts"])) + 1
    dconfig = {
        "dpath"                    : dpath,
        "num_q"                    : num_q,
        "num_c"                    : num_c,
        "input_type"               : ["questions", "concepts"],
        "max_concepts"             : MAX_CONCEPTS,
        "min_seq_len"              : min_seq_len,
        "maxlen"                   : maxlen,
        "emb_path"                 : "",
        "train_valid_original_file": "train_valid.csv",  # 原始日志没给，占位（pykt 只在个别模型用）
        "train_valid_file"         : "train_valid_sequences.csv",
        "folds"                    : sorted(int(f) for f in train["fold"].unique()),
        "test_original_file"       : "test.csv",
        "test_file"                : "test_sequences.csv",
        "test_window_file"         : "test_window_sequences.csv",
    }
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    all_config = {}
    if os.path.exists(config_file):
        text = open(config_file).read().strip()
        all_config = json.loads(text) if text else {}
    all_config["xes3g5m"] = dconfig
    with open(config_file, "w") as fout:
        json.dump(all_config, fout, ensure_ascii=False, indent=4)
    print(f"[xes3g5m] num_q={num_q}, num_c={num_c} -> {config_file}")
    return dconfig

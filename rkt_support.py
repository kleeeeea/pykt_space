"""RKT 需要的 KC 关系矩阵 phi_array_{训练folds}.pkl。

pykt 的 train_model / evaluate 对 rkt 会读 {dpath}/phi_array{_1_2_3_4}.pkl（只用训练 folds 统计），
但 pykt 仓库里没有生成这个文件的脚本，所以这里按 RKT 论文（Pandey & Srivastava, CIKM 2020）
"基于作答表现的 exercise 关系" 的 phi 系数定义自己算：

  对每个学生序列里所有 "先做 i、后做 j" 的有序对（i 在前 j 在后，不要求相邻），
  按 (i 是否答对, j 是否答对) 统计 2x2 列联表 n11 n10 n01 n00，
      phi(i, j) = (n11*n00 - n10*n01) / sqrt((n11+n10)(n01+n00)(n11+n01)(n10+n00))
  分母为 0 时记 0。RKT 模型里再用 theta（默认 0）把 < theta 的关系置 0。

注意：这是我们按论文复现的版本（"有序对是否要求相邻" 论文没写死，这里取所有前后对），
和 RKT 原作者 / pykt 内部生成的文件可能有细节差异。assist2015 没有 question id，
RKT 在 num_q=0 时用 concept 当 item，所以矩阵按 KC 建，形状 (num_c, num_c)。

单独调试：
    rel = build_phi_array(dconfig, fold=0)          # 生成（已存在就直接读）
    rel.shape, rel.min(), rel.max()
"""
import os

import numpy as np
import pandas as pd


def phi_array_path(dconfig, fold):
    """和 pykt train_model / wandb_predict 里拼文件名的方式一致。"""
    train_folds = sorted(set(dconfig["folds"]) - {fold})
    folds_str = "_" + "_".join(str(f) for f in train_folds)
    return os.path.join(dconfig["dpath"], f"phi_array{folds_str}.pkl")


def count_ordered_pairs(concepts, responses, num_c):
    """一条序列里所有 i<j 的有序对，按 (c_i, c_j, r_i, r_j) 计数，返回长度 num_c*num_c*4 的计数向量。"""
    c = np.asarray(concepts)
    r = np.asarray(responses)
    keep = c != -1  # 去掉 padding
    c, r = c[keep], r[keep]
    a, b = np.triu_indices(len(c), k=1)  # a < b：a 在前，b 在后
    idx = ((c[a] * num_c + c[b]) * 2 + r[a]) * 2 + r[b]
    return np.bincount(idx, minlength=num_c * num_c * 4)


def phi_from_counts(counts, num_c):
    """counts[i, j, r_i, r_j] -> phi 系数矩阵 (num_c, num_c)。"""
    t = counts.reshape(num_c, num_c, 2, 2).astype(np.float64)
    n11, n10, n01, n00 = t[..., 1, 1], t[..., 1, 0], t[..., 0, 1], t[..., 0, 0]
    denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(denom > 0, (n11 * n00 - n10 * n01) / denom, 0.0)
    return phi


def build_phi_array(dconfig, fold):
    """用 train_valid 里除验证 fold 以外的序列算 phi 矩阵，存成 pykt 期望的 pkl；已存在且比数据新就直接读。"""
    out = phi_array_path(dconfig, fold)
    train_file = os.path.join(dconfig["dpath"], dconfig["train_valid_file"])
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(train_file):
        return pd.read_pickle(out)

    num_c = dconfig["num_c"]
    df = pd.read_csv(train_file)
    df = df[df["fold"] != fold]
    counts = np.zeros(num_c * num_c * 4, dtype=np.int64)
    for concepts, responses in zip(df["concepts"], df["responses"]):
        counts += count_ordered_pairs([int(x) for x in concepts.split(",")],
                                      [int(x) for x in responses.split(",")], num_c)
    phi = phi_from_counts(counts, num_c)
    pd.to_pickle(phi, out)
    print(f"[rkt] phi 关系矩阵 {phi.shape}，范围 [{phi.min():.3f}, {phi.max():.3f}]，"
          f"正相关占比 {(phi > 0).mean():.2%} -> {out}")
    return phi

"""DIMKT 的数据 / 训练 / 评估，供 run_pykt_demo.py 调用。

PyPI 上的 pykt-toolkit 0.0.38 没有 DIMKT：模型和 dataloader 从 pykt GitHub 复制到 vendor_pykt/，
而已安装的 pykt.models.train_model / evaluate 里也没有 dimkt 分支，所以这里按 GitHub 版
train_model.py / evaluate_model.py 里的 dimkt 分支写了等价的训练和评估循环。

每个函数都能单独调用调试，例如：
    loaders = build_dimkt_loaders(dconfig, fold=0, batch_size=64, difficult_levels=100)
    model = init_dimkt(model_config, dconfig)
    auc, acc = evaluate_dimkt(model, loaders["valid"])
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader

from vendor_pykt.dimkt import DIMKT
from vendor_pykt.dimkt_dataloader import DIMKTDataset, difficult_compute

device = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_difficulty_files(dconfig, difficult_levels):
    """预先生成 skills_difficult_{L}.csv / questions_difficult_{L}.csv。

    DIMKTDataset 发现文件不存在时会去读写死的 {dpath}/train_valid_sequences.csv 的 questions 列，
    assist2015 的这个文件没有 questions 列会 KeyError；这里改用 dconfig 里（带 questions 列的）
    train_valid_file 调同一个官方函数 difficult_compute 算难度。
    和官方一样用整个 train_valid（包含验证 fold）统计正确率。
    """
    dpath = dconfig["dpath"]
    sds_path = os.path.join(dpath, f"skills_difficult_{difficult_levels}.csv")
    qds_path = os.path.join(dpath, f"questions_difficult_{difficult_levels}.csv")
    train_file = os.path.join(dpath, dconfig["train_valid_file"])
    fresh = all(os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(train_file)
                for p in [sds_path, qds_path])
    if not fresh:
        print(f"[dimkt] 计算 KC / question 难度（{difficult_levels} 档）<- {train_file}")
        difficult_compute(pd.read_csv(train_file), sds_path, qds_path, diff_level=difficult_levels)
    return sds_path, qds_path


def build_dimkt_loaders(dconfig, fold, batch_size, difficult_levels, test_batch_size=1):
    """对应 pykt init_dataset4train + init_test_datasets 里的 dimkt 分支，返回 4 个 loader 的 dict。"""
    ensure_difficulty_files(dconfig, difficult_levels)
    dpath, input_type = dconfig["dpath"], dconfig["input_type"]
    all_folds = set(dconfig["folds"])

    def make(file_key, folds):
        return DIMKTDataset(dpath, os.path.join(dpath, dconfig[file_key]), input_type, folds,
                            diff_level=difficult_levels)

    return {
        "train"      : DataLoader(make("train_valid_file", all_folds - {fold}), batch_size=batch_size),
        "valid"      : DataLoader(make("train_valid_file", {fold}), batch_size=batch_size),
        "test"       : DataLoader(make("test_file", {-1}), batch_size=test_batch_size, shuffle=False),
        "test_window": DataLoader(make("test_window_file", {-1}), batch_size=test_batch_size, shuffle=False),
    }


def init_dimkt(model_config, dconfig):
    """对应 GitHub 版 init_model 的 dimkt 分支。"""
    return DIMKT(dconfig["num_q"], dconfig["num_c"], **model_config,
                 emb_type="qid", emb_path=dconfig["emb_path"]).to(device)


def load_dimkt(model_config, dconfig, ckpt_path, ckpt_name="qid_model.ckpt"):
    model = init_dimkt(model_config, dconfig)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name), map_location=device))
    return model


def _forward(model, dcur):
    """GitHub 版 train_model.model_forward / evaluate 里的 dimkt 分支：返回 (y, rshft, sm)。"""
    get = lambda k: dcur[k].to(device)
    q, c, r, sd, qd = get("qseqs"), get("cseqs"), get("rseqs"), get("sdseqs"), get("qdseqs")
    qshft, cshft, sdshft, qdshft = get("shft_qseqs"), get("shft_cseqs"), get("shft_sdseqs"), get("shft_qdseqs")
    rshft, sm = get("shft_rseqs"), get("smasks")
    y = model(q.long(), c.long(), sd.long(), qd.long(), r.long(),
              qshft.long(), cshft.long(), sdshft.long(), qdshft.long())
    return y, rshft, sm


def evaluate_dimkt(model, loader):
    """对应 pykt evaluate：只在 selectmask==1 的位置算 AUC 和 acc（阈值 0.5）。"""
    model.eval()
    ys, ts = [], []
    with torch.no_grad():
        for dcur in loader:
            y, rshft, sm = _forward(model, dcur)
            ys.append(torch.masked_select(y, sm).detach().cpu().numpy())
            ts.append(torch.masked_select(rshft, sm).detach().cpu().numpy())
    ps, ts = np.concatenate(ys), np.concatenate(ts)
    auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
    acc = metrics.accuracy_score(ts, (ps >= 0.5).astype(int))
    return auc, acc


def train_dimkt(model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                ckpt_name="qid_model.ckpt", patience=10):
    """对应 pykt train_model：每个 epoch 后在验证集上评估，AUC 变好就存 checkpoint，
    连续 patience 个 epoch 没提升就提前停。返回 (validauc, validacc, best_epoch)。"""
    best_auc, best_acc, best_epoch = -1, -1, -1
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for dcur in train_loader:
            y, rshft, sm = _forward(model, dcur)
            loss = binary_cross_entropy(torch.masked_select(y, sm).double(),
                                        torch.masked_select(rshft, sm).double())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        auc, acc = evaluate_dimkt(model, valid_loader)
        if auc > best_auc:
            best_auc, best_acc, best_epoch = auc, acc, epoch
            torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))
        print(f"Epoch: {epoch}, validauc: {auc:.4f}, validacc: {acc:.4f}, best epoch: {best_epoch}, "
              f"best auc: {best_auc:.4f}, train loss: {np.mean(losses):.4f}, model: dimkt")
        if epoch - best_epoch >= patience:
            break
    return best_auc, best_acc, best_epoch

"""AT-DKT 的数据 / 训练 / 评估，供 run_pykt_demo.py 调用。

PyPI 上的 pykt-toolkit 0.0.38 没有 AT-DKT：模型和 dataloader 从 pykt GitHub 复制到 vendor_pykt/，
已安装的 pykt.models.train_model / evaluate 里也没有 atdkt 分支，这里按 GitHub 版
train_model.py（model_forward + cal_loss）和 evaluate_model.py 里的 atdkt 分支写了等价的循环。

AT-DKT = DKT + 两个辅助任务，由 emb_type 字符串控制（官方 sweep 用 qiddelxembhistranscembpredcurc）：
  - predcurc：用 question 表示（transformer 编码）预测当前 KC，loss 权重 l2
  - his     ：从 start 步之后预测学生的历史正确率（MSE），loss 权重 l3
  总 loss = l1 * BCE(答对) + l2 * KC 分类 CE + l3 * 历史正确率 MSE

每个函数都能单独调用调试，例如：
    loaders = build_atdkt_loaders(dconfig, fold=0, batch_size=64)
    model = init_atdkt(model_config, dconfig)
    auc, acc = evaluate_atdkt(model, loaders["valid"])
"""
import os

import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy, one_hot
from torch.utils.data import DataLoader

from vendor_pykt.atdkt import ATDKT
from vendor_pykt.atdkt_dataloader import ATDKTDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_atdkt_loaders(dconfig, fold, batch_size, test_batch_size=1):
    """对应 GitHub 版 init_dataset4train + init_test_datasets 里的 atdkt 分支，返回 4 个 loader 的 dict。"""
    dpath, input_type = dconfig["dpath"], dconfig["input_type"]
    all_folds = set(dconfig["folds"])

    def make(file_key, folds):
        return ATDKTDataset(os.path.join(dpath, dconfig[file_key]), input_type, folds)

    return {
        "train"      : DataLoader(make("train_valid_file", all_folds - {fold}), batch_size=batch_size),
        "valid"      : DataLoader(make("train_valid_file", {fold}), batch_size=batch_size),
        "test"       : DataLoader(make("test_file", {-1}), batch_size=test_batch_size, shuffle=False),
        "test_window": DataLoader(make("test_window_file", {-1}), batch_size=test_batch_size, shuffle=False),
    }


def init_atdkt(model_config, dconfig):
    """对应 GitHub 版 init_model 的 atdkt 分支；emb_type 放在 model_config 里传。"""
    return ATDKT(dconfig["num_q"], dconfig["num_c"], **model_config,
                 emb_path=dconfig["emb_path"]).to(device)


def load_atdkt(model_config, dconfig, ckpt_path, ckpt_name="qid_model.ckpt"):
    model = init_atdkt(model_config, dconfig)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name), map_location=device))
    return model


def _to_device(dcur):
    """dataloader 里没有的字段（tseqs 等）是空 list，原样保留。"""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dcur.items()}


def _select_next(model, y, dcur):
    """y 是对所有 KC 的预测 [bs, len, num_c]，取下一步 KC 那一列 -> [bs, len]。
    对应官方 `if emb_type 里没有 bkt / addcshft: y = (y * one_hot(cshft)).sum(-1)`。"""
    if model.emb_type.find("bkt") == -1 and model.emb_type.find("addcshft") == -1:
        y = (y * one_hot(dcur["shft_cseqs"].long(), model.num_c)).sum(-1)
    return y


def _loss(model, y, y2, y3, dcur):
    """GitHub 版 cal_loss 的 atdkt 分支。"""
    sm, rshft = dcur["smasks"], dcur["shft_rseqs"]
    loss1 = binary_cross_entropy(torch.masked_select(y, sm).double(),
                                 torch.masked_select(rshft, sm).double())
    if model.emb_type.find("predcurc") != -1:
        if model.emb_type.find("his") != -1:
            return model.l1 * loss1 + model.l2 * y2 + model.l3 * y3
        return model.l1 * loss1 + model.l2 * y2
    if model.emb_type.find("predhis") != -1:
        return model.l1 * loss1 + model.l2 * y2
    return loss1


def evaluate_atdkt(model, loader):
    """对应 pykt evaluate：只在 selectmask==1 的位置算 AUC 和 acc（阈值 0.5）。"""
    model.eval()
    ys, ts = [], []
    with torch.no_grad():
        for dcur in loader:
            dcur = _to_device(dcur)
            y = _select_next(model, model(dcur), dcur)
            sm = dcur["smasks"]
            ys.append(torch.masked_select(y, sm).detach().cpu().numpy())
            ts.append(torch.masked_select(dcur["shft_rseqs"], sm).detach().cpu().numpy())
    ps, ts = np.concatenate(ys), np.concatenate(ts)
    auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
    acc = metrics.accuracy_score(ts, (ps >= 0.5).astype(int))
    return auc, acc


def train_atdkt(model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                ckpt_name="qid_model.ckpt", patience=10):
    """对应 pykt train_model：每个 epoch 后在验证集上评估，AUC 变好就存 checkpoint，
    连续 patience 个 epoch 没提升就提前停。返回 (validauc, validacc, best_epoch)。"""
    best_auc, best_acc, best_epoch = -1, -1, -1
    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for dcur in train_loader:
            dcur = _to_device(dcur)
            y, y2, y3 = model(dcur, train=True)
            loss = _loss(model, _select_next(model, y, dcur), y2, y3, dcur)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        auc, acc = evaluate_atdkt(model, valid_loader)
        if auc > best_auc:
            best_auc, best_acc, best_epoch = auc, acc, epoch
            torch.save(model.state_dict(), os.path.join(ckpt_path, ckpt_name))
        print(f"Epoch: {epoch}, validauc: {auc:.4f}, validacc: {acc:.4f}, best epoch: {best_epoch}, "
              f"best auc: {best_auc:.4f}, train loss: {np.mean(losses):.4f}, model: atdkt")
        if epoch - best_epoch >= patience:
            break
    return best_auc, best_acc, best_epoch

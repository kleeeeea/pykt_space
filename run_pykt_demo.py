"""自包含的 pykt 最小 demo：预处理 -> 训练 DKT -> 在测试集上评估。

只调用 pip 包 `pykt-toolkit` 暴露的库 API，不需要 clone GitHub 仓库里的 examples/。
流程与 quick_start.md.txt 一致：
  1. process_raw_data      对应 examples/data_preprocess.py 的第一步
  2. split_datasets.main   对应 examples/data_preprocess.py 的第二步（顺带写 data_config.json）
  3. init_model/train_model 对应 examples/wandb_dkt_train.py
  4. init_test_datasets/evaluate 对应 examples/wandb_predict.py

注意：assist2015 只有 concept（sequence_id）没有 question id，所以只能做 KC-level 评估，
quick_start 里的 early/late fusion（question-level）在这个数据集上不适用。
"""
import argparse
import json
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(HERE, "2015_100_skill_builders_main_problems.csv")


def prepare_raw(dpath, max_users):
    """把原始 csv 放到 data/{dataset_name}/ 下，max_users>0 时只取前 N 个用户加速 demo。"""
    os.makedirs(dpath, exist_ok=True)
    dst = os.path.join(dpath, os.path.basename(RAW_CSV))
    if max_users > 0:
        import pandas as pd

        df = pd.read_csv(RAW_CSV)
        keep = set(df["user_id"].drop_duplicates().head(max_users))
        sub = df[df["user_id"].isin(keep)]
        sub.to_csv(dst, index=False)
        print(f"[data] 采样 {len(keep)} 个用户 / {len(sub)} 条交互 -> {dst}")
    else:
        shutil.copyfile(RAW_CSV, dst)
        print(f"[data] 使用全量数据 -> {dst}")
    return dst


def preprocess(args, dpath, raw_csv, config_file):
    """预处理 + 切分 + 写 data_config.json。"""
    from pykt.preprocess import process_raw_data
    from pykt.preprocess.split_datasets import main as split_concept

    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    # write_config 会读取已有的 config 文件；缺 dataset key 时它会 KeyError，这里先补好
    if os.path.exists(config_file):
        with open(config_file) as fin:
            text = fin.read().strip()
        data_config = json.loads(text) if text else {}
    else:
        data_config = {}
    data_config.setdefault(args.dataset_name, {})
    with open(config_file, "w") as fout:
        json.dump(data_config, fout, ensure_ascii=False, indent=4)

    dname, writef = process_raw_data(args.dataset_name, {args.dataset_name: raw_csv})
    print(f"[preprocess] data.txt -> {writef}")
    split_concept(
        dname,
        writef,
        args.dataset_name,
        config_file,
        min_seq_len=args.min_seq_len,
        maxlen=args.maxlen,
        kfold=args.kfold,
    )
    print(f"[preprocess] data_config -> {config_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--workdir", type=str, default=os.path.join(HERE, "work"))
    parser.add_argument("--max_users", type=int, default=3000,
                        help="只取前 N 个用户跑 demo；0 表示用全量数据")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0, help="用作验证集的 fold")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--emb_size", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_preprocess", action="store_true",
                        help="即使已有切分结果也重新预处理")
    args = parser.parse_args()

    import torch

    # pykt 的 dataloader 里写死了 `from torch.cuda import FloatTensor, LongTensor`，
    # 在 CPU-only 机器上会报 "Torch not compiled with CUDA enabled"，所以导入前先打补丁
    if not torch.cuda.is_available():
        torch.cuda.FloatTensor = torch.FloatTensor
        torch.cuda.LongTensor = torch.LongTensor

    from pykt.datasets import init_dataset4train, init_test_datasets
    from pykt.models import evaluate, init_model, load_model, train_model
    from pykt.utils import set_seed

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__}, device={device}")

    dpath = os.path.join(args.workdir, "data", args.dataset_name)
    config_file = os.path.join(args.workdir, "configs", "data_config.json")

    # ---------- 1. 预处理 ----------
    done_flag = os.path.join(dpath, "test_sequences.csv")
    if args.force_preprocess or not os.path.exists(done_flag):
        raw_csv = prepare_raw(dpath, args.max_users)
        t0 = time.time()
        preprocess(args, dpath, raw_csv, config_file)
        print(f"[preprocess] 耗时 {time.time() - t0:.1f}s")
    else:
        print(f"[preprocess] 复用已有切分结果：{dpath}（要重跑加 --force_preprocess）")

    with open(config_file) as fin:
        data_config = json.load(fin)
    dconfig = data_config[args.dataset_name]
    print(f"[data] num_c={dconfig['num_c']}, num_q={dconfig['num_q']}, "
          f"input_type={dconfig['input_type']}")

    # ---------- 2. 训练 ----------
    train_loader, valid_loader, _, _ = init_dataset4train(
        args.dataset_name, args.model_name, data_config, args.fold, args.batch_size)

    model_config = {"emb_size": args.emb_size, "dropout": args.dropout}
    ckpt_path = os.path.join(args.workdir, "saved_model",
                             f"{args.dataset_name}_{args.model_name}_fold{args.fold}")
    os.makedirs(ckpt_path, exist_ok=True)

    model = init_model(args.model_name, model_config, dconfig, "qid")
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    t0 = time.time()
    (_, _, _, _, validauc, validacc, best_epoch) = train_model(
        model, train_loader, valid_loader, args.num_epochs, opt, ckpt_path,
        test_loader=None, test_window_loader=None, save_model=True)
    print(f"[train] 耗时 {time.time() - t0:.1f}s, best_epoch={best_epoch}, "
          f"validauc={validauc:.4f}, validacc={validacc:.4f}")

    # ---------- 3. 评估最优模型 ----------
    best_model = load_model(args.model_name, model_config, dconfig, "qid", ckpt_path)
    test_loader, test_window_loader, test_q_loader, test_qw_loader = init_test_datasets(
        dconfig, args.model_name, args.batch_size)

    testauc, testacc = evaluate(best_model, test_loader, args.model_name)
    wauc, wacc = evaluate(best_model, test_window_loader, args.model_name)

    print("=" * 60)
    print(f"dataset={args.dataset_name} model={args.model_name} fold={args.fold} "
          f"best_epoch={best_epoch}")
    print(f"valid       : auc={validauc:.4f} acc={validacc:.4f}")
    print(f"test        : auc={testauc:.4f} acc={testacc:.4f}")
    print(f"test window : auc={wauc:.4f} acc={wacc:.4f}")
    if test_q_loader is None:
        print("question-level(early/late fusion) 评估跳过：assist2015 无 question id")
    print(f"模型 checkpoint: {ckpt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

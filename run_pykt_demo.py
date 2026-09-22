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




# 论文里的名字和 pykt 内部 model_name 不一致的，单独映射
MODEL_ALIASES = {"hawkeskt": "hawkes"}


def normalize_model_name(name):
    """命令行里的写法 -> pykt 内部 model_name，例如 "DKT-Forget" -> "dkt_forget"，"HawkesKT" -> "hawkes"。"""
    name = name.strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(name, name)


# 伪时间戳里每一步对应的毫秒数。pykt 的 DktForgetDataset.calC 按 (t2-t1)/1000/60 换算成分钟，
# 设成 1 分钟 => rgap = round(log2(距上次同一 KC 的步数 + 1)) + 1
STEP_MS = 60 * 1000


def add_step_timestamps(src_csv, dst_csv):
    """给切分好的 sequences csv 补一列 timestamps（第 i 步 = i * STEP_MS，padding 位置 = -1）。

    assist2015 原始数据没有时间戳，而 DKT-Forget 的 rgap/sgap 都要从 timestamps 算，
    这里用交互序号代替真实时间：rgap 退化成"隔了几步"，sgap 恒定，pcount 不受影响。
    """
    import pandas as pd

    if os.path.exists(dst_csv) and os.path.getmtime(dst_csv) >= os.path.getmtime(src_csv):
        return dst_csv
    df = pd.read_csv(src_csv)

    def row_timestamps(concepts):
        return ",".join("-1" if c == "-1" else str(i * STEP_MS)
                        for i, c in enumerate(concepts.split(",")))

    df["timestamps"] = df["concepts"].map(row_timestamps)
    df.to_csv(dst_csv, index=False)
    print(f"[dkt_forget] 补伪时间戳 -> {dst_csv}")
    return dst_csv


def with_step_timestamps(data_config, dataset_name):
    """复制一份 data_config，把 train_valid/test/test_window 文件换成带伪时间戳的版本。

    返回 (new_data_config, new_dconfig)。用副本是因为 init_dataset4train 会往 dconfig 里
    写 num_rgap/num_sgap/num_pcount，不想污染其它模型用的那份。
    """
    import copy

    new_data_config = copy.deepcopy(data_config)
    dconfig = new_data_config[dataset_name]
    if "timestamps" in dconfig.get("input_type", []):
        return new_data_config, dconfig  # 数据集本身有真实时间戳，不用造
    for key in ["train_valid_file", "test_file", "test_window_file"]:
        src = os.path.join(dconfig["dpath"], dconfig[key])
        dst_name = dconfig[key].replace(".csv", "_steptime.csv")
        add_step_timestamps(src, os.path.join(dconfig["dpath"], dst_name))
        dconfig[key] = dst_name
    return new_data_config, dconfig


def add_concepts_as_questions(src_csv, dst_csv):
    """给 sequences csv 补一列 questions，直接复制 concepts（每个 KC 当成一道"题"）。"""
    import pandas as pd

    if os.path.exists(dst_csv) and os.path.getmtime(dst_csv) >= os.path.getmtime(src_csv):
        return dst_csv
    df = pd.read_csv(src_csv)
    df["questions"] = df["concepts"]
    df.to_csv(dst_csv, index=False)
    print(f"[pseudo_q] 用 concepts 充当 questions -> {dst_csv}")
    return dst_csv


def with_hawkes_inputs(data_config, dataset_name):
    """HawkesKT 同时需要 questions 和 timestamps；assist2015 两个都没有，这里都造出来。

    - timestamps：复用 with_step_timestamps 的伪时间戳（第 i 步 = i 分钟）
    - questions ：没有 question id 时用 concept 充当，num_q = num_c，
                  这样 HawkesKT 的 problem_base 和 skill_base 等价于同一个 KC 偏置学了两份
    返回 (new_data_config, new_dconfig)，不改动传进来的 data_config。
    """
    new_data_config, _ = with_step_timestamps(data_config, dataset_name)
    return with_concepts_as_questions(new_data_config, dataset_name)


def with_concepts_as_questions(data_config, dataset_name):
    """复制一份 data_config；没有 question id 时把 sequences 换成带 questions 列（= concepts）的版本，
    并设 input_type=["questions","concepts"]、num_q=num_c。返回 (new_data_config, new_dconfig)。"""
    import copy

    new_data_config = copy.deepcopy(data_config)
    dconfig = new_data_config[dataset_name]
    if dconfig["num_q"] > 0:
        return new_data_config, dconfig  # 数据集本身有 question id
    for key in ["train_valid_file", "test_file", "test_window_file"]:
        src = os.path.join(dconfig["dpath"], dconfig[key])
        dst_name = dconfig[key].replace(".csv", "_q.csv")
        add_concepts_as_questions(src, os.path.join(dconfig["dpath"], dst_name))
        dconfig[key] = dst_name
    dconfig["input_type"] = ["questions", "concepts"]
    dconfig["num_q"] = dconfig["num_c"]
    return new_data_config, dconfig


def with_quelevel_inputs(data_config, dataset_name):
    """question 级模型（pykt 的 que_type_models：iekt / qdkt）走 KTQueDataset，读 *_file_quelevel。

    pykt 正常是用 split_datasets_que 另外切一份 question 级文件；这里 question == concept，
    每道"题"只有 1 个 KC（max_concepts=1），question 级序列和 with_concepts_as_questions 生成的
    *_q.csv 完全一样（KTQueDataset 按 "_" 拆多 KC，单 KC 时就是一个数），所以直接指过去。
    """
    new_data_config, dconfig = with_concepts_as_questions(data_config, dataset_name)
    if "train_valid_file_quelevel" in dconfig:
        return new_data_config, dconfig  # 数据集本身已有 question 级切分
    assert dconfig["max_concepts"] == 1, "多 KC 的题需要 pykt split_datasets_que 重新切分"
    for key in ["train_valid_file", "test_file", "test_window_file"]:
        dconfig[f"{key}_quelevel"] = dconfig[key]
    return new_data_config, dconfig


def with_lpkt_inputs(data_config, dataset_name):
    """LPKT 需要 questions + Q 矩阵（question -> KC）。

    - questions：用 concept 充当（同 HawkesKT）
    - Q 矩阵  ：pykt 的 generate_qmatrix 写死读原始 train_valid.csv/test.csv 的 questions 列，
                assist2015 没有这一列会 KeyError；question == concept 时 Q 矩阵就是单位阵，
                这里直接生成 qmatrix.npz，init_model 发现文件已存在就不会再调 generate_qmatrix
    - 时间     ：不造。没有 timestamps 时 LPKTDataset 把 interval time 全设成 1；answer time 训练时本来就没用
    """
    import numpy as np

    pseudo_q = data_config[dataset_name]["num_q"] == 0
    new_data_config, dconfig = with_concepts_as_questions(data_config, dataset_name)
    qmatrix_path = os.path.join(dconfig["dpath"], "qmatrix.npz")
    if pseudo_q and not os.path.exists(qmatrix_path):
        n = dconfig["num_c"]
        np.savez(qmatrix_path, matrix=np.eye(n + 1))  # 形状 (num_q+1, num_c+1)，和 generate_qmatrix 一致
        print(f"[lpkt] question == concept，写单位阵 Q 矩阵 -> {qmatrix_path}")
    return new_data_config, dconfig


finished_models = "dkt,dkt,DKT-Forget,KQN,DKVMN,GKT,HawkesKT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str,
                        # iterate over dkt+ and dkt, keep the  model_config consistent

                        default="Deep-IRT,LPKT,DIMKT,IEKT,AT-DKT",
                        help="逗号分隔的模型列表，依次训练+评估，共用同一份 emb_size/dropout")
    # dkt+ 额外的正则项系数（DKTPlus 构造函数必填），默认值取自 pykt examples/wandb_dkt_plus_train.py
    parser.add_argument("--lambda_r", type=float, default=0.01)
    parser.add_argument("--lambda_w1", type=float, default=0.003)
    parser.add_argument("--lambda_w2", type=float, default=3.0)
    # KQN 没有 emb_size，用 n_hidden（知识状态/技能向量维度）对应 emb_size；下面两个 0 表示也跟 emb_size 一致
    parser.add_argument("--n_rnn_hidden", type=int, default=0)
    parser.add_argument("--n_mlp_hidden", type=int, default=0)
    # DKVMN 用 dim_s（key/value 向量维度）对应 emb_size；size_m 是记忆槽（潜在概念）个数
    parser.add_argument("--size_m", type=int, default=50)
    # ATKT 用 skill_dim/answer_dim/hidden_dim 都对应 emb_size；下面三个是对抗训练相关，默认值取自 ATKT 构造函数
    parser.add_argument("--attention_dim", type=int, default=80)
    parser.add_argument("--epsilon", type=float, default=10, help="ATKT 对抗扰动的 L2 大小")
    parser.add_argument("--beta", type=float, default=0.2, help="ATKT 对抗 loss 的权重")
    # GKT 的 emb_size 直接沿用，hidden_dim 也取 emb_size；graph_type 决定 KC 图怎么建
    parser.add_argument("--graph_type", type=str, default="dense", choices=["dense", "transition"],
                        help="GKT 概念图：dense=全连接均匀图，transition=按训练序列里 KC 相邻出现的频次统计")
    # HawkesKT 的 emb_size 直接沿用；time_log 是时间间隔取对数的底：delta_t = log(秒数)/log(time_log)
    parser.add_argument("--time_log", type=float, default=5)
    # LPKT 的 d_a/d_e/d_k 都对应 emb_size；gamma 是 Q 矩阵里非相关 KC 的填充值，默认值取自 LPKT 构造函数
    parser.add_argument("--gamma", type=float, default=0.03)
    # DIMKT 把 KC / 题目的正确率离散成多少档难度，默认值取自 pykt examples/wandb_dimkt_train.py
    parser.add_argument("--difficult_levels", type=int, default=100)
    parser.add_argument("--retrain", action="store_true",
                        help="忽略已有 checkpoint 缓存，强制重新训练")
    parser.add_argument("--workdir", type=str, default=os.path.join(HERE, "work"))
    parser.add_argument("--max_users", type=int, default=3000,
                        help="只取前 N 个用户跑 demo；0 表示用全量数据")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0, help="用作验证集的 fold")
    parser.add_argument("--batch_size", type=int, default=64)
    # 64
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--emb_size", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_preprocess", action="store_true",
                        help="即使已有切分结果也重新预处理")
    parser.add_argument("--export_dir", type=str, default="",
                        help="预测明细 csv 的输出目录，默认 {workdir}/pred")
    parser.add_argument("--no_export", action="store_true",
                        help="不导出每一步的 输入/预测/label 明细 csv")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="每个 loader 只取前 N 个 batch（冒烟测试，证明能跑通）；0 表示用全部数据；"
                             "不传时按模型取默认值，见 DEFAULT_MAX_BATCHES")
    args = parser.parse_args()

    import torch

    # pykt 的 dataloader 里写死了 `from torch.cuda import FloatTensor, LongTensor`，
    # 在 CPU-only 机器上会报 "Torch not compiled with CUDA enabled"，所以导入前先打补丁
    if not torch.cuda.is_available():
        torch.cuda.FloatTensor = torch.FloatTensor
        torch.cuda.LongTensor = torch.LongTensor

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
    # 100
    with open(config_file) as fin:
        data_config = json.load(fin)
    dconfig = data_config[args.dataset_name]
    # json 里存的是绝对路径，项目挪过目录后会失效，这里强制用当前 workdir 下的路径
    dconfig["dpath"] = dpath
    print(f"[data] num_c={dconfig['num_c']}, num_q={dconfig['num_q']}, "
          f"input_type={dconfig['input_type']}")

    # ---------- 2+3. 依次训练 + 评估每个模型 ----------
    model_names = [normalize_model_name(m) for m in args.model_name.split(",") if m.strip()]
    # 所有模型共用的超参，保证 dkt / dkt+ 的对比是公平的
    base_model_config = {"emb_size": args.emb_size, "dropout": args.dropout}
    results = []
    for model_name in model_names:
        set_seed(args.seed)  # 每个模型用同样的随机种子初始化
        results.append(train_and_eval(args, model_name, base_model_config, data_config, dconfig))

    # ---------- 4. 汇总 ----------
    print("=" * 60)
    print(f"dataset={args.dataset_name} fold={args.fold} 共用 model_config={base_model_config}")
    print(f"{'model':<8}{'best_ep':>8}{'valid_auc':>11}{'test_auc':>10}{'test_acc':>10}"
          f"{'win_auc':>9}{'win_acc':>9}")
    for res in results:
        print(f"{res['model_name']:<8}{res['best_epoch']:>8}{res['validauc']:>11.4f}"
              f"{res['testauc']:>10.4f}{res['testacc']:>10.4f}"
              f"{res['wauc']:>9.4f}{res['wacc']:>9.4f}")
    print("=" * 60)


# 不传 --max_batches 时各模型的默认值；gkt 在 CPU 上一个 batch 要 20s+，默认只跑 1 个 batch 证明能跑通
DEFAULT_MAX_BATCHES = {"gkt": 1}


def resolve_max_batches(args, model_name):
    """命令行显式传了 --max_batches 就用它，否则按模型查 DEFAULT_MAX_BATCHES（查不到 = 0 = 全量）。"""
    if args.max_batches is not None:
        return args.max_batches
    return DEFAULT_MAX_BATCHES.get(model_name, 0)


def limit_loader(loader, max_batches):
    """只保留 loader 的前 max_batches 个 batch（按原顺序，不打乱），max_batches<=0 时原样返回。

    用 Subset 重新包一个 DataLoader，而不是 islice，因为 pykt 的 train/evaluate 会多次遍历 loader。
    """
    if loader is None or max_batches <= 0:
        return loader
    from torch.utils.data import DataLoader, Subset

    n = min(len(loader.dataset), max_batches * loader.batch_size)
    return DataLoader(Subset(loader.dataset, range(n)), batch_size=loader.batch_size, shuffle=False)


def build_model_config(args, model_name, base_model_config):
    """在共用的 base_model_config 上补各模型必需的专属参数（dkt+ 需要三个正则系数）。"""
    model_config = dict(base_model_config)
    if model_name == "dkt+":
        model_config.update({"lambda_r" : args.lambda_r,
                             "lambda_w1": args.lambda_w1,
                             "lambda_w2": args.lambda_w2})
    elif model_name == "kqn":
        # KQN(n_skills, n_hidden, n_rnn_hidden, n_mlp_hidden, dropout, ...) 不接受 emb_size
        emb_size = model_config.pop("emb_size")
        model_config.update({"n_hidden"    : emb_size,
                             "n_rnn_hidden": args.n_rnn_hidden or emb_size,
                             "n_mlp_hidden": args.n_mlp_hidden or emb_size})
    elif model_name in ["dkvmn", "deep_irt"]:
        # DKVMN / DeepIRT(num_c, dim_s, size_m, dropout, ...) 不接受 emb_size；DeepIRT = DKVMN + IRT 输出层
        model_config.update({"dim_s" : model_config.pop("emb_size"),
                             "size_m": args.size_m})
    elif model_name in ["atkt", "atktfix"]:
        # ATKT(num_c, skill_dim, answer_dim, hidden_dim, attention_dim, epsilon, beta, dropout, ...) 不接受 emb_size
        emb_size = model_config.pop("emb_size")
        model_config.update({"skill_dim"    : emb_size,
                             "answer_dim"   : emb_size,
                             "hidden_dim"   : emb_size,
                             "attention_dim": args.attention_dim,
                             "epsilon"      : args.epsilon,
                             "beta"         : args.beta})
    elif model_name == "gkt":
        # GKT(num_c, hidden_dim, emb_size, graph_type, graph, dropout, ...)；init_model 会读 graph_type 建图
        model_config.update({"hidden_dim": model_config["emb_size"],
                             "graph_type": args.graph_type})
    elif model_name == "hawkes":
        # HawkesKT(n_skills, n_problems, emb_size, time_log) 不接受 dropout
        model_config.pop("dropout")
        model_config["time_log"] = args.time_log
    elif model_name == "lpkt":
        # LPKT(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, gamma, dropout, q_matrix, ...) 不接受 emb_size
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_a"  : emb_size,
                             "d_e"  : emb_size,
                             "d_k"  : emb_size,
                             "gamma": args.gamma})
    elif model_name == "dimkt":
        # DIMKT(num_q, num_c, dropout, emb_size, batch_size, num_steps, difficult_levels, ...)
        # num_steps 官方默认 199 = maxlen - 1；batch_size 只是初始值，forward 里会按实际 batch 改
        model_config.update({"batch_size"      : args.batch_size,
                             "num_steps"       : args.maxlen - 1,
                             "difficult_levels": args.difficult_levels})
    return model_config


CKPT_NAME = "qid_model.ckpt"  # pykt train_model 保存的文件名：{emb_type}_model.ckpt
META_NAME = "train_meta.json"  # 我们自己记录的训练配置 + 验证集指标


def train_signature(args, model_name, model_config, max_batches):
    """决定 checkpoint 能否复用的训练配置；任何一项变了都要重新训练。"""
    return {"model_name"   : model_name,
            "model_config" : model_config,
            "num_epochs"   : args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size"   : args.batch_size,
            "seed"         : args.seed,
            "fold"         : args.fold,
            "max_users"    : args.max_users,
            "maxlen"       : args.maxlen,
            "min_seq_len"  : args.min_seq_len,
            "max_batches"  : max_batches}


def load_train_cache(ckpt_path, signature):
    """返回 (是否可复用, meta)。

    - 有 ckpt + meta 且 signature 一致   -> 复用，meta 里带 validauc/best_epoch
    - 有 ckpt 但没有 meta（加缓存之前训的旧模型，比如 Deep-IRT）-> 也复用，但训练配置无法核对，meta=None
    - 其它情况                              -> 重新训练
    """
    if not os.path.exists(os.path.join(ckpt_path, CKPT_NAME)):
        return False, None
    meta_file = os.path.join(ckpt_path, META_NAME)
    if not os.path.exists(meta_file):
        return True, None
    with open(meta_file) as fin:
        meta = json.load(fin)
    # 过一遍 json 再比较，避免 tuple/list、int/float 之类的表示差异
    return meta.get("signature") == json.loads(json.dumps(signature)), meta


def save_train_cache(ckpt_path, signature, validauc, validacc, best_epoch):
    with open(os.path.join(ckpt_path, META_NAME), "w") as fout:
        json.dump({"signature": signature, "validauc": validauc, "validacc": validacc,
                   "best_epoch": best_epoch}, fout, ensure_ascii=False, indent=4)


def fix_hawkes_on_cpu(model):
    """pykt 的 HawkesKT.forward 里写的是 `mask.cuda() if self.gpu != ''`，
    而 self.gpu 是 torch.device("cpu")，永远 != ''，CPU 上会报错；这里把它置空。"""
    import torch

    if getattr(model, "model_name", "") == "hawkes" and not torch.cuda.is_available():
        model.gpu = ""
    return model


# ---------- 模型相关操作的统一入口：dimkt 走 dimkt_support.py，其它走已安装的 pykt ----------
# PyPI 版 pykt 没有 DIMKT，它的 train_model / evaluate / init_model / 数据加载都不认 "dimkt"

def kt_train_loaders(args, model_name, data_config, dconfig):
    """返回 (train_loader, valid_loader)。"""
    if model_name == "dimkt":
        from dimkt_support import build_dimkt_loaders
        loaders = build_dimkt_loaders(dconfig, args.fold, args.batch_size, args.difficult_levels)
        return loaders["train"], loaders["valid"]
    from pykt.datasets import init_dataset4train
    train_loader, valid_loader, _, _ = init_dataset4train(
            args.dataset_name, model_name, data_config, args.fold, args.batch_size)
    return train_loader, valid_loader


def kt_test_loaders(args, model_name, dconfig):
    """返回 (test_loader, test_window_loader, test_q_loader)；test_q_loader 为 None 表示没有 question-level 评估。"""
    if model_name == "dimkt":
        from dimkt_support import build_dimkt_loaders
        loaders = build_dimkt_loaders(dconfig, args.fold, args.batch_size, args.difficult_levels)
        return loaders["test"], loaders["test_window"], None
    from pykt.datasets import init_test_datasets
    test_loader, test_window_loader, test_q_loader, _ = init_test_datasets(
            dconfig, model_name,
            1,
            # args.batch_size
    )
    return test_loader, test_window_loader, test_q_loader


def kt_init_model(model_name, model_config, dconfig):
    if model_name == "dimkt":
        from dimkt_support import init_dimkt
        return init_dimkt(model_config, dconfig)
    from pykt.models import init_model
    return fix_hawkes_on_cpu(init_model(model_name, model_config, dconfig, "qid"))


def kt_load_model(model_name, model_config, dconfig, ckpt_path):
    if model_name == "dimkt":
        from dimkt_support import load_dimkt
        return load_dimkt(model_config, dconfig, ckpt_path, CKPT_NAME)
    from pykt.models import load_model
    return fix_hawkes_on_cpu(load_model(model_name, model_config, dconfig, "qid", ckpt_path))


def kt_train_model(model_name, model, train_loader, valid_loader, num_epochs, opt, ckpt_path):
    """返回 (validauc, validacc, best_epoch)；最优模型存到 ckpt_path/CKPT_NAME。"""
    if model_name == "dimkt":
        from dimkt_support import train_dimkt
        return train_dimkt(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, CKPT_NAME)
    from pykt.models import train_model
    (_, _, _, _, validauc, validacc, best_epoch) = train_model(
            model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
            test_loader=None, test_window_loader=None, save_model=True)
    return validauc, validacc, best_epoch


def kt_evaluate(model_name, model, loader):
    """返回 (auc, acc)。"""
    if model_name == "dimkt":
        from dimkt_support import evaluate_dimkt
        return evaluate_dimkt(model, loader)
    from pykt.models import evaluate
    return evaluate(model, loader, model_name)


def train_and_eval(args, model_name, base_model_config, data_config, dconfig):
    """单个模型：训练 -> 加载最优 checkpoint -> 导出预测 -> 评估，返回指标 dict。"""
    import torch

    print("-" * 60)
    print(f"[model] {model_name}")
    if model_name == "dkt_forget":
        # DKT-Forget 需要 timestamps 列；assist2015 没有，换成带伪时间戳的 sequences 文件
        data_config, dconfig = with_step_timestamps(data_config, args.dataset_name)
    elif model_name == "hawkes":
        # HawkesKT 需要 questions + timestamps；assist2015 都没有，用 concept 和伪时间戳代替
        data_config, dconfig = with_hawkes_inputs(data_config, args.dataset_name)
    elif model_name == "lpkt":
        # LPKT 需要 questions + Q 矩阵；assist2015 没有 question id，用 concept 充当
        data_config, dconfig = with_lpkt_inputs(data_config, args.dataset_name)
    elif model_name == "dimkt":
        # DIMKT 需要 question id 来算题目难度；assist2015 没有，用 concept 充当（题目难度 = KC 难度）
        data_config, dconfig = with_concepts_as_questions(data_config, args.dataset_name)
    elif model_name in ["iekt", "qdkt"]:
        # question 级模型读 *_file_quelevel；assist2015 没有 question id，用 concept 充当
        data_config, dconfig = with_quelevel_inputs(data_config, args.dataset_name)

    # ---------- 2. 训练 ----------
    train_loader, valid_loader = kt_train_loaders(args, model_name, data_config, dconfig)
    max_batches = resolve_max_batches(args, model_name)
    if max_batches > 0:
        print(f"[data] {model_name} 每个 loader 只取前 {max_batches} 个 batch（冒烟测试，指标无参考意义）")
    train_loader = limit_loader(train_loader, max_batches)
    valid_loader = limit_loader(valid_loader, max_batches)

    model_config = build_model_config(args, model_name, base_model_config)
    ckpt_path = os.path.join(args.workdir, "saved_model",
                             f"{args.dataset_name}_{model_name}_fold{args.fold}")
    os.makedirs(ckpt_path, exist_ok=True)

    # 注意：init_dataset4train 上面必须照常调用，它会往 dconfig 里写 num_at/num_rgap 等 init_model 要用的字段
    signature = train_signature(args, model_name, model_config, max_batches)
    reuse, meta = (False, None) if args.retrain else load_train_cache(ckpt_path, signature)
    best_model = None
    if reuse:
        try:
            best_model = kt_load_model(model_name, model_config, dconfig, ckpt_path)
        except RuntimeError as e:  # 旧 checkpoint 和当前 model_config 形状对不上
            print(f"[cache] checkpoint 加载失败，重新训练：{str(e).splitlines()[0]}")

    if best_model is not None and meta is not None:
        validauc, validacc, best_epoch = meta["validauc"], meta["validacc"], meta["best_epoch"]
        print(f"[cache] 训练配置一致，复用 checkpoint：{ckpt_path}（要重训加 --retrain）")
    elif best_model is not None:
        # 加缓存之前训练的旧 checkpoint：没有 meta，只能重新算一遍验证集指标
        validauc, validacc = kt_evaluate(model_name, best_model, valid_loader)
        best_epoch = -1
        print(f"[cache] 复用旧 checkpoint（无 {META_NAME}，训练配置无法核对，best_epoch 未知）：{ckpt_path}"
              f"（要重训加 --retrain）")
    else:
        model = kt_init_model(model_name, model_config, dconfig)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        t0 = time.time()
        validauc, validacc, best_epoch = kt_train_model(
                model_name, model, train_loader, valid_loader, args.num_epochs, opt, ckpt_path)
        print(f"[train] 耗时 {time.time() - t0:.1f}s, best_epoch={best_epoch}, "
              f"validauc={validauc:.4f}, validacc={validacc:.4f}")
        save_train_cache(ckpt_path, signature, validauc, validacc, best_epoch)

        # ---------- 3. 评估最优模型 ----------
        best_model = kt_load_model(model_name, model_config, dconfig, ckpt_path)
    test_loader, test_window_loader, test_q_loader = kt_test_loaders(args, model_name, dconfig)
    test_loader = limit_loader(test_loader, max_batches)
    test_window_loader = limit_loader(test_window_loader, max_batches)

    # 能不能导出模型具体的input 和output 以及label，做成一个csv
    testauc, testacc = kt_evaluate(model_name, best_model, test_loader)
    wauc, wacc = kt_evaluate(model_name, best_model, test_window_loader)

    print(f"dataset={args.dataset_name} model={model_name} fold={args.fold} "
          f"best_epoch={best_epoch} model_config={model_config}")
    print(f"valid       : auc={validauc:.4f} acc={validacc:.4f}")
    print(f"test        : auc={testauc:.4f} acc={testacc:.4f}")
    print(f"test window : auc={wauc:.4f} acc={wacc:.4f}")
    if test_q_loader is None:
        print("question-level(early/late fusion) 评估跳过：assist2015 无 question id")
    print(f"模型 checkpoint: {ckpt_path}")

    return {"model_name": model_name, "best_epoch": best_epoch,
            "validauc"  : validauc, "validacc": validacc,
            "testauc"   : testauc, "testacc": testacc, "wauc": wauc, "wacc": wacc}


if __name__ == "__main__":
    main()

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

# 各数据集的原始文件在哪、按哪一列采样用户、什么分隔符。
# 名字必须是 pykt 认的（见 pykt/preprocess/data_proprocess.py 里的分支），预处理按名字选对应的解析器。
# 除 assist2015 外，原始数据都用 data/run.sh 下载，路径相对本目录。
RAW_DATASETS = {
    "assist2015" : {"raw"     : "2015_100_skill_builders_main_problems.csv",
                    "user_col": "user_id"},
    "assist2009" : {"raw"     : "data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv",
                    "user_col": "user_id",
                    # 这份 csv 有非 utf-8 字节，pykt 的解析器自己用 dtype=str 读，这里采样也要对齐
                    "encoding": "ISO-8859-1"},
    "assist2017" : {"raw"     : "data/anonymized_full_release_competition_dataset/"
                                "anonymized_full_release_competition_dataset.csv",
                    "user_col": "studentId"},
    "algebra2005": {"raw"     : "data/algebra_2005_2006/algebra_2005_2006/algebra_2005_2006_train.txt",
                    "user_col": "Anon Student Id",
                    "sep"     : "\t"},  # KDD Cup 的数据是制表符分隔，pykt 用 read_table 读
    # nips_task34 的解析器还要读同目录下的 metadata/，所以 extra_dirs 里的目录要一并拷过去
    "nips_task34": {"raw"       : "data/NIPS2020/public_data/train_data/train_task_3_4.csv",
                    "user_col"  : "UserId",
                    "extra_dirs": ["data/NIPS2020/public_data/metadata"]},
    # xes3g5m 不在这张表里：它给的已经是切好的序列，不需要原始日志预处理，见 xes3g5m_support.py
}
# XES3G5M 用 KC 级那套（和其它数据集的 KC 级评估口径一致）；question_level/ 暂未接
XES3G5M_KC_LEVEL = "data/XES3G5M/XES3G5M/kc_level"


def prepare_raw(dataset_name, dpath, max_users):
    """把原始数据放到 work/data/{dataset_name}/ 下，max_users>0 时只取前 N 个用户加速 demo。"""
    if dataset_name not in RAW_DATASETS:
        raise KeyError(f"没有配 {dataset_name} 的原始数据，先在 RAW_DATASETS 里加一条；"
                       f"已配好的：{sorted(RAW_DATASETS)}")
    spec = RAW_DATASETS[dataset_name]
    src = os.path.join(HERE, spec["raw"])
    if not os.path.exists(src):
        raise FileNotFoundError(f"原始数据不在 {src}，先跑 data/run.sh 下载")
    sep, enc = spec.get("sep", ","), spec.get("encoding")
    os.makedirs(dpath, exist_ok=True)
    dst = os.path.join(dpath, os.path.basename(src))

    # 注意：写出去一律用 utf-8。pykt 的各个解析器都是按 utf-8 读的（assist2009 那份原始 csv
    # 是 ISO-8859-1，照原样拷过去会 UnicodeDecodeError），所以这里顺便做一次编码转换
    if max_users > 0 or enc:
        import pandas as pd

        df = pd.read_csv(src, sep=sep, encoding=enc, low_memory=False)
        if max_users > 0:
            keep = set(df[spec["user_col"]].drop_duplicates().head(max_users))
            df = df[df[spec["user_col"]].isin(keep)]
            print(f"[data] 采样 {len(keep)} 个用户 / {len(df)} 条交互 -> {dst}")
        else:
            print(f"[data] 使用全量数据（转 utf-8）-> {dst}")
        df.to_csv(dst, sep=sep, index=False)
    else:
        shutil.copyfile(src, dst)
        print(f"[data] 使用全量数据 -> {dst}")

    # 有的解析器（nips_task34）还要读原始数据旁边的 metadata 目录
    for rel in spec.get("extra_dirs", []):
        extra_dst = os.path.join(dpath, os.path.basename(rel))
        if not os.path.exists(extra_dst):
            shutil.copytree(os.path.join(HERE, rel), extra_dst)
            print(f"[data] 附带目录 -> {extra_dst}")
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
MODEL_ALIASES = {"hawkeskt": "hawkes", "at_dkt": "atdkt", "lefokt": "lefokt_akt", "moc_kt": "mockt"}


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


def with_denoisekt_inputs(data_config, dataset_name):
    """DenoiseKT 是 question 级模型，还要 {dpath}/questions_concepts.pt：一张题目-题目稀疏邻接图
    （两道题共享 KC 就相连），GCN 用它聚合题目表示。

    官方只给了 Google Drive 上的生成脚本，仓库里没有。assist2015 用 concept 充当 question，
    "共享 KC" 退化成"只和自己相连"，所以这里直接生成单位阵；也就是说这个模型的图部分在
    assist2015 上不起作用（GCN 退化成一次线性变换），去噪注意力部分不受影响。
    """
    import torch

    pseudo_q = data_config[dataset_name]["num_q"] == 0
    new_data_config, dconfig = with_quelevel_inputs(data_config, dataset_name)
    qs_cs_path = os.path.join(dconfig["dpath"], "questions_concepts.pt")
    if pseudo_q and not os.path.exists(qs_cs_path):
        n = dconfig["num_q"]
        torch.save(torch.eye(n).to_sparse(), qs_cs_path)  # 形状 (num_q, num_q)，GCN 里做 sparse.mm(adj, x)
        print(f"[denoisekt] question == concept，题目图退化成单位阵 ({n}x{n}) -> {qs_cs_path}")
    return new_data_config, dconfig


def with_lpkt_inputs(data_config, dataset_name):
    """LPKT 需要 questions + Q 矩阵（question -> KC）。

    - questions：用 concept 充当（同 HawkesKT）
    - Q 矩阵  ：pykt 的 generate_qmatrix 写死读原始 train_valid.csv/test.csv 的 questions 列，
                assist2015 没有这一列会 KeyError；question == concept 时 Q 矩阵就是单位阵，
                这里直接生成 qmatrix.npz，init_model 发现文件已存在就不会再调 generate_qmatrix
    - 时间     ：不造。没有 timestamps 时 LPKTDataset 把 interval time 全设成 1；answer time 训练时本来就没用

    注意：pykt_gh 的 init_dataset4train / init_test_datasets 里 lpkt 读的是 *_file_quelevel，
    所以这里用 with_quelevel_inputs（会同时设好 *_file 和 *_file_quelevel）。
    """
    import numpy as np

    pseudo_q = data_config[dataset_name]["num_q"] == 0
    new_data_config, dconfig = with_quelevel_inputs(data_config, dataset_name)
    qmatrix_path = os.path.join(dconfig["dpath"], "qmatrix.npz")
    if pseudo_q and not os.path.exists(qmatrix_path):
        n = dconfig["num_c"]
        np.savez(qmatrix_path, matrix=np.eye(n + 1))  # 形状 (num_q+1, num_c+1)，和 generate_qmatrix 一致
        print(f"[lpkt] question == concept，写单位阵 Q 矩阵 -> {qmatrix_path}")
    return new_data_config, dconfig


# 已经接通并实测跑通的模型，按接入顺序排列；下面 --model_name 的默认值就是它
# 跳过的：HCGKT（官方没提供 assist2015 的题目-KC 图，且该 commit 的 train_model 分支引用未定义变量）
#         OPERA（需要题目文本 / 选项解释，assist2015 只有 user_id,log_id,sequence_id,correct）
finished_models = ("dkt,dkt+,DKT-Forget,KQN,SAKT,AKT,DKVMN,SKVMN,ATKT,GKT,HawkesKT,Deep-IRT,LPKT,DIMKT,IEKT,ATDKT,"
                   "simpleKT,QIKT,sparseKT-soft,sparseKT-topK,RKT,FoLiBiKT,Dtransformer,stableKT,extraKT,"
                   "reKT,csKT,FlucKT,lefoKT,UKT,MTKT,RobustKT,DenoiseKT,MoC-KT,FA-KT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str,
                        # iterate over dkt+ and dkt, keep the  model_config consistent
                        default="SKVMN",
                        help="逗号分隔的模型列表，依次训练+评估，共用同一份 emb_size/dropout")
    # dkt+ 额外的正则项系数（DKTPlus 构造函数必填），默认值取自 pykt examples/wandb_dkt_plus_train.py
    parser.add_argument("--lambda_r", type=float, default=0.01)
    parser.add_argument("--lambda_w1", type=float, default=0.003)
    parser.add_argument("--lambda_w2", type=float, default=3.0)
    # SAKT 的自注意力块个数，默认值取自 pykt examples/wandb_sakt_train.py
    parser.add_argument("--num_en", type=int, default=1)
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
    # AT-DKT：emb_type 决定开哪些辅助任务；默认取官方 sweep（examples/seedwandb/atdkt.yaml）用的完整版，
    # emb_type="qid" 时 AT-DKT 退化成普通 DKT。其余默认值取自 examples/wandb_atdkt_train.py
    parser.add_argument("--atdkt_emb_type", type=str, default="qiddelxembhistranscembpredcurc")
    parser.add_argument("--num_layers", type=int, default=2, help="AT-DKT 预测 KC 用的 transformer 层数")
    parser.add_argument("--num_attn_heads", type=int, default=5, help="AT-DKT transformer 头数，需整除 emb_size")
    parser.add_argument("--l1", type=float, default=0.5, help="AT-DKT 答对预测 BCE 的权重")
    parser.add_argument("--l2", type=float, default=0.5, help="AT-DKT 预测当前 KC 的权重")
    parser.add_argument("--l3", type=float, default=0.5, help="AT-DKT 预测历史正确率的权重")
    parser.add_argument("--start", type=int, default=50, help="AT-DKT 从第几步开始算历史正确率 loss")
    # simpleKT 等 attention 模型：d_model / d_ff 都对应 emb_size，头数沿用上面的 --num_attn_heads（需整除 emb_size）；
    # 下面几个默认值取自 pykt examples/wandb_simplekt_train.py
    parser.add_argument("--n_blocks", type=int, default=2, help="attention block 层数")
    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    # QIKT：MLP 层数，默认值取自 pykt examples/wandb_qikt_train.py；各输出/损失的权重见 QIKT_OTHER_CONFIG
    parser.add_argument("--mlp_layer_num", type=int, default=2)
    # sparseKT：soft 变体累加注意力到 sparse_ratio 为止；topK 变体只保留前 k_index 个。默认值取自 wandb_sparsekt_train.py
    parser.add_argument("--sparse_ratio", type=float, default=0.8)
    parser.add_argument("--k_index", type=int, default=5)
    # RKT：embed_size 对应 emb_size，头数沿用 --num_attn_heads；下面默认值取自 pykt examples/wandb_rkt_train.py
    parser.add_argument("--num_attn_layers", type=int, default=4, help="RKT attention 层数")
    parser.add_argument("--grad_clip", type=float, default=10, help="RKT 梯度裁剪")
    parser.add_argument("--theta", type=float, default=0, help="RKT 关系阈值：phi < theta 的置 0")
    parser.add_argument("--time_span", type=int, default=100000)
    # DTransformer：n_know 个知识概念向量做诊断；lambda_cl 是对比学习 loss 权重（wandb_dtransformer_train.py 默认值）
    parser.add_argument("--n_know", type=int, default=16)
    parser.add_argument("--lambda_cl", type=float, default=0.1)
    # stableKT / csKT 的双曲锥注意力 penumbral(q, k, r, gamma)（--gamma 已被 LPKT 占用，这里单独起名）；
    # 不传时按模型取官方默认值，见 CONE_DEFAULTS
    parser.add_argument("--cone_r", type=float, default=None)
    parser.add_argument("--cone_gamma", type=float, default=None)
    # UKT：是否开不确定性感知对比学习（官方 wandb_ukt_train.py 默认 0=关），及其 loss 权重
    parser.add_argument("--ukt_use_cl", type=int, default=0)
    parser.add_argument("--ukt_cl_weight", type=float, default=0.02)
    parser.add_argument("--retrain", action="store_true",
                        help="忽略已有 checkpoint 缓存，强制重新训练")
    parser.add_argument("--fail_fast", action="store_true",
                        help="默认某个模型报错只跳过它、继续跑后面的；加这个则直接中断（调试时用）")
    parser.add_argument("--workdir", type=str, default=os.path.join(HERE, "work"))
    parser.add_argument("--max_users", type=int, default=3000,
                        help="只取前 N 个用户跑 demo；0 表示用全量数据")
    # 目前只有 xes3g5m 用：它的 test_window 是按步长 1 滑窗，学生序列又长（平均 370+ 步），
    # 3000 个学生能切出 3GB+ 的文件，所以测试集的学生数单独限一下
    parser.add_argument("--max_test_users", type=int, default=0,
                        help="测试集单独的学生数上限；0 表示和 --max_users 一致")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0, help="用作验证集的 fold")
    parser.add_argument("--batch_size", type=int, default=64)
    # 64
    # 评估用的 batch size。原来写死 1（一条序列一次前向），测试集 605 条 + window 1687 条会很慢；
    # 评估是逐位置 mask 后算 AUC/acc，分批不影响结果。0 表示跟训练的 batch_size 一致
    parser.add_argument("--eval_batch_size", type=int, default=0)
    # probe 模式：每个模型从头初始化，在训练集前 N 个 batch 上真训一遍（走完整训练路径，方便打断点），
    # 再在验证集前 N 个 batch 上评估；checkpoint 存到 {workdir}/probe/，不碰正式那批，也不读缓存。
    # 默认 1：不加任何参数直接跑脚本 = 全部模型各训一个 batch，验证训练路径都能跑通
    # 要走完整流程（训练/复用缓存 + 全测试集评估）传 --probe_batches 0
    parser.add_argument("--probe_batches", type=int, default=1,
                        help=">0 时进入 probe 模式：每个模型在训练集前 N 个 batch 上从头训一遍；"
                             "0 表示完整训练+评估流程")
    # 默认 1：现有 checkpoint 缓存都是按 1 个 epoch 训的（逐个模型接通时的冒烟测试），
    # 不传参数直接跑就能全部命中缓存。要正式结果就显式加大，比如 --num_epochs 20（会全部重训）
    parser.add_argument("--num_epochs", type=int, default=1)
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
        t0 = time.time()
        if args.dataset_name == "xes3g5m":
            # XES3G5M 给的已经是切好的序列，不走 pykt 的预处理，只采样 + 切测试集 + 写 config
            from xes3g5m_support import prepare_xes3g5m
            prepare_xes3g5m(os.path.join(HERE, XES3G5M_KC_LEVEL), dpath, config_file,
                            args.max_users, maxlen=args.maxlen, min_seq_len=args.min_seq_len,
                            max_test_users=args.max_test_users)
        else:
            raw_csv = prepare_raw(args.dataset_name, dpath, args.max_users)
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
    # GitHub 版 pykt 的 init_test_datasets / init_model 会读 data_config["dataset_name"]
    dconfig["dataset_name"] = args.dataset_name
    print(f"[data] num_c={dconfig['num_c']}, num_q={dconfig['num_q']}, "
          f"input_type={dconfig['input_type']}")

    # ---------- 2+3. 依次训练 + 评估每个模型 ----------
    model_names = [normalize_model_name(m) for m in args.model_name.split(",") if m.strip()]
    # 所有模型共用的超参，保证 dkt / dkt+ 的对比是公平的
    base_model_config = {"emb_size": args.emb_size, "dropout": args.dropout}
    if args.probe_batches > 0:
        probe_models(args, model_names, base_model_config, data_config, dconfig)
        return

    results, failed = [], []
    for model_name in model_names:
        set_seed(args.seed)  # 每个模型用同样的随机种子初始化
        try:
            results.append(train_and_eval(args, model_name, base_model_config, data_config, dconfig))
        except Exception as e:
            # 一个模型挂了不该让整轮（35 个模型、几小时）白跑；打完栈继续下一个，最后统一汇报
            if args.fail_fast:
                raise
            import traceback
            traceback.print_exc()
            failed.append((model_name, f"{type(e).__name__}: {e}"))
            print(f"[错误] {model_name} 跑挂了，跳过（要让它直接中断加 --fail_fast）")

    # ---------- 4. 汇总 ----------
    print("=" * 60)
    print(f"dataset={args.dataset_name} fold={args.fold} 共用 model_config={base_model_config}")
    print(f"{'model':<15}{'best_ep':>8}{'valid_auc':>11}{'test_auc':>10}{'test_acc':>10}"
          f"{'win_auc':>9}{'win_acc':>9}")
    for res in results:
        print(f"{res['model_name']:<15}{res['best_epoch']:>8}{res['validauc']:>11.4f}"
              f"{res['testauc']:>10.4f}{res['testacc']:>10.4f}"
              f"{res['wauc']:>9.4f}{res['wacc']:>9.4f}")
    if failed:
        print(f"-- 失败 {len(failed)} 个（上面各自有完整栈）:")
        for name, err in failed:
            print(f"   {name:<15}{err}")
    print("=" * 60)


# 不传 --max_batches 时各模型的默认值；下面这些在 CPU 上一个 batch 要 15s+，默认只跑 1 个 batch 证明能跑通
# （gkt ~20s/batch，dtransformer ~15-30s/batch）；正式结果用 --max_batches 0 放 GPU 上跑
DEFAULT_MAX_BATCHES = {"gkt": 1, "dtransformer": 1}


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
    elif model_name == "sakt":
        # SAKT(num_c, seq_len, emb_size, num_attn_heads, dropout, num_en, ...)
        # num_en：自注意力块个数，官方 wandb_sakt_train.py 默认 1（构造函数默认是 2）；
        # wandb_train.py 会给 sakt 补 seq_len；头数沿用 --num_attn_heads（需整除 emb_size）
        model_config.update({"seq_len"       : args.maxlen,
                             "num_attn_heads": args.num_attn_heads,
                             "num_en"        : args.num_en})
    elif model_name == "kqn":
        # KQN(n_skills, n_hidden, n_rnn_hidden, n_mlp_hidden, dropout, ...) 不接受 emb_size
        emb_size = model_config.pop("emb_size")
        model_config.update({"n_hidden"    : emb_size,
                             "n_rnn_hidden": args.n_rnn_hidden or emb_size,
                             "n_mlp_hidden": args.n_mlp_hidden or emb_size})
    elif model_name in ["dkvmn", "deep_irt", "skvmn"]:
        # DKVMN / DeepIRT / SKVMN(num_c, dim_s, size_m, dropout, ...) 都不接受 emb_size
        # DeepIRT = DKVMN + IRT 输出层；SKVMN = DKVMN + 序列建模（找历史上相同知识状态的时刻做 hop）
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
    elif model_name == "atdkt":
        # ATDKT(num_q, num_c, seq_len, emb_size, dropout, emb_type, num_layers, num_attn_heads, l1, l2, l3, start)
        # emb_type 不放进 model_config：pykt_gh 的 init_model 单独传它（见 gh_emb_type）
        model_config.update({"seq_len"       : args.maxlen,
                             "num_layers"    : args.num_layers,
                             "num_attn_heads": args.num_attn_heads,
                             "l1"            : args.l1,
                             "l2"            : args.l2,
                             "l3"            : args.l3,
                             "start"         : args.start})
    elif model_name in ["simplekt", "sparsekt_soft", "sparsekt_topk"]:
        # simpleKT(n_question, n_pid, d_model, n_blocks, dropout, d_ff, ..., seq_len, final_fc_dim, final_fc_dim2,
        #          num_attn_heads)；assist2015 的 num_q=0 -> n_pid=0，simpleKT 退化成只用 KC（无 Rasch 题目差异项）
        # sparseKT 是在 simpleKT 上加稀疏注意力，构造参数相同，另多 sparse_ratio / k_index
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads,
                             "final_fc_dim"  : args.final_fc_dim,
                             "final_fc_dim2" : args.final_fc_dim2})
        if model_name == "simplekt":
            model_config["seq_len"] = args.maxlen  # 官方 wandb_train.py 只给 simplekt 传 seq_len，sparsekt 用构造函数默认值
        else:
            model_config.update({"sparse_ratio": args.sparse_ratio,
                                 "k_index"     : args.k_index})
    elif model_name == "qikt":
        # QIKT(num_q, num_c, emb_size, dropout, mlp_layer_num, other_config, ...)
        model_config.update({"mlp_layer_num": args.mlp_layer_num,
                             "other_config" : dict(QIKT_OTHER_CONFIG)})
    elif model_name in ["folibikt", "extrakt", "akt"]:
        # AKT / folibiKT / extraKT(n_question, n_pid, d_model, n_blocks, dropout, d_ff, kq_same, final_fc_dim,
        #                          num_attn_heads[, seq_len], ...)：三者构造参数相同
        # （folibiKT = AKT + ALiBi 偏置，extraKT = AKT + 可外推的位置编码）
        # 官方训练脚本都不传 final_fc_dim（用构造函数默认 512）；wandb_train.py 只给 folibikt 补 seq_len，
        # akt / extrakt 用构造函数默认 200（= 这里的 maxlen）
        # ALiBi 斜率按 2^a 个头设计，头数不是 2 的幂时官方代码有近似处理
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads})
        if model_name == "folibikt":
            model_config["seq_len"] = args.maxlen
    elif model_name in ["fluckt", "mockt"]:
        # FlucKT / MocKT(n_question, n_pid, d_model, n_blocks, dropout, d_ff, bar_d, ..., num_attn_heads,
        #                ..., kernel_size[, sequence1, sequence2])：构造参数几乎相同
        # 官方训练脚本都不传 final_fc_dim / seq_len（用构造函数默认 512 / 200），其余结构超参见 FLUCKT_EXTRA
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads})
        model_config.update(FLUCKT_EXTRA)
        if model_name == "mockt":
            # MoC-KT 的差异：卷积核 4（FlucKT 是 5），另有两个按序列长度分段的阈值
            model_config.update({"kernel_size": 4, "sequence1": 50, "sequence2": 150})
    elif model_name == "denoisekt":
        # DenoiseKT(num_c, num_q, d_model, n_blocks, dropout, dropout1, bf, d_ff, seq_len, ...,
        #           final_fc_dim, final_fc_dim2, num_attn_heads)；默认值取自 wandb_denoisekt_train.py
        # dropout1：题目图 GCN 的 dropout；bf：去噪注意力里按 KC 相似度调权的底数（bf ** 相似度计数）
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads,
                             "final_fc_dim"  : args.final_fc_dim,
                             "final_fc_dim2" : args.final_fc_dim2,
                             "dropout1"      : 0.1,
                             "bf"            : 0.9})
    elif model_name == "robustkt":
        # Robustkt(n_question, n_pid, d_model, n_blocks, dropout, ks, d_ff, kq_same, final_fc_dim, num_attn_heads, ...)
        # ks：Smooth 模块里因果卷积的核大小（平滑窗口，平滑结果当"认知模式"，残差当"随机因素"）；
        # 官方 wandb_robustkt_train.py 把 ks 声明成 float，但它要喂给 nn.Conv1d 的 kernel_size，这里传 int 5；
        # 官方不传 final_fc_dim（构造函数默认 512）
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads,
                             "ks"            : 5})
    elif model_name == "fa_kt":
        # FA_KT(n_question, n_pid, num_rgap, num_sgap, num_pcount, d_model, n_blocks, dropout, d_ff, ...,
        #       kernel_size1, kernel_size2, final_fc_dim, final_fc_dim2, num_attn_heads, use_moe, num_experts,
        #       confidence_thresholds, mamba_*)；num_rgap 等由 init_dataset4train 写进 dconfig
        # 官方不传 seq_len（构造函数默认 200）；MoE / 阈值 / mamba 参数用构造函数默认值（= 官方脚本默认值）
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads,
                             "final_fc_dim"  : args.final_fc_dim,
                             "final_fc_dim2" : args.final_fc_dim2,
                             "kernel_size1"  : 5,
                             "kernel_size2"  : 5,
                             # 官方默认 num_experts=4（cnn/lstm/attention/mamba），我们在 pykt_gh/models/fa_kt.py
                             # 里去掉了要装 mamba_ssm（CUDA 专用）的 mamba 专家，这里跟着改成 3，
                             # 否则路由里 topk(k=4) 会 "selected index k out of range"
                             "num_experts"   : 3})
    elif model_name == "mtkt":
        # MTKT(n_question, n_pid, num_rgap, num_sgap, num_pcount, k1, k2, d_model, n_blocks, dropout, d_ff, ...,
        #      seq_len, final_fc_dim, final_fc_dim2, num_attn_heads)；num_rgap 等由 init_dataset4train 写进 dconfig
        # k1 / k2：因果交互卷积（CIC）里短期 / 长期两路卷积核大小；默认值取自 wandb_mtkt_train.py；
        # wandb_train.py 会给 mtkt 补 seq_len
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads,
                             "final_fc_dim"  : args.final_fc_dim,
                             "final_fc_dim2" : args.final_fc_dim2,
                             "seq_len"       : args.maxlen,
                             "k1"            : 1,
                             "k2"            : 5})
    elif model_name == "ukt":
        # UKT(n_question, n_pid, d_model, n_blocks, dropout, d_ff, ..., final_fc_dim, final_fc_dim2, num_attn_heads,
        #     use_CL, cl_weight, use_uncertainty_aug, atten_type, ...)；官方不传 seq_len（构造函数默认 200）
        # atten_type="w2"：用 2-Wasserstein 距离算注意力；use_uncertainty_aug 只在 use_CL 打开时生成增强样本
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"            : emb_size,
                             "d_ff"               : emb_size,
                             "n_blocks"           : args.n_blocks,
                             "num_attn_heads"     : args.num_attn_heads,
                             "final_fc_dim"       : args.final_fc_dim,
                             "final_fc_dim2"      : args.final_fc_dim2,
                             "use_CL"             : args.ukt_use_cl,
                             "cl_weight"          : args.ukt_cl_weight,
                             "use_uncertainty_aug": 1,
                             "atten_type"         : "w2"})
    elif model_name == "lefokt_akt":
        # LEFOKT_AKT(n_question, n_pid, d_model, n_blocks, dropout, d_ff, num_buckets, max_distance, init_c, init_L,
        #            bar_d, kq_same, final_fc_dim, num_attn_heads, ...)；官方不传 final_fc_dim / seq_len
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads})
        model_config.update(LEFOKT_EXTRA)
    elif model_name == "rekt":
        # ReKT(skill_max, pro_max, d, dropout)：d 对应 emb_size（官方 wandb_rekt_train.py 默认 d=128、dropout=0.4）
        model_config = {"d": model_config["emb_size"], "dropout": model_config["dropout"]}
    elif model_name in ["stablekt", "cskt"]:
        # stableKT / csKT(n_question, n_pid, d_model, n_blocks, dropout, d_ff, ..., seq_len, r, gamma,
        #                 final_fc_dim, final_fc_dim2, num_attn_heads, ...)：构造参数相同，官方 emb_type 都是 "qid"
        # wandb_train.py 只给 stablekt 补 seq_len，cskt 用构造函数默认 200（= 这里的 maxlen）
        default_r, default_gamma = CONE_DEFAULTS[model_name]
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "n_blocks"      : args.n_blocks,
                             "num_attn_heads": args.num_attn_heads,
                             "final_fc_dim"  : args.final_fc_dim,
                             "final_fc_dim2" : args.final_fc_dim2})
        if model_name == "stablekt":
            model_config["seq_len"] = args.maxlen
        model_config.update({"r"    : args.cone_r if args.cone_r is not None else default_r,
                             "gamma": args.cone_gamma if args.cone_gamma is not None else default_gamma})
    elif model_name == "dtransformer":
        # DTransformer(n_question, n_pid, d_model, d_ff, num_attn_heads, n_know, n_blocks, dropout,
        #              lambda_cl, proj, hard_neg, window, ...)；默认值取自 pykt examples/wandb_dtransformer_train.py
        emb_size = model_config.pop("emb_size")
        model_config.update({"d_model"       : emb_size,
                             "d_ff"          : emb_size,
                             "num_attn_heads": args.num_attn_heads,
                             "n_blocks"      : args.n_blocks,
                             "n_know"        : args.n_know,
                             "lambda_cl"     : args.lambda_cl,
                             "proj"          : True,
                             "hard_neg"      : False,
                             "window"        : 1})
    elif model_name == "rkt":
        # RKT(num_c, num_q, embed_size, num_attn_layers, num_heads, batch_size, grad_clip, theta,
        #     seq_len, drop_prob, time_span, ...)：不接受 emb_size / dropout，改名传入
        model_config = {"embed_size"     : model_config["emb_size"],
                        "drop_prob"      : model_config["dropout"],
                        "num_attn_layers": args.num_attn_layers,
                        "num_heads"      : args.num_attn_heads,
                        "batch_size"     : args.batch_size,
                        "grad_clip"      : args.grad_clip,
                        "theta"          : args.theta,
                        "seq_len"        : args.maxlen,
                        "time_span"      : args.time_span}
    return model_config


# FlucKT 的其余结构超参，取自 pykt examples/wandb_fluckt_train.py 的默认值：
#   kernel_size：分解短期波动的因果卷积核大小；bar_d：分解后的特征维度
#   num_buckets/max_distance（t5 位置编码）、n_hat/max_local_shift（longrope）、d_state/d_conv/expand（mamba）
#   在默认 emb_type=qid_conv_ker_noexp 下不起作用，照官方原样传入
FLUCKT_EXTRA = {"bar_d"          : 64,
                "kernel_size"    : 5,
                "num_buckets"    : 256,
                "max_distance"   : 256,
                "n_hat"          : 0,
                "max_local_shift": 0.0,
                "d_state"        : 16,
                "d_conv"         : 2,
                "expand"         : 1}


# lefoKT 的其余位置编码超参，取自 pykt examples/wandb_lefokt_akt_train.py 的默认值：
#   init_c/init_L：FIRE 偏置的对数变换系数 / 距离归一化阈值初值（当前 emb_type=qid_fire 用的就是它们）
#   num_buckets/max_distance（t5）、bar_d（sandwich）在 qid_fire 下不起作用，照官方原样传入
LEFOKT_EXTRA = {"num_buckets" : 16,
                "max_distance": 100,
                "init_c"      : 0.01,
                "init_L"      : 50,
                "bar_d"       : 32}


# 锥注意力 (r, gamma) 的官方默认值：wandb_stablekt_train.py 是 (0.5, 0.7)，wandb_cskt_train.py 是 (1, 1)
CONE_DEFAULTS = {"stablekt": (0.5, 0.7), "cskt": (1.0, 1.0)}


# QIKT 的输出模式和各子模块 loss / 输出权重，取自 pykt examples/wandb_qikt_train.py 的默认值
#   output_mode="an"：最终预测 = 各模块输出按 output_*_lambda 加权后的组合
#   loss_*_lambda 全 0：只用最终预测的 BCE 训练，不加各子模块的辅助 loss
QIKT_OTHER_CONFIG = {"output_mode"         : "an",
                     "loss_q_all_lambda"   : 0,
                     "loss_c_all_lambda"   : 0,
                     "loss_q_next_lambda"  : 0,
                     "loss_c_next_lambda"  : 0,
                     "output_q_all_lambda" : 1,
                     "output_c_all_lambda" : 1,
                     "output_q_next_lambda": 0,
                     "output_c_next_lambda": 1}


CKPT_NAME = "qid_model.ckpt"  # pykt train_model 保存的文件名：{emb_type}_model.ckpt
META_NAME = "train_meta.json"  # 我们自己记录的训练配置 + 验证集指标


# emb_type 不是 "qid" 的模型（取官方训练脚本的默认 emb_type）；atdkt 的在 gh_emb_type 里从命令行取
# sparseKT 两个变体只差 emb_type：topK = qid_sparseattn（保留注意力最高的 k_index 个），
#                                 soft = qid_accumulative_attn（按分数累加到 sparse_ratio 为止）
GH_EMB_TYPE = {"qikt"         : "iekt",
               "sparsekt_topk": "qid_sparseattn",
               "sparsekt_soft": "qid_accumulative_attn",
               "folibikt"     : "qid_alibi",  # FoLiBi = AKT + 按距离线性衰减的注意力偏置（ALiBi）
               "dtransformer" : "qid_cl",     # 开对比学习：loss 额外加 lambda_cl * 对比损失
               "fluckt"       : "qid_conv_ker_noexp",  # 因果卷积分解长期趋势/短期波动 + kernel 化注意力偏置
               "mockt"        : "qid_conv_ker_noexp",  # MoC-KT：多尺度因果卷积 + kerple 衰减注意力
               "fa_kt"        : "qidband",  # FA-KT：按频段分解交互序列，交给一组频率偏好不同的专家(MoE)
               # lefoKT：官方训练脚本默认 qid_alibi，但它和 extraKT(emb_type=qid) 走的是同一条
               # "AKT + ALiBi" 分支，超参 / 种子相同时两者计算完全一样。这里换成 FIRE（可学习的
               # 相对距离遗忘偏置）：官方脚本专门把 FIRE 的 init_c / init_L 从构造函数默认改成了 0.01 / 50，
               # 推测作者用过；论文具体用哪种偏置没能核实
               "lefokt_akt"   : "qid_fire",
               "ukt"          : "stoc_qid"}  # UKT：KC 表示成高斯分布（均值 + 协方差），注意力用 Wasserstein 距离
# 同一个 pykt 模型的不同变体：我们用变体名区分 checkpoint 目录 / 缓存，调 pykt 时换回它的 model_name
GH_PYKT_NAME = {"sparsekt_topk": "sparsekt",
                "sparsekt_soft": "sparsekt"}


def gh_emb_type(args, model_name):
    if model_name == "atdkt":
        return args.atdkt_emb_type  # AT-DKT 的 emb_type 决定开哪些辅助任务，可从命令行换
    return GH_EMB_TYPE.get(model_name, "qid")


def gh_pykt_name(model_name):
    return GH_PYKT_NAME.get(model_name, model_name)


def ckpt_name_for(args, model_name):
    """pykt 的 train_model 按 {emb_type}_model.ckpt 存最优模型；emb_type 非 qid 时文件名跟着变。"""
    return f"{gh_emb_type(args, model_name)}_model.ckpt"


def train_signature(args, model_name, model_config, max_batches):
    """决定 checkpoint 能否复用的训练配置；任何一项变了都要重新训练。"""
    signature = _base_train_signature(args, model_name, model_config, max_batches)
    # impl 记录跑这个模型用的是哪份 pykt 实现；全部统一到 pykt_gh 之后，
    # 之前用已安装 pykt 0.0.38 训的 checkpoint 会因为这一项不同而重训（lpkt 两版实现确有差异）
    signature["impl"] = "pykt_gh"
    if gh_emb_type(args, model_name) != "qid":
        signature["emb_type"] = gh_emb_type(args, model_name)
    return signature


def _base_train_signature(args, model_name, model_config, max_batches):
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


def load_train_cache(ckpt_path, signature, ckpt_name=CKPT_NAME):
    """返回 (是否可复用, meta)。

    - 有 ckpt + meta 且 signature 一致   -> 复用，meta 里带 validauc/best_epoch
    - 有 ckpt 但没有 meta（加缓存之前训的旧模型，比如 Deep-IRT）-> 也复用，但训练配置无法核对，meta=None
    - 其它情况                              -> 重新训练
    """
    if not os.path.exists(os.path.join(ckpt_path, ckpt_name)):
        return False, None
    meta_file = os.path.join(ckpt_path, META_NAME)
    if not os.path.exists(meta_file):
        return True, None
    with open(meta_file) as fin:
        meta = json.load(fin)
    # 过一遍 json 再比较，避免 tuple/list、int/float 之类的表示差异
    cached = meta.get("signature")
    current = json.loads(json.dumps(signature))
    if cached == current:
        return True, meta
    # 不一致时把具体哪几项变了打出来，免得只看到"重新训练"不知道为什么
    diff = [f"{k}: 缓存={cached.get(k)!r} 本次={current.get(k)!r}"
            for k in sorted(set(cached or {}) | set(current))
            if (cached or {}).get(k) != current.get(k)]
    print(f"[cache] 训练配置变了，需要重训（{'; '.join(diff)}）")
    return False, meta


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


# ---------- 模型相关操作的统一入口 ----------
# 所有模型都走 pykt_gh（GitHub 版 pykt 整包，固定在 commit 2fc4d64）：官方的 init_model /
# train_model / evaluate / 数据加载。已安装的 PyPI 版 pykt 0.0.38 只剩两处间接用途：
# vendor_pykt/atdkt.py 从它导入 ut_mask；pykt_gh 里缺的模型（本项目用不到）也不再回退到它。
# 跳过的模型：hcgkt（官方没提供 assist2015 的题目-KC 图，且该 commit 的 train_model 里 hcgkt 分支
# 引用未定义变量 step_size 等，官方流程本身跑不通）

# 这些模型的 init_dataset4train 有副作用：往 dconfig 里写 init_model 需要的字段
# （dkt_forget/mtkt/fa_kt 的 num_rgap 等、lpkt 的 num_at/num_it），
# 所以即使命中缓存、不训练，也必须照常建一次训练集 loader
NEEDS_TRAIN_LOADER = ["dkt_forget", "mtkt", "fa_kt", "lpkt"]


def kt_train_loaders(args, model_name, data_config, dconfig):
    """返回 (train_loader, valid_loader)。"""
    from pykt_gh.datasets import init_dataset4train
    return init_dataset4train(args.dataset_name, gh_pykt_name(model_name), data_config, args.fold,
                              args.batch_size, diff_level=args.difficult_levels)


def eval_batch_size(args):
    return args.eval_batch_size or args.batch_size


def kt_test_loaders(args, model_name, dconfig):
    """返回 (test_loader, test_window_loader, test_q_loader)；test_q_loader 为 None 表示没有 question-level 评估。"""
    from pykt_gh.datasets import init_test_datasets
    test_loader, test_window_loader, test_q_loader, _ = init_test_datasets(
            dconfig, gh_pykt_name(model_name),
            eval_batch_size(args),
            # args.batch_size
            diff_level=args.difficult_levels,
    )
    return test_loader, test_window_loader, test_q_loader


def kt_init_model(args, model_name, model_config, dconfig):
    from pykt_gh.models import init_model
    return fix_hawkes_on_cpu(init_model(gh_pykt_name(model_name), model_config, dconfig,
                                        gh_emb_type(args, model_name), dconfig["dataset_name"]))


def kt_load_model(args, model_name, model_config, dconfig, ckpt_path):
    from pykt_gh.models import load_model
    return fix_hawkes_on_cpu(load_model(gh_pykt_name(model_name), model_config, dconfig,
                                        gh_emb_type(args, model_name), ckpt_path, dconfig["dataset_name"]))


def kt_train_model(model_name, model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
                   dconfig=None, fold=None):
    """返回 (validauc, validacc, best_epoch)；最优模型存到 ckpt_path/{emb_type}_model.ckpt。
    dconfig / fold 只有 rkt 用：pykt train_model 要靠它们找 phi_array_{训练folds}.pkl。"""
    from pykt_gh.models import train_model
    (_, _, _, _, validauc, validacc, best_epoch) = train_model(
            model, train_loader, valid_loader, num_epochs, opt, ckpt_path,
            test_loader=None, test_window_loader=None, save_model=True,
            data_config=dconfig, fold=fold)
    return validauc, validacc, best_epoch


def kt_evaluate(model_name, model, loader, rel=None):
    """返回 (auc, acc)。rel 只有 rkt 用（KC 关系矩阵，见 rkt_support.py）。"""
    from pykt_gh.models import evaluate
    return evaluate(model, loader, gh_pykt_name(model_name), rel)


def probe_models(args, model_names, base_model_config, data_config, dconfig):
    """--probe_batches N：每个模型从头初始化，在训练集前 N 个 batch 上真训一遍，再在验证集前 N 个
    batch 上评估。用来快速验证"每个模型的训练路径都能跑通"，也方便打断点调试 loss。

    - 走的是完整训练路径（model_forward -> cal_loss -> backward -> opt.step），
      所以 pykt_gh/models/train_model.py 里各模型的 loss 分支都会执行到
    - 不读也不写正式 checkpoint：checkpoint 存到 {workdir}/probe/ 下，正式的那批不受影响
    - 各模型的 loader 都是 shuffle=False，且 *_q / *_steptime 这些派生文件只是给原始 sequences
      加列、行序不变，所以"前 N 个 batch"在所有模型上是同一批学生序列
    - N 个 batch 训出来的指标没有参考意义，只看能不能跑通

    返回 (结果 list, 模型 dict)；模型 dict 里是训练后的模型，方便在 REPL 里接着用。
    """
    import torch
    from pykt.utils import set_seed

    results, models = [], {}
    for model_name in model_names:
        print("-" * 60)
        print(f"[probe] {model_name}")
        set_seed(args.seed)  # 每个模型用同样的随机种子初始化
        cur_data_config, cur_dconfig, rel = prepare_data_for_model(args, model_name, data_config, dconfig)
        model_config = build_model_config(args, model_name, base_model_config)

        train_loader, valid_loader = kt_train_loaders(args, model_name, cur_data_config, cur_dconfig)
        train_loader = limit_loader(train_loader, args.probe_batches)
        valid_loader = limit_loader(valid_loader, args.probe_batches)
        n_seq = len(train_loader.dataset)

        # 单独的 probe 目录：pykt 的 train_model 会往里存"最好的一轮"，别覆盖正式 checkpoint
        ckpt_path = os.path.join(args.workdir, "probe", f"{args.dataset_name}_{model_name}_fold{args.fold}")
        os.makedirs(ckpt_path, exist_ok=True)

        model = kt_init_model(args, model_name, model_config, cur_dconfig)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        t0 = time.time()
        validauc, validacc, _ = kt_train_model(model_name, model, train_loader, valid_loader,
                                               1, opt, ckpt_path, dconfig=cur_dconfig, fold=args.fold)
        train_s = time.time() - t0
        print(f"[probe] 训练 {n_seq} 条序列: validauc={validauc:.4f} validacc={validacc:.4f}（{train_s:.1f}s）")

        models[model_name] = model
        results.append({"model_name": model_name, "n_seq": n_seq, "auc": validauc, "acc": validacc,
                        "train_s": train_s})

    print("=" * 60)
    print(f"dataset={args.dataset_name} fold={args.fold} probe：每个模型从头训练集前 {args.probe_batches} 个 batch "
          f"(batch_size={args.batch_size})，验证同样前 {args.probe_batches} 个 batch —— 指标只说明跑通，没有参考意义")
    print(f"{'model':<15}{'n_seq':>7}{'valid_auc':>11}{'valid_acc':>11}{'train_s':>9}")
    for res in results:
        print(f"{res['model_name']:<15}{res['n_seq']:>7}{res['auc']:>11.4f}{res['acc']:>11.4f}"
              f"{res['train_s']:>9.1f}")
    print("=" * 60)
    return results, models


def prepare_data_for_model(args, model_name, data_config, dconfig):
    """按模型补齐它需要的输入（伪 question / 伪时间戳 / Q 矩阵 / 题目图 ...）。

    返回 (data_config, dconfig, rel)；rel 只有 rkt 用（KC 关系矩阵）。
    传进来的 data_config 不会被改动，各模型拿到的是各自的副本。
    """
    if model_name in ["dkt_forget", "mtkt", "fa_kt"]:
        # DKT-Forget / MTKT / FA-KT 都用 DktForgetDataset 从 timestamps 算 rgap/sgap/pcount；
        # assist2015 没有时间戳，换成带伪时间戳（第 i 步 = i 分钟）的 sequences 文件
        data_config, dconfig = with_step_timestamps(data_config, args.dataset_name)
    elif model_name == "hawkes":
        # HawkesKT 需要 questions + timestamps；assist2015 都没有，用 concept 和伪时间戳代替
        data_config, dconfig = with_hawkes_inputs(data_config, args.dataset_name)
    elif model_name == "lpkt":
        # LPKT 需要 questions + Q 矩阵；assist2015 没有 question id，用 concept 充当
        data_config, dconfig = with_lpkt_inputs(data_config, args.dataset_name)
    elif model_name == "dimkt":
        # DIMKT 要 question id 算题目难度；另外 pykt 的 DIMKTDataset 只会去读写死的
        # train_valid_sequences.csv（没有 questions 列会 KeyError），所以先把难度文件生成好
        data_config, dconfig = with_concepts_as_questions(data_config, args.dataset_name)
        from dimkt_support import ensure_difficulty_files
        ensure_difficulty_files(dconfig, args.difficult_levels)
    elif model_name in ["atdkt", "rekt"]:
        # DIMKT 要 question id 算题目难度，AT-DKT 要 question 表示去预测当前 KC，
        # reKT 的 init_model 固定用题目版 ReKT（有 num_q x d 的题目 embedding，num_q=0 时无法用）；
        # assist2015 没有 question id，用 concept 充当
        data_config, dconfig = with_concepts_as_questions(data_config, args.dataset_name)
    elif model_name == "denoisekt":
        # question 级模型 + 需要题目-题目图，assist2015 上图退化成单位阵
        data_config, dconfig = with_denoisekt_inputs(data_config, args.dataset_name)
    elif model_name in ["iekt", "qdkt", "qikt"]:
        # question 级模型读 *_file_quelevel；assist2015 没有 question id，用 concept 充当
        data_config, dconfig = with_quelevel_inputs(data_config, args.dataset_name)

    rel = None  # RKT 的 KC 关系矩阵；其它模型用不到
    if model_name == "rkt":
        # pykt 仓库没有生成 phi_array 的脚本，按 RKT 论文的 phi 系数从训练 folds 算（见 rkt_support.py）
        from rkt_support import build_phi_array
        rel = build_phi_array(dconfig, args.fold)
    return data_config, dconfig, rel


def train_and_eval(args, model_name, base_model_config, data_config, dconfig):
    """单个模型：训练 -> 加载最优 checkpoint -> 导出预测 -> 评估，返回指标 dict。"""
    import torch

    print("-" * 60)
    print(f"[model] {model_name}")
    data_config, dconfig, rel = prepare_data_for_model(args, model_name, data_config, dconfig)

    # ---------- 2. 训练 ----------
    max_batches = resolve_max_batches(args, model_name)
    if max_batches > 0:
        print(f"[data] {model_name} 每个 loader 只取前 {max_batches} 个 batch（冒烟测试，指标无参考意义）")

    model_config = build_model_config(args, model_name, base_model_config)
    ckpt_path = os.path.join(args.workdir, "saved_model",
                             f"{args.dataset_name}_{model_name}_fold{args.fold}")
    os.makedirs(ckpt_path, exist_ok=True)

    signature = train_signature(args, model_name, model_config, max_batches)
    reuse, meta = (False, None) if args.retrain else load_train_cache(ckpt_path, signature,
                                                                      ckpt_name_for(args, model_name))
    # 命中缓存且 meta 齐全时不用训练集 / 验证集，省掉一次数据加载；
    # 但 NEEDS_TRAIN_LOADER 里的模型例外：init_dataset4train 会顺带往 dconfig 里写
    # num_rgap/num_sgap/num_pcount（dkt_forget/mtkt/fa_kt）或 num_at/num_it（lpkt），init_model 要用
    train_loader = valid_loader = None
    if not (reuse and meta is not None) or model_name in NEEDS_TRAIN_LOADER:
        train_loader, valid_loader = kt_train_loaders(args, model_name, data_config, dconfig)
        train_loader = limit_loader(train_loader, max_batches)
        valid_loader = limit_loader(valid_loader, max_batches)

    best_model = None
    if reuse:
        try:
            best_model = kt_load_model(args, model_name, model_config, dconfig, ckpt_path)
        except RuntimeError as e:  # 旧 checkpoint 和当前 model_config 形状对不上
            print(f"[cache] checkpoint 加载失败，重新训练：{str(e).splitlines()[0]}")
            if train_loader is None:
                # 上面因为"命中缓存"跳过了建 loader，现在要回退去训练，得补上
                train_loader, valid_loader = kt_train_loaders(args, model_name, data_config, dconfig)
                train_loader = limit_loader(train_loader, max_batches)
                valid_loader = limit_loader(valid_loader, max_batches)

    if best_model is not None and meta is not None:
        validauc, validacc, best_epoch = meta["validauc"], meta["validacc"], meta["best_epoch"]
        print(f"[cache] 训练配置一致，复用 checkpoint：{ckpt_path}（要重训加 --retrain）")
    elif best_model is not None:
        # 加缓存之前训练的旧 checkpoint：没有 meta，只能重新算一遍验证集指标
        validauc, validacc = kt_evaluate(model_name, best_model, valid_loader, rel)
        best_epoch = -1
        print(f"[cache] 复用旧 checkpoint（无 {META_NAME}，训练配置无法核对，best_epoch 未知）：{ckpt_path}"
              f"（要重训加 --retrain）")
    else:
        model = kt_init_model(args, model_name, model_config, dconfig)
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        t0 = time.time()
        validauc, validacc, best_epoch = kt_train_model(
                model_name, model, train_loader, valid_loader, args.num_epochs, opt, ckpt_path,
                dconfig=dconfig, fold=args.fold)
        print(f"[train] 耗时 {time.time() - t0:.1f}s, best_epoch={best_epoch}, "
              f"validauc={validauc:.4f}, validacc={validacc:.4f}")
        save_train_cache(ckpt_path, signature, validauc, validacc, best_epoch)

        # ---------- 3. 评估最优模型 ----------
        best_model = kt_load_model(args, model_name, model_config, dconfig, ckpt_path)
    test_loader, test_window_loader, test_q_loader = kt_test_loaders(args, model_name, dconfig)
    test_loader = limit_loader(test_loader, max_batches)
    test_window_loader = limit_loader(test_window_loader, max_batches)

    # 能不能导出模型具体的input 和output 以及label，做成一个csv
    testauc, testacc = kt_evaluate(model_name, best_model, test_loader, rel)
    wauc, wacc = kt_evaluate(model_name, best_model, test_window_loader, rel)

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

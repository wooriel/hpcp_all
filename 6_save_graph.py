import os
from pathlib import Path
import argparse
from graph import parse_train_log, plot_metric_trend, parse_eval_log, plot_eval_metric_by_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--check_dir", type=str, default="checkpoints", help="checkpoint directory name")
parser.add_argument("--exp_name", type=str, default="moco_attn_v9", help="name of training experiment") # moco_attn_v9 hpcp_attn_v10
parser.add_argument("--log_type", type=str, default="eval", help="one between train and eval")
parser.add_argument("--save_dir", type=str, default="save", help="name of saving directory name")
args = parser.parse_args()

checkpoint_dir = Path("checkpoints") / args.exp_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_path = checkpoint_dir / "log_{}.txt".format(args.log_type)
save_path = Path(args.save_dir) / args.exp_name

if args.log_type == "train":
    log_df = parse_train_log(log_path)
    for key in ["auc", "mrr", "map", "top1"]:
        plot_metric_trend(log_df, key, save_path)
else:
    log_df = parse_eval_log(log_path) # later change into eval
    for key in ["auc", "mrr", "map", "top1"]:
        plot_eval_metric_by_dataset(log_df, key, save_path)

import os
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import cs_cover80dataset



parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="covers80_32k", help="data directory name")
parser.add_argument("--cache_dir", type=str, default="cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
args = parser.parse_args()

# path
base_path = Path.cwd()
data_path = os.path.join(base_path, "data", args.data_dir)
cache_dir = os.path.join(base_path, args.cache_dir, "_".join([args.rep, args.cache_dir]))
cache_path = os.path.join(cache_dir, "covers80")

# load dataset
c80dataset = cs_cover80dataset.Cover80Dataset(lpath_src="list1.list", lpath_tgt="list2.list", dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, seed=2026, cache_rep=True, cache_dir=cache_path)

c80loader = DataLoader(c80dataset, batch_size=16, num_workers=4, pin_memory=True) # pin_memory=True for small dataset

for batch in c80loader:
    src_work_id = batch["src_work_id"]
    pos_work_id = batch["pos_work_id"]
    neg_work_id = batch["neg_work_id"]
    src_label = batch["src_label"]
    pos_label = batch["pos_label"]
    neg_label = batch["neg_label"]
    orig_inp = batch["orig_inp"]
    pos_inp = batch["pos_inp"]
    neg_inp = batch["neg_inp"]


    print(src_work_id.shape)
    print(pos_work_id.shape)
    print(neg_work_id.shape)
    print(src_label)
    print(pos_label)
    print(neg_label)
    print(orig_inp)
    print(pos_inp)
    print(neg_inp)

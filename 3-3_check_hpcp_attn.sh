#!/usr/bin/bash

for epoch in $(seq 60 5 65); do
    printf -v ep "%03d" "$epoch"
    python 3-3_eval_hpcp_attn.py --check_name="model_epoch_${ep}.pt"
done
#!/bin/bash
cd /usershome/cs671_user2/p16_blv
export CUDA_VISIBLE_DEVICES=2
/usershome/cs671_user2/miniconda3/bin/python src/training/09_dpo_training.py >> logs/dpo_v2.log 2>&1

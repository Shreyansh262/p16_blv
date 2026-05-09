#!/bin/bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn | head -1 | awk -F',' '{print $1}' | tr -d ' '

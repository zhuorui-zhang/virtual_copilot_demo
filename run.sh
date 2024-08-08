#!/bin/bash

# 激活 conda 环境
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
source ~/miniconda3/etc/profile.d/conda.sh

conda activate py38

cd /home/zhangzr/virtual_copilot_demo

# 运行 Streamlit 应用
streamlit run web_app_v2.py

read -p "Press [Enter] key to close..."
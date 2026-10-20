#!/usr/bin/env bash
# 一键重建 R 环境(conda-forge 清华镜像,无需 sudo)
set -e
~/miniconda3/bin/conda create -n r \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ --override-channels \
  r-base r-ggplot2 r-dplyr r-tidyr r-readr r-data.table r-scales r-patchwork \
  r-ggrepel r-ragg r-svglite r-showtext r-irkernel r-languageserver -y
echo "R 环境已就绪: ~/miniconda3/envs/r/bin/Rscript"

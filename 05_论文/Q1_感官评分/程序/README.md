# 第一问复现程序

Python 3.10；numpy、pandas、scipy、scikit-learn；作图使用R/ggplot2、ragg、dplyr、tidyr、readr、patchwork、svglite。

程序使用附带的固定划分和候选参数。所有填补、变换、基学习器选择及非负权重估计限制在相应训练数据中。

只复算已有折外指标：
```
python run_nnls.py --recompute-oof ../数据与结果/final_model_oof.npz --output 复算结果
```

完整嵌套验证（需要官方附件1.csv、4.csv）：
```
python run_nnls.py --attachments 官方附件目录 --execute --output 运行结果
```

新样本推断（先在全部训练样本内部重建折外基预测与堆叠，再对新样本预测）：
```
python run_nnls.py --attachments 官方附件目录 --predict-chemistry 新化学表.csv --output 新样本结果
```

绘图时在本目录执行`Rscript plot_final.R`；图片写入相邻的图片目录。0—6为训练经验分布的相对强度，不是原始评分。

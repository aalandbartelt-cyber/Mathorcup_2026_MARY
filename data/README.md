# 数据目录

新赛题数据按 ultralytics 标准结构放入:

```
data/<数据集名>/
├── images/{train,val,test}
└── labels/{train,val,test}
```

然后在 configs/ 下按 data.example.yaml 复制一份，改 path 和 names 即可。

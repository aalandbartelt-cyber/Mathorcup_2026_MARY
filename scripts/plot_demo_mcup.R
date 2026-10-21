# M 杯出图风格示例（R 版）：2x2 组合（训练曲线/混淆矩阵/PR/特征重要性）
# 运行：Rscript scripts/plot_demo_mcup.R（工作目录为项目根）
source("scripts/plot_style.R")
suppressPackageStartupMessages(library(patchwork))

set.seed(0)

# 1) 训练曲线（四模型，同 2023-2 风格）
models <- c("DenseNet121", "VGG16", "GoogLeNet", "ResNet50")
df <- do.call(rbind, lapply(seq_along(models), function(i) {
  data.frame(epoch = 1:50,
             acc = 0.86 + 0.13 * (1 - exp(-(1:50) / 7)) + rnorm(50, 0, 0.004 + 0.001 * i),
             model = models[i])
}))
df$model <- factor(df$model, levels = models)
p1 <- ggplot(df, aes(epoch, acc, colour = model)) +
  geom_line(linewidth = 0.7) +
  scale_colour_manual(values = setNames(CYCLE, models)) +
  coord_cartesian(ylim = c(0.70, 1.01)) +
  labs(x = "迭代次数", y = "准确率", title = "模型准确率随迭代次数变化") +
  theme_mcup(legend = "br")

# 2) 混淆矩阵
cm <- matrix(c(409, 4, 13, 400), nrow = 2, byrow = TRUE,
             dimnames = list(c("Normal", "Fault"), c("Normal", "Fault")))
p2 <- plot_confusion_matrix(cm, c("Normal", "Fault")) + labs(title = "混淆矩阵")

# 3) PR 曲线
rec <- seq(0, 1, length.out = 120)
p3 <- plot_pr_curves(list(
  dent  = list(recall = rec, precision = 1 - 0.35 * rec^6, ap = 0.804),
  hole  = list(recall = rec, precision = 1 - 0.45 * rec^4, ap = 0.794),
  rusty = list(recall = rec, precision = 1 - 0.30 * rec^8, ap = 0.833),
  all   = list(recall = rec, precision = 1 - 0.38 * rec^5, ap = 0.810)
))

# 4) 特征重要性
p4 <- plot_feature_importance(
  names  = c("新车价", "使用时长", "年款", "排量", "燃油类型_1", "里程", "品牌", "车身类型"),
  values = c(54.4, 9.5, 7.0, 3.7, 3.5, 2.9, 2.4, 1.8)
)

p <- (p1 | p2) / (p3 | p4)
save_paper(p, "figures/demo_m_cup_style_R.png", width = 11, height = 8)
cat("已保存 figures/demo_m_cup_style_R.png\n")

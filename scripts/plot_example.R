# R 出图示例: 生成一张论文级图表到 figures/
# 运行: Rscript scripts/plot_example.R(工作目录为项目根)
source("scripts/plot_style.R")

df <- data.frame(
  epoch = 1:5,
  mAP   = c(0.42, 0.55, 0.63, 0.67, 0.71)
)

p <- ggplot(df, aes(epoch, mAP)) +
  geom_line(colour = PALETTE[1], linewidth = 1) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 1:5) +
  labs(
    title = "模型精度随训练轮次的变化",
    x = "迭代轮次 (epoch)",
    y = "mAP50 (%)"
  ) +
  theme_paper()

save_paper(p, "figures/example_r_plot.png")
cat("已保存 figures/example_r_plot.png\n")

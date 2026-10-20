# 论文出图统一样式(R 版)
# 字体说明: paper 模式用宋体 SimSun(中文宋体,拉丁字符为 Times 风格衬线体);
#           sans 模式用黑体 SimHei。纯英文图可直接 family = "Times New Roman"。
# 用法:
#   source("scripts/plot_style.R")
#   p <- ggplot(...) + theme_paper()
#   save_paper(p, "figures/fig1.png")
suppressPackageStartupMessages({
  library(ggplot2)
  library(ragg)
})

FONT_PAPER <- "SimSun"
FONT_SANS  <- "SimHei"

# 统一主题: 黑白边框、浅灰网格、图例内置右上角
theme_paper <- function(mode = c("paper", "sans"), base_size = 11) {
  mode <- match.arg(mode)
  fam <- if (mode == "paper") FONT_PAPER else FONT_SANS
  theme_bw(base_size = base_size, base_family = fam) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey88", linewidth = 0.3),
      panel.border = element_rect(colour = "black", linewidth = 0.5),
      axis.text = element_text(colour = "black"),
      plot.title = element_text(size = base_size + 2, hjust = 0.5),
      legend.position = "inside",
      legend.position.inside = c(0.98, 0.98),
      legend.justification = c(1, 1),
      legend.background = element_blank(),
      legend.key = element_blank()
    )
}

# 保存为 300dpi PNG(论文印刷标准)
save_paper <- function(plot, file, width = 6, height = 4, dpi = 300) {
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  agg_png(file, width = width, height = height, units = "in", res = dpi, background = "white")
  on.exit(invisible(dev.off()), add = TRUE)
  print(plot)
  invisible(file)
}

# 常用配色(色盲友好)
PALETTE <- c("#2C7FB8", "#D95F02", "#1B9E77", "#7570B3", "#E6AB02", "#66A61E")

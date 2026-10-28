suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(grid)
})
source("scripts/plot_style.R")

out_dir <- "04_图表/Q3_风格生成/待审_20261027"
data_dir <- "05_论文/Q3_风格生成/待审_20261027/数据与结果/validation_v1"
d <- read.csv(file.path(data_dir, "独立置换逐重复差.csv"), check.names = FALSE)
s <- read.csv(file.path(data_dir, "独立置换汇总.csv"), check.names = FALSE)
style_order <- c("Tripel", "Kriek", "Lambic")
cols <- c("Tripel" = "#1F5A94", "Kriek" = "#67A995", "Lambic" = "#D4A35B")
d$style <- factor(d$style, levels = rev(style_order))
s$style <- factor(s$style, levels = rev(style_order))
PT <- 72.27 / 25.4
base_theme <- function() {
  theme_mcup(base_size = 12, legend = "none") + theme(
    panel.grid = element_blank(),
    panel.border = element_rect(colour = "#252525", fill = NA, linewidth = .45),
    axis.text = element_text(size = 11.5, colour = "#252525"),
    axis.title = element_text(size = 11.5, face = "bold"),
    axis.title.x = element_text(margin = margin(t = 8)),
    plot.subtitle = element_text(size = 12, face = "bold", hjust = 0,
                                 margin = margin(b = 7)),
    plot.margin = margin(9, 10, 4, 8),
    legend.position = "none"
  )
}

panel <- function(metric, title, xlabel, digits, breaks) {
  dd <- d
  dd$value <- dd[[paste0("delta_", metric)]]
  ss <- s
  ss$mean <- ss[[paste0("delta_", metric, "_mean")]]
  ss$sd <- ss[[paste0("delta_", metric, "_std")]]
  ss$low <- ss$mean - ss$sd
  ss$high <- ss$mean + ss$sd
  bounds <- range(dd$value, ss$low, ss$high, 0)
  span <- diff(bounds)
  ss$label <- sprintf(paste0("%+.", digits, "f"), ss$mean)
  ss$label_x <- bounds[2] + span * .12
  ggplot(dd, aes(value, style)) +
    geom_vline(xintercept = 0, colour = "#737373", linetype = "dashed", linewidth = .5) +
    geom_point(aes(colour = style), size = 1.7, alpha = .62,
               position = position_jitter(width = 0, height = .13, seed = 2710)) +
    geom_segment(data = ss, aes(x = low, xend = high, y = style, yend = style,
                                colour = style), linewidth = 1.1) +
    geom_point(data = ss, aes(x = mean, y = style, fill = style),
               shape = 23, colour = "#252525", size = 3.5, stroke = .65) +
    geom_text(data = ss, aes(x = label_x, y = style, label = label),
              hjust = 0, family = FONT_PAPER, size = 11 / PT, colour = "#252525") +
    scale_colour_manual(values = cols, guide = "none") +
    scale_fill_manual(values = cols, guide = "none") +
    scale_x_continuous(breaks = breaks,
                       limits = c(bounds[1] - span * .08, bounds[2] + span * .60),
                       expand = c(0, 0)) +
    scale_y_discrete(expand = expansion(add = .65)) +
    labs(subtitle = title, x = xlabel, y = NULL) + base_theme()
}

left <- panel("energy_v", "（a）联合分布误差变化", "energy 差（置换 − 原谱）", 5,
              c(-.002, 0, .002, .004))
right <- panel("spearman_pair_rmse", "（b）相关结构误差变化", "Spearman 相关 RMSE 差", 4,
               c(0, .01, .02, .03))
key <- ggplot() +
  annotate("point", x = .04, y = .5, colour = "#1F5A94", size = 2.2, alpha = .7) +
  annotate("text", x = .063, y = .5, label = "一次外层重复", hjust = 0,
           family = FONT_PAPER, size = 11 / PT) +
  annotate("segment", x = .35, xend = .41, y = .5, yend = .5, colour = "#1F5A94", linewidth = 1) +
  annotate("point", x = .38, y = .5, shape = 23, fill = "#1F5A94", colour = "#252525",
           size = 3.2, stroke = .65) +
  annotate("text", x = .43, y = .5, label = "均值 ± 重复标准差", hjust = 0,
           family = FONT_PAPER, size = 11 / PT) +
  annotate("segment", x = .77, xend = .82, y = .5, yend = .5,
           colour = "#737373", linetype = "dashed", linewidth = .5) +
  annotate("text", x = .84, y = .5, label = "零差值", hjust = 0,
           family = FONT_PAPER, size = 11 / PT) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_void(base_family = FONT_PAPER) +
  theme(plot.margin = margin(0, 8, 0, 8))

fig <- wrap_plots(left + right, key, ncol = 1, heights = c(1, .13))
png_file <- file.path(out_dir, "图4_联合结构独立置换检验.png")
save_paper(fig, png_file, width = 8.4, height = 4.5, dpi = 300)
svglite::svglite(sub("\\.png$", ".svg", png_file), width = 8.4, height = 4.5, bg = "white")
print(fig)
invisible(dev.off())
write.csv(data.frame(
  图号 = "图4", 路径 = png_file,
  图题 = "保持边缘分布的独立置换下三种风格的联合结构检验",
  说明 = "差为置换减原谱；菱形与横线为十次重复均值和标准差，线段不是独立置信区间。",
  口径 = "30外折×3风格×5生成种子，每组5次独立置换；边缘W1不变；小风格相关误差仅描述。"
), file.path(out_dir, "Q3_补充图清单.csv"), row.names = FALSE, fileEncoding = "UTF-8")

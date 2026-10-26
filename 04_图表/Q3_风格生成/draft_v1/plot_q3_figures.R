suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(grid)
})
source("scripts/plot_style.R")

figure_dir <- "04_图表/Q3_风格生成/draft_v1"
data_dir <- file.path(figure_dir, "source_data")
BLUE <- "#1F5A94"; GREEN <- "#67A995"; ORANGE <- "#D4A35B"; GRAY <- "#8A929C"
INK <- "#252525"
PT <- 72.27 / 25.4

paper_theme <- function(grid_axis = "x") {
  theme_mcup(base_size = 11, legend = "none") + theme(
    text = element_text(family = FONT_PAPER, colour = INK),
    axis.text = element_text(family = FONT_PAPER, size = 10.5, colour = INK),
    axis.title = element_text(family = FONT_PAPER, size = 10.5, face = "bold"),
    axis.title.x = element_text(margin = margin(t = 7)),
    axis.title.y = element_text(margin = margin(r = 7)),
    panel.grid.major.x = if (grid_axis == "x") element_line(colour = "#E7EAEC", linewidth = .24) else element_blank(),
    panel.grid.major.y = if (grid_axis == "y") element_line(colour = "#E7EAEC", linewidth = .24) else element_blank(),
    panel.border = element_rect(colour = "#222222", fill = NA, linewidth = .45),
    plot.subtitle = element_text(family = FONT_PAPER, size = 10.5, face = "bold",
                                 hjust = 0, margin = margin(b = 5)),
    plot.margin = margin(8, 9, 5, 7),
    legend.position = "bottom", legend.title = element_blank(),
    legend.text = element_text(family = FONT_PAPER, size = 10.5),
    legend.key.width = unit(19, "pt"), legend.key.height = unit(11, "pt")
  )
}
save_figure <- function(plot, file, width, height) {
  png_path <- file.path(figure_dir, file)
  save_paper(plot, png_path, width = width, height = height, dpi = 300)
  svglite::svglite(sub("\\.png$", ".svg", png_path), width = width, height = height,
                   bg = "white")
  print(plot)
  invisible(dev.off())
}

# Figure 1: all four compared methods share the same frozen 10×3 outer protocol.
fig1_data <- read.csv(file.path(data_dir, "图1_十次重复稳健指标.csv"), check.names = FALSE)
method_order <- c("风格收缩 Copula", "整行局部混合", "PCA 邻域混合", "基函数邻域混合")
method_colors <- c("风格收缩 Copula" = BLUE, "整行局部混合" = GREEN,
                   "PCA 邻域混合" = ORANGE, "基函数邻域混合" = GRAY)
fig1_data$method_cn <- factor(fig1_data$method_cn, levels = rev(method_order))
metric_panel <- function(metric, subtitle, xlabel) {
  d <- fig1_data[fig1_data$metric == metric, ]
  means <- aggregate(value ~ method_cn, d, mean)
  spread <- diff(range(d$value))
  if (spread == 0) spread <- max(abs(d$value), 1) * .1
  label_x <- max(d$value) + .20 * spread
  means$label <- sprintf("%.3f", means$value)
  means$label_x <- label_x
  ggplot(d, aes(x = value, y = method_cn, fill = method_cn)) +
    geom_boxplot(width = .55, alpha = .62, outlier.shape = NA, linewidth = .42,
                 orientation = "y", colour = "#434343") +
    geom_point(aes(colour = method_cn), position = position_jitter(width = 0, height = .095, seed = 314),
               size = 1.35, alpha = .75, show.legend = FALSE) +
    geom_point(data = means, aes(x = value, y = method_cn), inherit.aes = FALSE,
               shape = 23, size = 2.6, stroke = .6, fill = "white", colour = "#1D1D1D") +
    geom_text(data = means, aes(x = label_x, y = method_cn, label = label),
              inherit.aes = FALSE, hjust = 1, family = FONT_PAPER,
              size = 10.5 / PT, colour = INK) +
    scale_fill_manual(values = method_colors, guide = "none") +
    scale_colour_manual(values = method_colors, guide = "none") +
    scale_x_continuous(limits = c(min(d$value) - .07 * spread, max(d$value) + .26 * spread),
                       breaks = pretty(range(d$value), n = 4), expand = c(0, 0)) +
    scale_y_discrete(drop = FALSE, expand = expansion(add = .6)) +
    labs(subtitle = subtitle, x = xlabel, y = NULL) + paper_theme("x") +
    theme(legend.position = "none", axis.text.y = element_text(size = 10.2))
}
p1a <- metric_panel("mean_w1", "（a）边缘距离", "稳健尺度 W1（越低越好）")
p1b <- metric_panel("energy_v", "（b）联合分布", "稳健尺度 energy（越低越好）")
p1c <- metric_panel("coverage", "（c）留出样本覆盖", "留出样本覆盖率（越高越好）")
p1d <- metric_panel("log_spread_error", "（d）样本离散程度", "对数离散误差（越低越好）")
fig1 <- wrap_plots(p1a, p1b, p1c, p1d, ncol = 2, widths = c(1, 1), heights = c(1, 1))
save_figure(fig1, "图1_生成质量与多样性.png", 8.4, 5.2)

# Figure 2: 900 postselection synthetic profiles. Intervals are sample quantiles,
# not confidence intervals or true sensory-panel scores.
profile <- read.csv(file.path(data_dir, "图2_三风格六维感官画像.csv"), check.names = FALSE)
target_order <- c("酒精感", "苦味", "酒花味", "4VG", "酸度", "乳酸感")
style_order <- c("Tripel", "Kriek", "Lambic")
style_colors <- c("Tripel" = BLUE, "Kriek" = GREEN, "Lambic" = ORANGE)
style_shapes <- c("Tripel" = 16, "Kriek" = 17, "Lambic" = 15)
style_offset <- c("Tripel" = .22, "Kriek" = 0, "Lambic" = -.22)
profile$style <- factor(profile$style, levels = style_order)
profile$y <- match(profile$target_cn, rev(target_order)) + style_offset[as.character(profile$style)]
fig2 <- ggplot(profile, aes(colour = style, shape = style)) +
  geom_segment(aes(x = q10, xend = q90, y = y, yend = y), linewidth = .72,
               show.legend = FALSE) +
  geom_point(aes(x = mean, y = y), size = 2.8, stroke = .4) +
  scale_colour_manual(values = style_colors, breaks = style_order) +
  scale_shape_manual(values = style_shapes, breaks = style_order) +
  scale_x_continuous(breaks = 0:6, limits = c(0, 6), expand = c(0, 0)) +
  scale_y_continuous(breaks = 1:6, labels = rev(target_order), limits = c(.55, 6.45),
                     expand = c(0, 0)) +
  labs(x = "相对感官等级（0–6）", y = NULL) + paper_theme("x") +
  theme(legend.position = "bottom", legend.direction = "horizontal",
        legend.key.width = unit(17, "pt"), axis.text.y = element_text(size = 11))
save_figure(fig2, "图2_三风格感官画像.png", 8.4, 4.6)

# Figure 3: the unavailable cases remain in the denominator of 33.
migration <- read.csv(file.path(data_dir, "图3_固定主规则迁移状态.csv"), check.names = FALSE)
count <- read.csv(file.path(data_dir, "图3_各风格状态计数.csv"), check.names = FALSE)
status_order <- c("待实测统计候选", "直接化学证据不足", "感官代理不稳定", "未找到搜索可行谱")
status_colors <- c("待实测统计候选" = BLUE, "直接化学证据不足" = GREEN,
                   "感官代理不稳定" = ORANGE, "未找到搜索可行谱" = "#C8CDD1")
status_shapes <- c("待实测统计候选" = 17, "直接化学证据不足" = 16,
                   "感官代理不稳定" = 15, "未找到搜索可行谱" = 4)
migration$status_cn <- factor(migration$status_cn, levels = status_order)
migration$candidate_available <- tolower(as.character(migration$candidate_available)) == "true"
migration$eligible <- tolower(as.character(migration$eligible)) == "true"
count$status_cn <- factor(count$status_cn, levels = status_order)
available <- migration[migration$candidate_available, ]
selected <- available[available$eligible, ]
selected$label <- paste0("啤酒", selected$beer, "→", selected$target_style)
available$gain_pct <- available$relative_target_gain * 100
selected$gain_pct <- selected$relative_target_gain * 100
selected$label_x <- .70
selected$label_y <- ifelse(selected$target_style == "Tripel", 41.5, 34.5)
p3a <- ggplot(available, aes(source_change_rms, gain_pct, colour = status_cn, shape = status_cn)) +
  geom_point(size = 2.8, stroke = .5, alpha = .9) +
  geom_segment(data = selected, aes(x = source_change_rms, y = gain_pct,
                                    xend = label_x - .02, yend = label_y),
               colour = "#6D7479", linewidth = .35, inherit.aes = FALSE) +
  geom_label(data = selected, aes(x = label_x, y = label_y, label = label),
             colour = INK, family = FONT_PAPER, fontface = "bold", size = 10.5 / PT,
             fill = "white", linewidth = 0, label.padding = unit(.06, "lines"),
             hjust = 0, inherit.aes = FALSE) +
  scale_colour_manual(values = status_colors, guide = "none") +
  scale_shape_manual(values = status_shapes, guide = "none") +
  scale_x_continuous(limits = c(0, 1.27), breaks = c(0, .3, .6, .9, 1.2), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 55), breaks = seq(0, 50, 10), expand = c(0, 0)) +
  labs(subtitle = "（a）有谱候选的收益与改动", x = "规范化谱改动 RMS", y = "目标距离下降（%）") +
  paper_theme("both") + theme(legend.position = "none")

count$target_style <- factor(count$target_style, levels = rev(style_order))
count$text_color <- ifelse(count$status_cn == "待实测统计候选", "white", INK)
p3b <- ggplot(count, aes(x = n, y = target_style, fill = status_cn, group = status_cn)) +
  geom_col(width = .55, colour = "white", linewidth = .4,
           position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = ifelse(n > 0, n, ""), colour = text_color),
            position = position_stack(vjust = .5, reverse = TRUE), family = FONT_PAPER,
            fontface = "bold", size = 10.5 / PT, show.legend = FALSE) +
  scale_colour_identity() +
  scale_fill_manual(values = status_colors, breaks = status_order, drop = FALSE) +
  scale_x_continuous(limits = c(0, 11), breaks = c(0, 2, 4, 6, 8, 10), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = .65)) +
  labs(subtitle = "（b）各目标风格的审计状态", x = "源啤酒数（每风格 11）", y = NULL) +
  paper_theme("x") + theme(legend.position = "bottom", legend.box = "horizontal",
                        legend.text = element_text(size = 9.8),
                        legend.key.width = unit(13, "pt")) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))
fig3 <- p3a + p3b + plot_layout(ncol = 2, widths = c(1.15, 1))
save_figure(fig3, "图3_迁移收益与审计状态.png", 8.4, 4.8)

cat("Q3 draft figures rendered: 3 PNG files\n")

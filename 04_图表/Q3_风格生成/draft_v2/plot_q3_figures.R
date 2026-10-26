suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(grid)
})
source("scripts/plot_style.R")

figure_dir <- "04_图表/Q3_风格生成/draft_v2"
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
fig1_key <- ggplot() +
  annotate("segment", x = .045, xend = .112, y = .5, yend = .5,
           colour = "#434343", linewidth = .55) +
  annotate("segment", x = c(.045, .112), xend = c(.045, .112),
           y = .29, yend = .71, colour = "#434343", linewidth = .55) +
  annotate("rect", xmin = .063, xmax = .094, ymin = .28, ymax = .72,
           fill = BLUE, alpha = .62, colour = "#434343", linewidth = .5) +
  annotate("segment", x = .078, xend = .078, y = .28, yend = .72,
           colour = "#434343", linewidth = .65) +
  annotate("text", x = .125, y = .5, label = "箱体 25%–75%；须为 1.5 IQR 内数据",
           hjust = 0, family = FONT_PAPER, size = 10.2 / PT, colour = INK) +
  annotate("point", x = .555, y = .5, colour = BLUE, size = 2.5) +
  annotate("text", x = .575, y = .5, label = "十次重复",
           hjust = 0, family = FONT_PAPER, size = 10.2 / PT, colour = INK) +
  annotate("point", x = .755, y = .5, shape = 23, size = 3, stroke = .7,
           fill = "white", colour = INK) +
  annotate("text", x = .78, y = .5, label = "重复均值",
           hjust = 0, family = FONT_PAPER, size = 10.2 / PT, colour = INK) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_void(base_family = FONT_PAPER) +
  theme(plot.margin = margin(0, 8, 0, 8))
fig1_panels <- wrap_plots(p1a, p1b, p1c, p1d, ncol = 2)
fig1 <- wrap_plots(fig1_panels, fig1_key, ncol = 1, heights = c(1, .095))
save_figure(fig1, "图1_生成质量与多样性.png", 8.4, 5.35)

# Figure 2: 900 postselection synthetic profiles. Grouped columns give each
# style a stable visual position; whiskers show synthetic-sample quantiles.
profile <- read.csv(file.path(data_dir, "图2_三风格六维感官画像.csv"), check.names = FALSE)
style_order <- c("Tripel", "Kriek", "Lambic")
style_colors <- c("Tripel" = BLUE, "Kriek" = GREEN, "Lambic" = ORANGE)
profile$style <- factor(profile$style, levels = style_order)
profile_panel <- function(targets, subtitle, show_y = FALSE) {
  d <- profile[profile$target_cn %in% targets, ]
  d$target_cn <- factor(d$target_cn, levels = targets)
  dodge <- position_dodge(width = .80)
  ggplot(d, aes(x = target_cn, y = mean, fill = style, group = style)) +
    geom_col(position = dodge, width = .68, colour = "#52616C", linewidth = .22,
             alpha = .88) +
    geom_errorbar(aes(ymin = q10, ymax = q90), position = dodge, width = .12,
                  linewidth = .46, colour = INK, show.legend = FALSE) +
    geom_text(aes(y = q90 + .17, label = sprintf("%.1f", mean)),
              position = dodge, size = 10 / PT, family = FONT_PAPER,
              colour = INK, show.legend = FALSE) +
    scale_fill_manual(values = style_colors, breaks = style_order) +
    scale_y_continuous(limits = c(0, 6.25), breaks = 0:6, expand = c(0, 0)) +
    scale_x_discrete(expand = expansion(add = .54)) +
    labs(subtitle = subtitle, x = NULL,
         y = if (show_y) "相对感官等级（0–6）" else NULL) +
    paper_theme("y") +
    theme(axis.text.x = element_text(size = 10.5),
          legend.position = "bottom")
}
p2a <- profile_panel(c("酒精感", "苦味", "酒花味"), "（a）酒精、苦味与酒花", TRUE)
p2b <- profile_panel(c("4VG", "酸度", "乳酸感"), "（b）酚香与酸感")
fig2_key <- ggplot() +
  annotate("rect", xmin = c(.235, .405, .575), xmax = c(.26, .43, .60),
           ymin = .61, ymax = .82, fill = unname(style_colors),
           colour = "#52616C", linewidth = .3) +
  annotate("text", x = c(.268, .438, .608), y = .715,
           label = style_order, hjust = 0, family = FONT_PAPER,
           size = 10.4 / PT, colour = INK) +
  annotate("rect", xmin = .205, xmax = .228, ymin = .20, ymax = .48,
           fill = BLUE, colour = "#52616C", linewidth = .3) +
  annotate("text", x = .238, y = .34, label = "柱高：预测均值",
           hjust = 0, family = FONT_PAPER, size = 10.2 / PT, colour = INK) +
  annotate("segment", x = .47, xend = .47, y = .20, yend = .48,
           colour = INK, linewidth = .65) +
  annotate("segment", x = c(.46, .46), xend = c(.48, .48),
           y = c(.20, .48), yend = c(.20, .48), colour = INK, linewidth = .65) +
  annotate("text", x = .492, y = .34,
           label = "样本第 10–90 百分位（非置信区间）",
           hjust = 0, family = FONT_PAPER, size = 10.2 / PT, colour = INK) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_void(base_family = FONT_PAPER) +
  theme(plot.margin = margin(0, 8, 0, 8))
fig2_panels <- (p2a + p2b + plot_layout(widths = c(1, 1))) &
  theme(legend.position = "none")
fig2 <- wrap_plots(fig2_panels, fig2_key, ncol = 1, heights = c(1, .15))
save_figure(fig2, "图2_三风格感官画像.png", 8.4, 4.8)

# Figure 3: the unavailable cases remain in the denominator of 33.
migration <- read.csv(file.path(data_dir, "图3_固定主规则迁移状态.csv"), check.names = FALSE)
count <- read.csv(file.path(data_dir, "图3_各风格状态计数.csv"), check.names = FALSE)
status_order <- c("待实测统计候选", "直接化学证据不足", "感官代理不稳定", "未找到搜索可行谱")
status_colors <- c("待实测统计候选" = BLUE, "直接化学证据不足" = GREEN,
                   "感官代理不稳定" = ORANGE, "未找到搜索可行谱" = "#C8CDD1")
status_shapes <- c("待实测统计候选" = 17, "直接化学证据不足" = 16,
                   "感官代理不稳定" = 15, "未找到搜索可行谱" = 4)
status_short <- c("待实测统计候选" = "待复测候选", "直接化学证据不足" = "直接证据不足",
                  "感官代理不稳定" = "代理不稳", "未找到搜索可行谱" = "无可行谱")
status_legend <- c("待实测统计候选" = "待实测统计候选",
                   "直接化学证据不足" = "直接化学证据不足",
                   "感官代理不稳定" = "感官代理不稳定",
                   "未找到搜索可行谱" = "未找到搜索可行谱")
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
  geom_point(size = 2.8, stroke = .5, alpha = .9, show.legend = TRUE) +
  geom_segment(data = selected, aes(x = source_change_rms, y = gain_pct,
                                    xend = label_x - .02, yend = label_y),
               colour = "#6D7479", linewidth = .35, inherit.aes = FALSE) +
  geom_label(data = selected, aes(x = label_x, y = label_y, label = label),
             colour = INK, family = FONT_PAPER, fontface = "bold", size = 10.5 / PT,
             fill = "white", linewidth = 0, label.padding = unit(.06, "lines"),
             hjust = 0, inherit.aes = FALSE) +
  scale_colour_manual(values = status_colors, limits = status_order, drop = FALSE,
                      labels = unname(status_legend), name = "审计状态") +
  scale_shape_manual(values = status_shapes, limits = status_order, drop = FALSE,
                     labels = unname(status_legend), name = "审计状态") +
  scale_x_continuous(limits = c(0, 1.27), breaks = c(0, .3, .6, .9, 1.2), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 55), breaks = seq(0, 50, 10), expand = c(0, 0)) +
  labs(subtitle = "（a）29 条有谱候选：收益与改动", x = "规范化谱改动 RMS", y = "目标距离下降（%）") +
  paper_theme("both") +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE,
                               override.aes = list(size = 2.6, alpha = 1)),
         shape = guide_legend(nrow = 2, byrow = TRUE))

count_total <- aggregate(n ~ status_cn, data = count, FUN = sum)
count_total$short <- factor(status_short[as.character(count_total$status_cn)],
                            levels = rev(unname(status_short)))
p3b <- ggplot(count_total, aes(x = n, y = short, colour = status_cn, shape = status_cn)) +
  geom_segment(aes(x = 0, xend = n, y = short, yend = short),
               linewidth = .85, show.legend = FALSE) +
  geom_point(size = 3.4, stroke = .8, show.legend = FALSE) +
  geom_text(aes(x = n + 1.1, label = n), colour = INK, family = FONT_PAPER,
            fontface = "bold", hjust = 0, size = 11 / PT, show.legend = FALSE) +
  scale_colour_manual(values = status_colors, guide = "none") +
  scale_shape_manual(values = status_shapes, guide = "none") +
  scale_x_continuous(limits = c(0, 30), breaks = c(0, 10, 20, 30), expand = c(0, 0)) +
  scale_y_discrete(expand = expansion(add = .7)) +
  labs(subtitle = "（b）33 组任务：审计去向", x = "任务组数", y = NULL) +
  paper_theme("x") + theme(legend.position = "none", axis.text.y = element_text(size = 10.5))
fig3 <- p3a + p3b + plot_layout(ncol = 2, widths = c(1.08, 1), guides = "collect") &
  theme(legend.position = "bottom", legend.text = element_text(family = FONT_PAPER, size = 10),
        legend.key.width = unit(15, "pt"))
save_figure(fig3, "图3_迁移收益与审计状态.png", 8.4, 4.7)

cat("Q3 draft figures rendered: 3 PNG files\n")

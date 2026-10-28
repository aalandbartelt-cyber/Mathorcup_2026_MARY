#!/usr/bin/env Rscript
# Q1 S2: summary-feature ablation, label-permutation negative control, leave-one-style-out.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(scales)
})
source("scripts/plot_style.R")

ABL <- path.expand("~/mathorcup/.worktrees/q1-nnls-opt/03_结果/Q1_感官评分/validation_ablation_v1")
OUT <- "04_图表/Q1_感官评分/validation_v1"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

summ <- read.csv(file.path(ABL, "q1_ablation_summary.csv"), check.names = FALSE) %>%
  mutate(setting = recode(setting,
                          "完整模型（重复0，3折）" = "完整模型",
                          "移除4个汇总特征" = "去汇总特征",
                          "标签置换负对照" = "标签置换"),
         setting = factor(setting, levels = c("完整模型", "去汇总特征", "标签置换")))

col_set <- c("完整模型" = "#367DA2", "去汇总特征" = "#32765F", "标签置换" = "#C88240")

p_a <- ggplot(summ, aes(setting, r2, fill = setting)) +
  geom_hline(yintercept = 0, linetype = 2, colour = "grey45", linewidth = 0.35) +
  geom_col(width = 0.58) +
  geom_text(aes(label = sprintf("%.4f", r2)), vjust = ifelse(summ$r2 >= 0, -0.3, 1.2),
            size = 2.5) +
  scale_fill_manual(values = col_set) +
  scale_y_continuous(expand = expansion(mult = c(0.12, 0.12))) +
  labs(x = NULL, y = expression("宏平均 " * R^2)) +
  theme_mcup(base_size = 9, legend = "none")

p_b <- ggplot(summ, aes(setting, mae, fill = setting)) +
  geom_col(width = 0.58) +
  geom_text(aes(label = sprintf("%.4f", mae)), vjust = -0.3, size = 2.5) +
  scale_fill_manual(values = col_set) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = "宏平均 MAE") +
  theme_mcup(base_size = 9, legend = "none")

leave <- read.csv(file.path(ABL, "q1_leave_style_out.csv"), check.names = FALSE) %>%
  mutate(style = factor(style, levels = style[order(spearman)]),
         n_group = ifelse(n >= 10, "n≥10", "n<10"))

p_c <- ggplot(leave, aes(spearman, style, fill = n_group)) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey45", linewidth = 0.35) +
  geom_col(width = 0.62) +
  geom_text(aes(label = sprintf("%.2f (n=%d)", spearman, n)),
            hjust = ifelse(leave$spearman >= 0, -0.08, 1.08), size = 2.2) +
  scale_fill_manual(values = c("n≥10" = "#367DA2", "n<10" = "#C88240")) +
  scale_x_continuous(limits = c(-0.15, 0.48), expand = expansion(mult = c(0.02, 0.08))) +
  labs(x = "留一风格 Spearman", y = NULL) +
  theme_mcup(base_size = 8.5, legend = "none") +
  theme(axis.text.y = element_text(size = 6.8))

p_d <- ggplot(leave, aes(mae, style, colour = n_group)) +
  geom_point(size = 1.6) +
  geom_text(aes(label = sprintf("%.3f", mae)), hjust = -0.30, size = 2.2) +
  scale_colour_manual(values = c("n≥10" = "#367DA2", "n<10" = "#C88240"), guide = "none") +
  scale_x_continuous(limits = c(0.24, 0.58), breaks = c(0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
                     expand = expansion(mult = c(0.01, 0.06))) +
  labs(x = "留一风格 MAE", y = NULL) +
  theme_mcup(base_size = 8.5, legend = "none") +
  theme(axis.text.y = element_blank())

p <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1, 1.7)) +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 9))

save_paper(p, file.path(OUT, "附图S2_消融与负对照.png"),
           width = 7.8, height = 6.0, dpi = 300)
ggsave(file.path(OUT, "附图S2_消融与负对照.pdf"), p,
       width = 7.8, height = 6.0, device = cairo_pdf, bg = "white")
cat("saved", OUT, "\n")

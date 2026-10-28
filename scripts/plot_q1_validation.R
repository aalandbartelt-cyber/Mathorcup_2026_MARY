#!/usr/bin/env Rscript
# Q1 validation figures: repeated-CV stability, calibration, interval coverage,
# per-target reliability. All data come from frozen OOF; no model is refit.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(scales)
})
source("scripts/plot_style.R")

DATA <- "05_论文/Q1_感官评分/数据与结果"
VAL <- file.path(DATA, "validation_v1")
OUT <- "04_图表/Q1_感官评分/validation_v1"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

rep <- read.csv(file.path(DATA, "model_metrics_by_repeat.csv"), check.names = FALSE)
rep <- rep %>%
  rename(repeat_id = "repeat") %>%
  filter(display_name %in% c("ExtraTrees基线", "感官组收缩NNLS")) %>%
  mutate(model = recode(display_name,
                        "ExtraTrees基线" = "基线",
                        "感官组收缩NNLS" = "最终模型"),
         model = factor(model, levels = c("基线", "最终模型")))

cal <- read.csv(file.path(VAL, "q1_calibration_bins.csv"), check.names = FALSE) %>%
  mutate(group_plot = recode(group_plot,
                             "香气" = "香气", "风味" = "风味",
                             "味觉" = "味觉/口感", "口感" = "味觉/口感",
                             "总体" = "总体"),
         group_plot = factor(group_plot, levels = c("香气", "风味", "味觉/口感", "总体")))

cov <- read.csv(file.path(VAL, "q1_interval_coverage.csv"), check.names = FALSE) %>%
  mutate(method = recode(method,
                         "repeat_sd" = "重复标准差",
                         "cross_repeat_residual" = "残差校准"),
         method = factor(method, levels = c("重复标准差", "残差校准")))

rel <- read.csv(file.path(VAL, "q1_target_reliability.csv"), check.names = FALSE) %>%
  mutate(tier = factor(reliability_tier, levels = c("较低", "中等", "较高")),
         target_code = factor(target_code, levels = target_code[order(r2)]))

col_model <- c("基线" = "#4A5560", "最终模型" = "#367DA2")
col_group <- c("香气" = "#367DA2", "风味" = "#32765F",
               "味觉/口感" = "#C88240", "总体" = "#6E6E6E")
col_method <- c("重复标准差" = "#C88240", "残差校准" = "#367DA2")
col_tier <- c("较低" = "#9A9A9A", "中等" = "#C88240", "较高" = "#32765F")

p_a <- ggplot(rep, aes(repeat_id, r2, colour = model, group = model)) +
  geom_line(linewidth = 0.45) +
  geom_point(size = 1.7) +
  scale_colour_manual(values = col_model) +
  scale_x_continuous(breaks = 0:9, limits = c(0, 10.4)) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.12))) +
  labs(x = "重复", y = expression("宏平均 " * R^2)) +
  theme_mcup(base_size = 9, legend = "none") +
  geom_text(data = filter(rep, repeat_id == 9),
            aes(x = 9.25, label = model), hjust = 0, size = 2.7,
            show.legend = FALSE)

p_b <- ggplot(rep, aes(repeat_id, mae, colour = model, group = model)) +
  geom_line(linewidth = 0.45) +
  geom_point(size = 1.7) +
  scale_colour_manual(values = col_model) +
  scale_x_continuous(breaks = 0:9, limits = c(0, 10.4)) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.12))) +
  labs(x = "重复", y = "宏平均 MAE") +
  theme_mcup(base_size = 9, legend = "none") +
  geom_text(data = filter(rep, repeat_id == 9),
            aes(x = 9.25, label = model), hjust = 0, size = 2.7,
            show.legend = FALSE)

p_c <- ggplot(cal, aes(mean_pred, mean_truth, colour = group_plot)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "grey45", linewidth = 0.35) +
  geom_point(size = 1.25, alpha = 0.8) +
  scale_colour_manual(values = col_group) +
  labs(x = "预测均值（5 分位箱）", y = "观测均值") +
  theme_mcup(base_size = 9, legend = "right") +
  theme(legend.text = element_text(size = 7.2))

cov_sum <- cov %>%
  group_by(method, nominal) %>%
  summarise(mean_cov = mean(coverage),
            q10 = quantile(coverage, 0.10),
            q90 = quantile(coverage, 0.90),
            .groups = "drop") %>%
  mutate(x_plot = nominal + ifelse(method == "重复标准差", -0.008, 0.008))

p_d <- ggplot(cov_sum, aes(x_plot, mean_cov, colour = method, group = method)) +
  geom_abline(intercept = 0, slope = 1, linetype = 2, colour = "grey45", linewidth = 0.35) +
  geom_errorbar(aes(ymin = q10, ymax = q90), width = 0.012, linewidth = 0.40) +
  geom_line(linewidth = 0.45) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = col_method) +
  scale_x_continuous(breaks = c(0.80, 0.90, 0.95),
                     labels = percent_format(accuracy = 1)) +
  scale_y_continuous(limits = c(0.10, 1.00),
                     breaks = c(0.2, 0.4, 0.6, 0.8, 1.0),
                     labels = percent_format(accuracy = 1)) +
  labs(x = "名义覆盖率", y = "实际覆盖率") +
  theme_mcup(base_size = 9, legend = "none") +
  annotate("text", x = 0.86, y = 0.40, label = "重复标准差",
           colour = col_method["重复标准差"], hjust = 0.5, size = 2.6) +
  annotate("text", x = 0.86, y = 0.98, label = "残差校准",
           colour = col_method["残差校准"], hjust = 0.5, size = 2.6)

p5 <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1, 1.18)) +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 9))

save_paper(p5, file.path(OUT, "图5_重复验证稳定性与预测校准.png"),
           width = 8.0, height = 4.9, dpi = 300)
ggsave(file.path(OUT, "图5_重复验证稳定性与预测校准.pdf"), p5,
       width = 8.0, height = 4.9, device = cairo_pdf, bg = "white")

rel <- rel %>%
  arrange(r2) %>%
  mutate(panel = ifelse(row_number() <= 25, "较低25项", "较高25项"))

s3_panel <- function(d, show_legend) {
  d <- d %>% mutate(target_code = factor(target_code, levels = target_code[order(r2)]))
  ggplot(d, aes(x = r2, y = target_code, colour = tier)) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey55", linewidth = 0.35) +
    geom_segment(aes(x = r2_ci_low, xend = r2_ci_high,
                     y = target_code, yend = target_code),
                 linewidth = 0.35, alpha = 0.9) +
    geom_point(size = 1.55) +
    scale_colour_manual(values = col_tier, name = "可靠性",
                        limits = names(col_tier), drop = FALSE) +
    scale_x_continuous(limits = c(-0.05, 0.90),
                       breaks = seq(0, 0.8, 0.2)) +
    labs(x = expression("折外 " * R^2 * "（95% bootstrap 区间）"), y = NULL) +
    theme_mcup(base_size = 9, legend = if (show_legend) "right" else "none") +
    theme(axis.text.y = element_text(size = 9.8),
          axis.text.x = element_text(size = 8.8),
          axis.title.x = element_text(size = 9.2),
          legend.text = element_text(size = 9.2),
          legend.title = element_text(size = 8.8))
}

p_s3 <- s3_panel(filter(rel, panel == "较低25项"), FALSE) |
  s3_panel(filter(rel, panel == "较高25项"), TRUE)
p_s3 <- p_s3 + plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 9))

save_paper(p_s3, file.path(OUT, "附图S3_逐目标预测可靠性.png"),
           width = 7.6, height = 6.4, dpi = 300)
ggsave(file.path(OUT, "附图S3_逐目标预测可靠性.pdf"), p_s3,
       width = 7.6, height = 6.4, device = cairo_pdf, bg = "white")

cat("saved:", OUT, "\n")

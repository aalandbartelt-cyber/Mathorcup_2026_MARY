#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
})

source("scripts/plot_style.R")

data_dir <- "03_结果/Q1_感官评分/figure_data"
figure_dir <- "04_图表/Q1_感官评分"
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

# 论文图文件内不写整图标题。图题只保存在 Q1_图清单.csv，
# 后续由 DOCX 在图下方统一排版。面板标识可用 (a)(b)，分面标签无底色并加粗。

group_labels <- c(
  "Aroma" = "香气",
  "Flavour-core" = "风味主体",
  "Taste-other" = "其他味觉",
  "Palate" = "口感",
  "Overall" = "总体"
)
group_colors <- c(
  "Aroma" = "#3B75AF",
  "Flavour-core" = "#EF8636",
  "Taste-other" = "#55A868",
  "Palate" = "#C44E52",
  "Overall" = "#8172B3"
)
tier_colors <- c("高" = "#2563A6", "中" = "#E39C34", "低" = "#8C8C8C")

panel_tags <- plot_annotation(
  tag_levels = "a",
  tag_prefix = "(",
  tag_suffix = ")"
)

# 图1：主模型对比。其他方法使用中性灰，ExtraTrees 作为主结论用深蓝突出。
comparison <- read_csv(file.path(data_dir, "model_comparison.csv"),
                       show_col_types = FALSE)
model_levels <- c("全局均值", "风格收缩均值", "PLS2（秩高斯）",
                  "Ridge（秩高斯）", "ExtraTrees")
model_colors <- c(
  "全局均值" = "#C9C9C9",
  "风格收缩均值" = "#AFAFAF",
  "PLS2（秩高斯）" = "#969696",
  "Ridge（秩高斯）" = "#737373",
  "ExtraTrees" = "#2563A6"
)
comparison <- comparison %>%
  mutate(display_name = factor(display_name, levels = model_levels))
focus_summary <- comparison %>%
  filter(display_name == "ExtraTrees") %>%
  summarise(r2 = mean(macro_r2), mae = mean(macro_mae))

p1a <- ggplot(comparison, aes(display_name, macro_r2, fill = display_name)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.35) +
  geom_boxplot(width = 0.56, linewidth = 0.38, outlier.shape = NA, alpha = 0.88) +
  geom_point(position = position_jitter(width = 0.10, height = 0),
             shape = 21, size = 1.55, stroke = 0.28) +
  annotate("text", x = 5, y = focus_summary$r2 + 0.015,
           label = sprintf("R² = %.3f", focus_summary$r2),
           family = FONT_PAPER, fontface = "bold", colour = "#1D4F91", size = 3.1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.16))) +
  labs(x = NULL, y = "宏平均 R²") +
  theme_mcup(base_size = 9.5, legend = "none") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

p1b <- ggplot(comparison, aes(display_name, macro_mae, fill = display_name)) +
  geom_boxplot(width = 0.56, linewidth = 0.38, outlier.shape = NA, alpha = 0.88) +
  geom_point(position = position_jitter(width = 0.10, height = 0),
             shape = 21, size = 1.55, stroke = 0.28) +
  annotate("text", x = 5, y = focus_summary$mae - 0.0028,
           label = sprintf("MAE = %.3f", focus_summary$mae),
           family = FONT_PAPER, fontface = "bold", colour = "#1D4F91", size = 3.1) +
  scale_fill_manual(values = model_colors) +
  scale_y_continuous(expand = expansion(mult = c(0.14, 0.08))) +
  labs(x = NULL, y = "宏平均 MAE（z 分数）") +
  theme_mcup(base_size = 9.5, legend = "none") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

p1 <- (p1a | p1b) + panel_tags &
  theme(plot.tag = element_text(face = "bold", size = 10.5),
        plot.tag.position = c(0.02, 0.98))
save_paper(p1, file.path(figure_dir, "图1_模型对比.png"),
           width = 8.2, height = 4.35)

# 图2：将两张高纵比图组成横版双联图，并强制使用同一目标顺序。
target_metrics <- read_csv(file.path(data_dir, "target_metrics_extra_trees.csv"),
                           show_col_types = FALSE) %>%
  arrange(r2)
target_order <- target_metrics$target_code
target_metrics <- target_metrics %>%
  mutate(target_code = factor(target_code, levels = target_order))

distribution <- read_csv(file.path(data_dir, "target_distribution.csv.gz"),
                         show_col_types = FALSE) %>%
  mutate(target_code = factor(target_code, levels = target_order))

p2a <- ggplot(distribution, aes(z_score, target_code, fill = target_group)) +
  geom_boxplot(width = 0.66, linewidth = 0.25, outlier.size = 0.30,
               outlier.alpha = 0.40) +
  scale_fill_manual(values = group_colors, labels = group_labels) +
  labs(x = "聚合感官强度（z 分数）", y = NULL) +
  theme_mcup(base_size = 8.7, legend = "none") +
  theme(axis.text.y = element_text(size = 5.8),
        legend.position = "bottom",
        legend.direction = "horizontal")

p2b <- ggplot(target_metrics, aes(r2, target_code, colour = target_group)) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "grey55") +
  geom_segment(aes(x = 0, xend = r2, yend = target_code), linewidth = 0.48) +
  geom_point(size = 2.05) +
  scale_colour_manual(values = group_colors, labels = group_labels) +
  scale_x_continuous(expand = expansion(mult = c(0.03, 0.08))) +
  labs(x = "mean-OOF R²", y = NULL) +
  theme_mcup(base_size = 8.7, legend = "none") +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

p2 <- (p2a | p2b) +
  plot_layout(widths = c(1.18, 0.82)) +
  panel_tags &
  theme(plot.tag = element_text(face = "bold", size = 10),
        plot.tag.position = c(0.015, 0.99))
save_paper(p2, file.path(figure_dir, "图2_目标分布与逐目标R2.png"),
           width = 11.2, height = 6.3)

# 图3：高/中/低各保留一个预登记代表目标，减少重复子图。
representative <- read_csv(file.path(data_dir, "representative_oof.csv.gz"),
                           show_col_types = FALSE)
representative3 <- representative %>%
  filter(target_code %in% c("acidity", "X4vg", "stale_hops")) %>%
  mutate(
    facet_label_clean = case_when(
      target_code == "acidity" ~ "高可预测：acidity   R²=0.73，MAE=0.29",
      target_code == "X4vg" ~ "中可预测：X4vg   R²=0.19，MAE=0.39",
      TRUE ~ "低可预测：stale_hops   R²=0.02，MAE=0.31"
    ),
    facet_label_clean = factor(
      facet_label_clean,
      levels = c(
        "高可预测：acidity   R²=0.73，MAE=0.29",
        "中可预测：X4vg   R²=0.19，MAE=0.39",
        "低可预测：stale_hops   R²=0.02，MAE=0.31"
      )
    )
  )

p3 <- ggplot(representative3, aes(y_true, y_pred_mean)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "#555555",
              linewidth = 0.48) +
  geom_point(aes(colour = tier), size = 1.45, alpha = 0.64) +
  facet_wrap(~facet_label_clean, nrow = 1) +
  scale_colour_manual(values = tier_colors) +
  labs(x = "观测值（z 分数）", y = "OOF 预测值（z 分数）") +
  theme_mcup(base_size = 9.5, legend = "none") +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "bold", size = 8.5,
                                  margin = margin(1, 0, 4, 0)))
save_paper(p3, file.path(figure_dir, "图3_OOF观测预测.png"),
           width = 8.8, height = 4.2)

# 图4：将风格稳健性与 0—6 输出口径组成横版双联图。
style_metrics <- read_csv(file.path(data_dir, "style_metrics_extra_trees.csv"),
                          show_col_types = FALSE) %>%
  arrange(mae_all_targets) %>%
  mutate(tasting_category_fine = factor(tasting_category_fine,
                                        levels = tasting_category_fine))
p4a <- ggplot(style_metrics,
              aes(mae_all_targets, tasting_category_fine,
                  colour = mae_all_targets)) +
  geom_segment(aes(x = min(mae_all_targets), xend = mae_all_targets,
                   yend = tasting_category_fine), colour = "grey78", linewidth = 0.45) +
  geom_point(aes(size = n_beers), alpha = 0.93) +
  geom_text(aes(label = paste0("n=", n_beers)), colour = "black",
            size = 2.35, hjust = -0.32) +
  scale_colour_gradient(low = "#2C7FB8", high = "#C43C4E", guide = "none") +
  scale_size_continuous(range = c(2.0, 5.0), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.21))) +
  labs(x = "全部目标的 MAE（z 分数）", y = NULL) +
  theme_mcup(base_size = 8.8, legend = "none") +
  theme(axis.text.y = element_text(size = 7.2))

mapping <- read_csv(file.path(data_dir, "scale_mapping_example.csv"),
                    show_col_types = FALSE)
p4b <- ggplot(mapping, aes(z_score, grade_0_6, colour = target_code)) +
  geom_step(linewidth = 0.78, direction = "hv") +
  scale_colour_manual(values = c("#2C7FB8", "#D1495B", "#55A868", "#8172B3")) +
  scale_y_continuous(breaks = 0:6, limits = c(0, 6)) +
  labs(x = "模型预测的聚合 z 分数", y = "0—6 相对强度等级") +
  guides(colour = guide_legend(ncol = 2, byrow = TRUE)) +
  theme_mcup(base_size = 8.8, legend = "none") +
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.text = element_text(size = 7.2))

p4 <- (p4a | p4b) +
  plot_layout(widths = c(1.12, 0.88)) +
  panel_tags &
  theme(plot.tag = element_text(face = "bold", size = 10),
        plot.tag.position = c(0.015, 0.99))
save_paper(p4, file.path(figure_dir, "图4_风格误差与量表映射.png"),
           width = 10.5, height = 5.9)

# 附图 S1：保留诊断价值，但不占用正文图数。
p_s1 <- ggplot(representative3, aes(y_pred_mean, residual)) +
  geom_hline(yintercept = 0, linetype = 2, colour = "#555555", linewidth = 0.45) +
  geom_point(aes(colour = tier), size = 1.30, alpha = 0.58) +
  geom_smooth(method = "loess", formula = y ~ x, se = TRUE,
              colour = "#1F1F1F", fill = "#C9C9C9", linewidth = 0.62,
              alpha = 0.24) +
  facet_wrap(~facet_label_clean, nrow = 1) +
  scale_colour_manual(values = tier_colors) +
  labs(x = "OOF 预测值（z 分数）", y = "残差（观测−预测）") +
  theme_mcup(base_size = 9.5, legend = "none") +
  theme(strip.background = element_blank(),
        strip.text = element_text(face = "bold", size = 8.5,
                                  margin = margin(1, 0, 4, 0)))
save_paper(p_s1, file.path(figure_dir, "附图S1_残差诊断.png"),
           width = 8.8, height = 4.2)

manifest <- tibble::tribble(
  ~figure_no, ~role, ~path, ~caption, ~explanation,
  "1", "main", file.path(figure_dir, "图1_模型对比.png"),
  "图1 五类基线在重复分层外层交叉验证中的性能比较",
  "ExtraTrees 同时提高宏平均 R² 并降低 MAE，散点表示 10 次重复划分。",
  "2", "main", file.path(figure_dir, "图2_目标分布与逐目标R2.png"),
  "图2 50 个感官目标的分布与 ExtraTrees 逐目标解释度",
  "两面板共用按 R² 排序的目标顺序，显示分布形态与可预测性的异质性。",
  "3", "main", file.path(figure_dir, "图3_OOF观测预测.png"),
  "图3 高、中、低可预测感官目标的 OOF 观测—预测关系",
  "acidity、X4vg 和 stale_hops 展示了预测性差异及低信号目标的回归到均值。",
  "4", "main", file.path(figure_dir, "图4_风格误差与量表映射.png"),
  "图4 不同啤酒风格的 OOF 误差与训练折经验分布的 0—6 映射",
  "左图显示风格层面误差与样本量，右图说明 0—6 输出是单调相对等级。",
  "S1", "supplementary", file.path(figure_dir, "附图S1_残差诊断.png"),
  "附图S1 代表感官目标的折外残差诊断",
  "LOESS 平滑线用于检查非线性偏差、异方差与极端区间系统误差。"
)
write_csv(manifest, file.path(figure_dir, "Q1_图清单.csv"))
cat("Generated 4 main figures and 1 supplementary figure at", figure_dir, "\n")

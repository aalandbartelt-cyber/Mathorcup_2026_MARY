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

# 图1：50 个目标在聚合 z 尺度上的分布。
distribution <- read_csv(file.path(data_dir, "target_distribution.csv.gz"),
                         show_col_types = FALSE) %>%
  mutate(target_group_cn = recode(target_group, !!!group_labels))
target_order <- distribution %>%
  group_by(target_code) %>%
  summarise(median_z = median(z_score), .groups = "drop") %>%
  arrange(median_z) %>% pull(target_code)
distribution <- distribution %>%
  mutate(target_code = factor(target_code, levels = target_order))
p1 <- ggplot(distribution, aes(z_score, target_code, fill = target_group)) +
  geom_boxplot(width = 0.68, linewidth = 0.28, outlier.size = 0.35,
               outlier.alpha = 0.45) +
  scale_fill_manual(values = group_colors, labels = group_labels) +
  labs(x = "聚合感官强度（z 分数）", y = NULL,
       title = "50 个感官目标的样本分布") +
  theme_mcup(base_size = 9, legend = "tr") +
  theme(axis.text.y = element_text(size = 7.2))
save_paper(p1, file.path(figure_dir, "图1_目标分布.png"), width = 7.2, height = 9.8)

# 图2：每次重复外层 OOF 的宏平均表现。
comparison <- read_csv(file.path(data_dir, "model_comparison.csv"),
                       show_col_types = FALSE)
model_levels <- c("全局均值", "风格收缩均值", "PLS2（秩高斯）",
                  "Ridge（秩高斯）", "ExtraTrees")
comparison <- comparison %>%
  mutate(display_name = factor(display_name, levels = model_levels))
p2a <- ggplot(comparison, aes(display_name, macro_r2, fill = display_name)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.35) +
  geom_boxplot(width = 0.56, linewidth = 0.35, outlier.shape = NA, alpha = 0.80) +
  geom_point(position = position_jitter(width = 0.10, height = 0),
             shape = 21, size = 1.45, stroke = 0.25) +
  scale_fill_manual(values = c("#B7B7B7", "#4C78A8", "#F2A65A", "#72B7B2", "#D1495B")) +
  labs(x = NULL, y = "宏平均 R²", title = "解释度") +
  theme_mcup(base_size = 9, legend = "none") +
  theme(axis.text.x = element_text(angle = 28, hjust = 1))
p2b <- ggplot(comparison, aes(display_name, macro_mae, fill = display_name)) +
  geom_boxplot(width = 0.56, linewidth = 0.35, outlier.shape = NA, alpha = 0.80) +
  geom_point(position = position_jitter(width = 0.10, height = 0),
             shape = 21, size = 1.45, stroke = 0.25) +
  scale_fill_manual(values = c("#B7B7B7", "#4C78A8", "#F2A65A", "#72B7B2", "#D1495B")) +
  labs(x = NULL, y = "宏平均 MAE（z 分数）", title = "绝对误差") +
  theme_mcup(base_size = 9, legend = "none") +
  theme(axis.text.x = element_text(angle = 28, hjust = 1))
p2 <- (p2a | p2b) + plot_annotation(title = "重复分层外层交叉验证的模型比较") &
  theme(plot.title = element_text(hjust = 0.5, family = FONT_PAPER, size = 10.5))
save_paper(p2, file.path(figure_dir, "图2_模型对比.png"), width = 8.0, height = 4.4)

# 图3：高、中、低可预测性的代表目标。
representative <- read_csv(file.path(data_dir, "representative_oof.csv.gz"),
                           show_col_types = FALSE)
facet_levels <- representative %>%
  distinct(facet_label, tier, target_code) %>%
  mutate(tier = factor(tier, levels = c("高", "中", "低"))) %>%
  arrange(tier, target_code) %>%
  pull(facet_label)
representative <- representative %>%
  mutate(facet_label = factor(facet_label, levels = facet_levels))
p3 <- ggplot(representative, aes(y_true, y_pred_mean)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "#555555",
              linewidth = 0.45) +
  geom_point(aes(colour = tier), size = 1.25, alpha = 0.68) +
  facet_wrap(~facet_label, ncol = 3) +
  scale_colour_manual(values = c("低" = "#9E9E9E", "中" = "#E39C34", "高" = "#2C7FB8")) +
  labs(x = "观测值（z 分数）", y = "OOF 预测值（z 分数）",
       title = "代表感官目标的观测—预测关系") +
  theme_mcup(base_size = 9, legend = "none")
save_paper(p3, file.path(figure_dir, "图3_OOF观测预测.png"), width = 8.0, height = 5.2)

# 图4：所有目标的 mean-OOF R²。
target_metrics <- read_csv(file.path(data_dir, "target_metrics_extra_trees.csv"),
                           show_col_types = FALSE) %>%
  arrange(r2) %>%
  mutate(target_code = factor(target_code, levels = target_code))
p4 <- ggplot(target_metrics, aes(r2, target_code, colour = target_group)) +
  geom_vline(xintercept = 0, linewidth = 0.35, colour = "grey55") +
  geom_segment(aes(x = 0, xend = r2, yend = target_code), linewidth = 0.45) +
  geom_point(size = 2.0) +
  scale_colour_manual(values = group_colors, labels = group_labels) +
  scale_x_continuous(expand = expansion(mult = c(0.03, 0.07))) +
  labs(x = "10 次重复预测均值的 OOF R²", y = NULL,
       title = "ExtraTrees 对 50 个感官目标的解释度") +
  theme_mcup(base_size = 9, legend = "tr") +
  theme(axis.text.y = element_text(size = 7.2))
save_paper(p4, file.path(figure_dir, "图4_逐目标R2棒棒糖.png"), width = 7.2, height = 9.8)

# 图5：风格层面的误差与样本量。
style_metrics <- read_csv(file.path(data_dir, "style_metrics_extra_trees.csv"),
                          show_col_types = FALSE) %>%
  arrange(mae_all_targets) %>%
  mutate(tasting_category_fine = factor(tasting_category_fine,
                                        levels = tasting_category_fine))
p5 <- ggplot(style_metrics,
             aes(mae_all_targets, tasting_category_fine,
                 colour = mae_all_targets)) +
  geom_segment(aes(x = min(mae_all_targets), xend = mae_all_targets,
                   yend = tasting_category_fine), colour = "grey75", linewidth = 0.45) +
  geom_point(aes(size = n_beers), alpha = 0.92) +
  geom_text(aes(label = paste0("n=", n_beers)), colour = "black",
            size = 2.5, hjust = -0.35) +
  scale_colour_gradient(low = "#2C7FB8", high = "#D1495B", guide = "none") +
  scale_size_continuous(range = c(2.2, 5.5), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.18))) +
  labs(x = "全部目标的平均绝对误差（z 分数）", y = NULL,
       title = "不同啤酒风格的 OOF 预测误差") +
  theme_mcup(base_size = 9, legend = "none")
save_paper(p5, file.path(figure_dir, "图5_风格误差.png"), width = 7.0, height = 6.2)

# 图6：仅使用一个外层训练折构造的经验 CDF 映射示例。
mapping <- read_csv(file.path(data_dir, "scale_mapping_example.csv"),
                    show_col_types = FALSE)
p6 <- ggplot(mapping, aes(z_score, grade_0_6, colour = target_code)) +
  geom_step(linewidth = 0.72, direction = "hv") +
  scale_colour_manual(values = c("#2C7FB8", "#D1495B", "#55A868", "#8172B3")) +
  scale_y_continuous(breaks = 0:6, limits = c(0, 6)) +
  labs(x = "模型预测的聚合 z 分数", y = "0—6 相对强度等级",
       title = "训练折经验分布定义的 0—6 单调映射") +
  theme_mcup(base_size = 9, legend = "right")
save_paper(p6, file.path(figure_dir, "图6_量表映射.png"), width = 6.8, height = 4.4)

# 图7：与图3相同代表目标的折外残差诊断。
p7 <- ggplot(representative, aes(y_pred_mean, residual)) +
  geom_hline(yintercept = 0, linetype = 2, colour = "#555555", linewidth = 0.45) +
  geom_point(aes(colour = tier), size = 1.1, alpha = 0.55) +
  geom_smooth(method = "loess", formula = y ~ x, se = TRUE,
              colour = "#222222", fill = "#BDBDBD", linewidth = 0.55,
              alpha = 0.28) +
  facet_wrap(~facet_label, ncol = 3) +
  scale_colour_manual(values = c("低" = "#9E9E9E", "中" = "#E39C34", "高" = "#2C7FB8")) +
  labs(x = "OOF 预测值（z 分数）", y = "残差（观测−预测）",
       title = "代表感官目标的折外残差诊断") +
  theme_mcup(base_size = 9, legend = "none")
save_paper(p7, file.path(figure_dir, "图7_残差诊断.png"), width = 8.0, height = 5.2)

manifest <- tibble::tribble(
  ~figure_no, ~path, ~caption, ~explanation,
  1, file.path(figure_dir, "图1_目标分布.png"),
  "图1 50 个感官目标在聚合 z 尺度上的分布",
  "展示不同目标的中心、离散程度与异常值，说明需按目标报告结果。",
  2, file.path(figure_dir, "图2_模型对比.png"),
  "图2 五类基线在重复分层外层交叉验证中的性能比较",
  "ExtraTrees 同时提高宏平均 R² 并降低 MAE，箱体反映划分不确定性。",
  3, file.path(figure_dir, "图3_OOF观测预测.png"),
  "图3 高、中、低可预测感官目标的 OOF 观测—预测关系",
  "六个代表目标揭示可预测性差异和回归到均值现象。",
  4, file.path(figure_dir, "图4_逐目标R2棒棒糖.png"),
  "图4 ExtraTrees 对 50 个感官目标的 mean-OOF 解释度",
  "全部目标 R² 为正，但不同属性的解释度差异明显。",
  5, file.path(figure_dir, "图5_风格误差.png"),
  "图5 不同啤酒风格的 OOF 平均绝对误差",
  "结合点大小所示样本量，识别小样本及高误差风格。",
  6, file.path(figure_dir, "图6_量表映射.png"),
  "图6 训练折经验分布定义的 z 分数到 0—6 相对等级映射",
  "说明题面量表是在训练折内单调重标定，而非对原始评分的逆变换。",
  7, file.path(figure_dir, "图7_残差诊断.png"),
  "图7 代表感官目标的折外残差诊断",
  "平滑线用于检查非线性偏差、异方差及极端区间系统误差。"
)
write_csv(manifest, file.path(figure_dir, "Q1_图清单.csv"))
cat("Generated", nrow(manifest), "figures at", figure_dir, "\n")

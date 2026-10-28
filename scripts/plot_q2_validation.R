#!/usr/bin/env Rscript
# Q2 validation figure: negative controls, shadow audit, anchor recovery, deletion sensitivity.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(scales)
})
source("scripts/plot_style.R")

Q2 <- path.expand("~/mathorcup/.worktrees/q2-evidence-opt")
VAL <- file.path(Q2, "03_结果/Q2_关键化合物/validation_v1")
OUT <- file.path(Q2, "04_图表/Q2_关键化合物/validation_v1")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

neg <- read.csv(file.path(VAL, "q2_validation_negative_controls.csv"), check.names = FALSE)
anc <- read.csv(file.path(VAL, "q2_validation_anchor.csv"), check.names = FALSE)
del <- read.csv(file.path(VAL, "q2_validation_deletion.csv"), check.names = FALSE)
e112 <- read.csv(file.path(Q2, "03_结果/Q2_关键化合物/q2_final_top3_E112.csv"),
                 check.names = FALSE)

# (a) null target permutation: real vs null at pre-registered thresholds
null_df <- neg %>%
  filter(control == "null target permutation") %>%
  mutate(threshold = factor(sprintf("%.1f", threshold), levels = c("0.6", "0.7", "0.8", "0.9"))) %>%
  tidyr::pivot_longer(cols = c(real_selected, null_or_shadow_selected),
                      names_to = "type", values_to = "n") %>%
  mutate(type = recode(type,
                       "real_selected" = "真实关系",
                       "null_or_shadow_selected" = "目标置换"))

p_a <- ggplot(null_df, aes(threshold, n, fill = type)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.62) +
  geom_text(aes(label = n), position = position_dodge(width = 0.72),
            vjust = -0.25, size = 2.5) +
  scale_fill_manual(values = c("真实关系" = "#367DA2", "目标置换" = "#C88240")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(x = "入选频率阈值", y = "入选关系数") +
  theme_mcup(base_size = 9, legend = "tr") +
  theme(legend.text = element_text(size = 7.2))

# (b) shadow feature audit: total selections over 2000 records
shadow <- neg %>%
  filter(control == "shadow feature audit") %>%
  mutate(real_rate = real_selected / denominator,
         shadow_rate = null_or_shadow_selected / denominator) %>%
  tidyr::pivot_longer(cols = c(real_rate, shadow_rate),
                      names_to = "type", values_to = "rate") %>%
  mutate(type = recode(type,
                       "real_rate" = "真实字段",
                       "shadow_rate" = "影子字段"))

p_b <- ggplot(shadow, aes(type, rate, fill = type)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = sprintf("%.2f", rate)), vjust = -0.25, size = 2.6) +
  scale_fill_manual(values = c("真实字段" = "#367DA2", "影子字段" = "#C88240")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(x = NULL, y = "每条记录平均入选次数") +
  theme_mcup(base_size = 9, legend = "none")

# (c) anchor recovery: bootstrap rank; top-3 threshold
label_map <- c(
  "X4vg" = "4-乙烯基愈创木酚",
  "alcohol" = "酒精感（乙醇）",
  "body" = "酒体（乙醇）",
  "co2" = "碳酸感（CO2）",
  "dms" = "DMS",
  "diacetyl" = "双乙酰",
  "t2n" = "反-2-壬烯醛"
)
anc2 <- anc %>%
  filter(status != "not_a_compound_candidate") %>%
  mutate(anchor_label = unname(label_map[target_code]),
         anchor_label = reorder(anchor_label, -bootstrap_rank),
         status = factor(status, levels = c("recovered_in_final_top3",
                                            "recovered_in_bootstrap_top3",
                                            "not_recovered")))
p_c <- ggplot(anc2, aes(bootstrap_rank, anchor_label, colour = status)) +
  geom_vline(xintercept = 3, linetype = 2, colour = "grey45", linewidth = 0.4) +
  geom_segment(aes(x = 1, xend = bootstrap_rank, y = anchor_label, yend = anchor_label),
               linewidth = 0.35, alpha = 0.35) +
  geom_point(size = 2.1) +
  scale_x_continuous(trans = "log10", breaks = c(1, 3, 10, 30, 100),
                     limits = c(1, 120)) +
  scale_colour_manual(values = c("recovered_in_final_top3" = "#32765F",
                                 "recovered_in_bootstrap_top3" = "#367DA2",
                                 "not_recovered" = "#C88240"),
                      labels = c("最终 Top-3", "bootstrap Top-3", "未恢复")) +
  labs(x = "bootstrap 名次（对数轴；虚线=第3名）", y = NULL) +
  theme_mcup(base_size = 8.5, legend = "br") +
  theme(axis.text.y = element_text(size = 7.0),
        legend.text = element_text(size = 6.8))

# (d) deletion sensitivity by target group
grp <- e112 %>% distinct(target_code, target_group)
del2 <- del %>%
  left_join(grp, by = "target_code") %>%
  mutate(group_plot = recode(target_group,
                             "Aroma" = "香气",
                             "Flavour-core" = "风味",
                             "Taste-other" = "味觉",
                             "Palate" = "口感",
                             "Overall" = "总体",
                             .default = "其他"),
         group_plot = factor(group_plot, levels = c("香气", "风味", "味觉", "口感", "总体")))
p_d <- ggplot(del2, aes(group_plot, mean_delta_mae, fill = group_plot)) +
  geom_hline(yintercept = 0, linetype = 2, colour = "grey45", linewidth = 0.35) +
  geom_boxplot(width = 0.58, outlier.size = 0.6, linewidth = 0.35) +
  geom_jitter(width = 0.12, size = 0.55, alpha = 0.45, show.legend = FALSE) +
  scale_fill_manual(values = c("香气" = "#367DA2", "风味" = "#32765F",
                               "味觉" = "#C88240", "口感" = "#6E6E6E",
                               "总体" = "#9E7BB5")) +
  labs(x = NULL, y = "删除 Top-3 后 ΔMAE（>0 表示变差）") +
  theme_mcup(base_size = 9, legend = "none")

p5 <- (p_a | p_b) / (p_c | p_d) +
  plot_layout(heights = c(1, 1.15)) +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 9))

save_paper(p5, file.path(OUT, "图5_负对照与锚点恢复.png"),
           width = 8.2, height = 5.2, dpi = 300)
ggsave(file.path(OUT, "图5_负对照与锚点恢复.pdf"), p5,
       width = 8.2, height = 5.2, device = cairo_pdf, bg = "white")
cat("saved:", OUT, "\n")

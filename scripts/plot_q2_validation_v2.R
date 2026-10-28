#!/usr/bin/env Rscript
# Q2 validation figure 5, v2: negative controls + direct anchor recovery only.
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

shadow <- neg %>% filter(control == "shadow feature audit") %>% slice(1)
real_rate <- shadow$real_selected / shadow$denominator
shadow_rate <- shadow$null_or_shadow_selected / shadow$denominator

neg_df <- data.frame(
  label = factor(c("目标置换", "影子字段", "真实字段"),
                 levels = c("目标置换", "影子字段", "真实字段")),
  rate = c(0, shadow_rate, real_rate),
  note = c("四个阈值均为 0", sprintf("%d 次 / %d 条记录", shadow$null_or_shadow_selected, shadow$denominator),
           sprintf("%d 次 / %d 条记录", shadow$real_selected, shadow$denominator))
)

p_a <- ggplot(neg_df, aes(rate, label, fill = label)) +
  geom_col(width = 0.55) +
  geom_text(aes(label = sprintf("%.2f", rate)), hjust = -0.12, size = 2.8) +
  scale_fill_manual(values = c("目标置换" = "#9A9A9A",
                               "影子字段" = "#C88240",
                               "真实字段" = "#367DA2")) +
  scale_x_continuous(limits = c(0, 2.75), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "每条记录平均入选次数", y = NULL) +
  theme_mcup(base_size = 9, legend = "none")

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

p_b <- ggplot(anc2, aes(bootstrap_rank, anchor_label, colour = status)) +
  geom_vline(xintercept = 3, linetype = 2, colour = "grey45", linewidth = 0.4) +
  geom_segment(aes(x = 1, xend = bootstrap_rank, y = anchor_label, yend = anchor_label),
               linewidth = 0.35, alpha = 0.35) +
  geom_point(size = 2.2) +
  scale_x_continuous(trans = "log10", breaks = c(1, 3, 10, 30, 100),
                     limits = c(1, 120)) +
  scale_colour_manual(values = c("recovered_in_final_top3" = "#32765F",
                                 "recovered_in_bootstrap_top3" = "#367DA2",
                                 "not_recovered" = "#C88240"),
                      labels = c("最终 Top-3", "bootstrap Top-3", "未恢复")) +
  labs(x = "bootstrap 名次（对数轴；虚线=第 3 名）", y = NULL) +
  theme_mcup(base_size = 9, legend = "tr") +
  theme(axis.text.y = element_text(size = 7.6),
        legend.text = element_text(size = 7.0))

p5 <- (p_a | p_b) +
  plot_layout(widths = c(1, 1.35)) +
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")") &
  theme(plot.tag = element_text(face = "bold", size = 9))

save_paper(p5, file.path(OUT, "图5_负对照与直接锚点.png"),
           width = 8.0, height = 3.6, dpi = 300)
ggsave(file.path(OUT, "图5_负对照与直接锚点.pdf"), p5,
       width = 8.0, height = 3.6, device = cairo_pdf, bg = "white")
cat("saved:", OUT, "\n")

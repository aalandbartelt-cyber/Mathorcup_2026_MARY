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
figure_dir <- Sys.getenv(
  "Q1_FIGURE_DIR",
  unset = "04_图表/Q1_感官评分/final_v3_draft"
)
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

# 原始字段代码继续留在数据层；论文图的刻度与图例使用中文展示名。
target_labels_zh <- c(
  "A_malt_all" = "麦芽香气-总体",
  "A_malt_grain" = "麦芽香气-谷物",
  "A_malt_bread" = "麦芽香气-面包",
  "A_malt_cara" = "麦芽香气-焦糖",
  "A_malt_burn" = "麦芽香气-烟熏",
  "A_hops_all" = "酒花香气-总体",
  "A_hops_citrus" = "酒花香气-柑橘",
  "A_hops_tropical" = "酒花香气-热带水果",
  "A_hops_noble" = "酒花香气-青草",
  "A_hops_woody" = "酒花香气-木质",
  "A_esters_all" = "酯类香气-总体",
  "A_esters_ethac" = "酯类香气-指甲油",
  "A_esters_isoaa" = "酯类香气-香蕉",
  "A_esters_flower" = "酯类香气-花香",
  "A_esters_fruity" = "酯类香气-果香",
  "F_malt_all" = "麦芽风味-总体",
  "F_malt_grain" = "麦芽风味-谷物",
  "F_malt_bread" = "麦芽风味-面包",
  "F_malt_cara" = "麦芽风味-焦糖",
  "F_malt_burn" = "麦芽风味-烟熏",
  "F_hops_all" = "酒花风味-总体",
  "F_hops_citrus" = "酒花风味-柑橘",
  "F_hops_tropical" = "酒花风味-热带水果",
  "F_hops_noble" = "酒花风味-青草",
  "F_hops_woody" = "酒花风味-木质",
  "F_esters_all" = "酯类风味-总体",
  "F_esters_ethac" = "酯类风味-指甲油",
  "F_esters_isoaa" = "酯类风味-香蕉",
  "F_esters_flower" = "酯类风味-花香",
  "F_esters_fruity" = "酯类风味-果香",
  "acidity" = "酸度",
  "bitternes" = "苦味",
  "sweetness" = "甜味",
  "X4vg" = "4VG",
  "diacetyl" = "双乙酰",
  "dms" = "二甲基硫（DMS）",
  "metallic" = "金属味",
  "stale_hops" = "陈旧酒花味",
  "t2n" = "反式-2-壬烯醛",
  "orange" = "橙香",
  "coriander" = "芫荽",
  "clove" = "丁香",
  "lactic" = "乳酸味",
  "acetic" = "乙酸味",
  "barnyard" = "畜棚味",
  "alcohol" = "酒精感",
  "aftertaste" = "回味",
  "body" = "酒体",
  "co2" = "碳酸感",
  "overall" = "总体喜好"
)

style_labels_zh <- c(
  "Amber" = "琥珀艾尔", "Blond" = "金色艾尔",
  "Brett/cofermented" = "Brett/混合发酵", "Brown" = "棕色艾尔",
  "Brut" = "Brut（干型）", "Christmas" = "圣诞啤酒",
  "Dubbel" = "双料啤酒", "Faro" = "法柔",
  "Flanders old brown" = "佛兰德老棕艾尔", "Fruitbeer" = "水果啤酒",
  "Hoppy" = "酒花型", "Kriek" = "克里克（樱桃）",
  "Lambic" = "兰比克", "Low/No alcohol" = "低/无醇",
  "Pils/Lager" = "皮尔森/拉格", "Saison" = "赛松",
  "Scotch" = "苏格兰艾尔", "Stout/Porter" = "世涛/波特",
  "Strong ale" = "强艾尔", "Tripel" = "三料啤酒",
  "West Flanders ale" = "西佛兰德艾尔", "Wheat" = "小麦啤酒"
)

panel_header_theme <- theme(
  plot.subtitle = element_text(
    colour = "black", face = "bold", size = 9.5, hjust = 0,
    margin = margin(b = 5)
  )
)

# 图1：主模型对比。五个模型用低饱和色区分，ExtraTrees 以深蓝作为主视觉。
comparison <- read_csv(file.path(data_dir, "model_comparison.csv"),
                       show_col_types = FALSE)
model_levels <- c("全局均值", "风格收缩均值", "PLS2（秩高斯）",
                  "Ridge（秩高斯）", "ExtraTrees")
model_colors <- c(
  "全局均值" = "#88A9D3",
  "风格收缩均值" = "#67A995",
  "PLS2（秩高斯）" = "#D4A35B",
  "Ridge（秩高斯）" = "#9A82B8",
  "ExtraTrees" = "#1F5A94"
)
model_outlines <- c(
  "全局均值" = "#4F719B",
  "风格收缩均值" = "#3F7568",
  "PLS2（秩高斯）" = "#8C6735",
  "Ridge（秩高斯）" = "#66527F",
  "ExtraTrees" = "#123A65"
)
comparison <- comparison %>%
  mutate(display_name = factor(display_name, levels = model_levels))
focus_summary <- comparison %>%
  filter(display_name == "ExtraTrees") %>%
  summarise(
    r2 = mean(macro_r2),
    mae = mean(macro_mae),
    r2_label_y = max(macro_r2) + 0.008,
    mae_label_y = min(macro_mae) - 0.0025
  ) %>%
  mutate(display_name = factor("ExtraTrees", levels = model_levels))

p1a <- ggplot(comparison, aes(display_name, macro_r2,
                              fill = display_name, colour = display_name)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.35) +
  geom_boxplot(width = 0.56, linewidth = 0.38, outlier.shape = NA, alpha = 0.88) +
  geom_point(position = position_jitter(width = 0.10, height = 0,
                                        seed = 20261023),
             shape = 21, size = 1.55, stroke = 0.28) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.35,
               stroke = 0.45, fill = "white", colour = "black") +
  geom_text(
    data = focus_summary,
    aes(x = display_name, y = r2_label_y, label = sprintf("%.3f", r2)),
    inherit.aes = FALSE, colour = "#123A65", size = 3.0,
    fontface = "bold", vjust = 0
  ) +
  scale_fill_manual(values = model_colors) +
  scale_colour_manual(values = model_outlines) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.16))) +
  labs(x = NULL, y = "宏平均 R²", subtitle = "（a）解释度") +
  theme_mcup(base_size = 9.5, legend = "none") +
  panel_header_theme +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

p1b <- ggplot(comparison, aes(display_name, macro_mae,
                              fill = display_name, colour = display_name)) +
  geom_boxplot(width = 0.56, linewidth = 0.38, outlier.shape = NA, alpha = 0.88) +
  geom_point(position = position_jitter(width = 0.10, height = 0,
                                        seed = 20261023),
             shape = 21, size = 1.55, stroke = 0.28) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.35,
               stroke = 0.45, fill = "white", colour = "black") +
  geom_text(
    data = focus_summary,
    aes(x = display_name, y = mae_label_y, label = sprintf("%.3f", mae)),
    inherit.aes = FALSE, colour = "#123A65", size = 3.0,
    fontface = "bold", vjust = 1
  ) +
  scale_fill_manual(values = model_colors) +
  scale_colour_manual(values = model_outlines) +
  scale_y_continuous(expand = expansion(mult = c(0.14, 0.08))) +
  labs(x = NULL, y = "宏平均 MAE（z 分数）", subtitle = "（b）绝对误差") +
  theme_mcup(base_size = 9.5, legend = "none") +
  panel_header_theme +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

p1 <- (p1a | p1b)
save_paper(p1, file.path(figure_dir, "图1_模型对比.png"),
           width = 8.2, height = 4.35)

# 图2：逐目标 OOF 性能图谱。三栏分别承载香气、风味和其余目标，
# 每行用同一条短连线比较 R² 与 Spearman，避免 50 行双栏图的视觉拥挤。
target_metrics <- read_csv(file.path(data_dir, "target_metrics_extra_trees.csv"),
                           show_col_types = FALSE)
target_metrics <- target_metrics %>%
  mutate(
    panel_group = case_when(
      target_group == "Aroma" ~ "香气",
      target_group == "Flavour-core" ~ "风味",
      TRUE ~ "味觉·口感·总体"
    ),
    detail_order = case_when(
      target_group == "Taste-other" ~ 1L,
      target_group == "Palate" ~ 2L,
      target_group == "Overall" ~ 3L,
      TRUE ~ 1L
    )
  )

target_panel_plot <- function(data, panel_title, y_text_size = 6.5,
                              separators = numeric()) {
  panel_data <- data %>%
    arrange(detail_order, r2) %>%
    mutate(
      target_label = unname(target_labels_zh[target_code]),
      target_label = factor(target_label, levels = target_label)
    )
  metric_long <- bind_rows(
    panel_data %>% transmute(target_label, value = r2, metric = "R²"),
    panel_data %>% transmute(target_label, value = spearman,
                             metric = "Spearman")
  )

  plot <- ggplot(panel_data, aes(y = target_label)) +
    geom_vline(xintercept = c(0.2, 0.4, 0.6, 0.8),
               colour = "#E5E9EF", linewidth = 0.34) +
    geom_segment(aes(x = r2, xend = spearman, yend = target_label),
                 colour = "#C9D0DA", linewidth = 0.62, lineend = "round") +
    geom_point(data = metric_long,
               aes(x = value, shape = metric, fill = metric, colour = metric),
               size = 2.05, stroke = 0.72) +
    scale_shape_manual(values = c("R²" = 21, "Spearman" = 23)) +
    scale_fill_manual(values = c("R²" = "#1F5A94", "Spearman" = "white")) +
    scale_colour_manual(values = c("R²" = "#123A65",
                                   "Spearman" = "#D06B32")) +
    scale_x_continuous(limits = c(0, 0.9), breaks = seq(0, 0.8, 0.2),
                       expand = expansion(mult = c(0.015, 0.02))) +
    labs(x = "OOF 性能", y = NULL, subtitle = panel_title) +
    theme_mcup(base_size = 8.6, legend = "right") +
    panel_header_theme +
    theme(
      axis.text.y = element_text(size = y_text_size),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.text = element_text(size = 7.2),
      plot.margin = margin(4, 5, 3, 4)
    )

  if (length(separators) > 0) {
    plot <- plot + geom_hline(
      yintercept = separators, colour = "#9AA3AE",
      linewidth = 0.42, linetype = "dotted"
    )
  }
  plot
}

p2a <- target_panel_plot(
  filter(target_metrics, panel_group == "香气"),
  "（a）香气目标"
)
p2b <- target_panel_plot(
  filter(target_metrics, panel_group == "风味"),
  "（b）风味目标"
)
p2c <- target_panel_plot(
  filter(target_metrics, panel_group == "味觉·口感·总体"),
  "（c）味觉、口感与总体", y_text_size = 6.0,
  separators = c(15.5, 19.5)
)

p2 <- (p2a | p2b | p2c) +
  plot_layout(widths = c(1, 1, 1), guides = "collect") &
  theme(legend.position = "bottom")
save_paper(p2, file.path(figure_dir, "图2_目标分布与逐目标R2.png"),
           width = 11.2, height = 6.3)

# 图3：高/中/低各保留一个预登记代表目标，减少重复子图。
representative <- read_csv(file.path(data_dir, "representative_oof.csv.gz"),
                           show_col_types = FALSE)
representative3 <- representative %>%
  filter(target_code %in% c("acidity", "X4vg", "stale_hops")) %>%
  mutate(
    facet_label_clean = case_when(
      target_code == "acidity" ~ "高可预测：酸度   R²=0.73，MAE=0.29",
      target_code == "X4vg" ~ "中可预测：4VG   R²=0.19，MAE=0.39",
      TRUE ~ "低可预测：陈旧酒花味   R²=0.02，MAE=0.31"
    ),
    facet_label_clean = factor(
      facet_label_clean,
      levels = c(
        "高可预测：酸度   R²=0.73，MAE=0.29",
        "中可预测：4VG   R²=0.19，MAE=0.39",
        "低可预测：陈旧酒花味   R²=0.02，MAE=0.31"
      )
    )
  )

p3 <- ggplot(representative3, aes(y_true, y_pred_mean)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "#555555",
              linewidth = 0.48) +
  geom_smooth(aes(colour = tier, fill = tier), method = "lm", formula = y ~ x,
              se = TRUE, linewidth = 0.68, alpha = 0.12) +
  geom_point(aes(colour = tier), size = 1.25, alpha = 0.52) +
  facet_wrap(~facet_label_clean, nrow = 1) +
  scale_colour_manual(values = tier_colors) +
  scale_fill_manual(values = tier_colors) +
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
mean_oof_mae <- mean(target_metrics$mae)
p4a <- ggplot(style_metrics,
              aes(mae_all_targets, tasting_category_fine,
                  colour = mae_all_targets)) +
  geom_vline(xintercept = mean_oof_mae, linetype = 2,
             colour = "#222222", linewidth = 0.48) +
  geom_segment(aes(x = min(mae_all_targets), xend = mae_all_targets,
                   yend = tasting_category_fine), colour = "grey78", linewidth = 0.45) +
  geom_point(aes(size = n_beers), alpha = 0.93) +
  geom_text(aes(label = paste0("n=", n_beers)), colour = "black",
            size = 2.35, hjust = 0, nudge_x = 0.008) +
  annotate(
    "text", x = mean_oof_mae + 0.003, y = 21.35,
    label = sprintf("总体 MAE=%.3f", mean_oof_mae),
    hjust = 0, vjust = -0.20,
    colour = "#222222", size = 2.45, fontface = "bold"
  ) +
  scale_colour_gradient(
    low = "#2C7FB8", high = "#C43C4E",
    name = "颜色：MAE（蓝低，红高）",
    guide = guide_colourbar(
      direction = "horizontal", title.position = "top",
      barwidth = unit(3.6, "cm"), barheight = unit(0.18, "cm")
    )
  ) +
  scale_size_continuous(range = c(2.0, 5.0), guide = "none") +
  scale_y_discrete(labels = style_labels_zh) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.21))) +
  labs(x = "全部目标的 MAE（z 分数）", y = NULL,
       subtitle = "（a）不同啤酒风格的 OOF 误差") +
  theme_mcup(base_size = 8.8, legend = "none") +
  panel_header_theme +
  theme(axis.text.y = element_text(size = 7.2),
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.title = element_text(face = "bold", size = 7.3),
        legend.text = element_text(size = 6.8))

mapping <- read_csv(file.path(data_dir, "scale_mapping_example.csv"),
                    show_col_types = FALSE) %>%
  mutate(
    target_display = recode(
      target_code,
      "A_malt_all" = "总体麦芽香气",
      "F_esters_flower" = "花香型酯类风味",
      "acidity" = "酸度",
      "alcohol" = "酒精感"
    )
  )
mapping_colors <- c(
  "总体麦芽香气" = "#2C7FB8",
  "花香型酯类风味" = "#D1495B",
  "酒精感" = "#55A868",
  "酸度" = "#8172B3"
)
mapping_linetypes <- c(
  "总体麦芽香气" = "solid",
  "花香型酯类风味" = "longdash",
  "酒精感" = "dotdash",
  "酸度" = "twodash"
)
p4b <- ggplot(mapping,
              aes(z_score, grade_0_6,
                  colour = target_display, linetype = target_display)) +
  geom_step(linewidth = 0.82, direction = "hv") +
  scale_colour_manual(values = mapping_colors, name = NULL) +
  scale_linetype_manual(values = mapping_linetypes, name = NULL) +
  scale_y_continuous(breaks = 0:6, limits = c(0, 6)) +
  labs(x = "模型预测的聚合 z 分数", y = "0—6 相对强度等级",
       subtitle = "（b）训练折经验分布的单调映射") +
  guides(
    colour = guide_legend(nrow = 2, byrow = TRUE),
    linetype = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  theme_mcup(base_size = 8.8, legend = "right") +
  panel_header_theme +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.text = element_text(size = 6.8),
    legend.key.width = unit(1.15, "cm"),
    legend.spacing.x = unit(0.12, "cm")
  )

p4 <- (p4a | p4b) +
  plot_layout(widths = c(1.12, 0.88))
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
  "图2 50 个感官目标的 ExtraTrees OOF R²与 Spearman 性能图谱",
  "三栏按感官家族展示全部目标；同一目标的 R²与 Spearman 共同刻画可预测性异质性。",
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

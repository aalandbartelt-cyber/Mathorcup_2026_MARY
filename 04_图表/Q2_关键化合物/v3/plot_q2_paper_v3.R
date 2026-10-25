suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(scales)
})

source("scripts/plot_style.R")

figure_dir <- "04_图表/Q2_关键化合物/v3"
source_dir <- file.path(figure_dir, "source_data")
dir.create(file.path(figure_dir, "qa"), recursive = TRUE, showWarnings = FALSE)

save_q2 <- function(plot, filename) {
  save_paper(plot, file.path(figure_dir, filename), width = 9, height = 6, dpi = 300)
}

panel_annotation <- plot_annotation(
  tag_levels = "a", tag_prefix = "(", tag_suffix = ")",
  theme = theme(plot.tag = element_text(family = FONT_PAPER, face = "bold", size = 10.5))
)

method_order <- c(
  "Spearman Top-3", "PLS/VIP", "ElasticNet", "有符号图ElasticNet",
  "ExtraTrees条件归因", "感官组收缩NNLS"
)
method_colors <- c(
  "Spearman Top-3" = "#9E9E9E", "PLS/VIP" = "#6F8FAF",
  "ElasticNet" = "#7BA6A1", "有符号图ElasticNet" = "#B0A27A",
  "ExtraTrees条件归因" = "#D17A22", "感官组收缩NNLS" = "#17365D"
)

# 图1：阈值选择与网络结构。
graph <- read_csv(file.path(source_dir, "图1_阈值与网络结构.csv"), show_col_types = FALSE)
edge_long <- graph |>
  select(相关阈值, 正相关边, 负相关边, 选定阈值) |>
  pivot_longer(c(正相关边, 负相关边), names_to = "边类型", values_to = "边数")
structure_long <- graph |>
  select(相关阈值, 孤立节点, 连通分量, 最大连通分量节点数, 选定阈值) |>
  pivot_longer(c(孤立节点, 连通分量, 最大连通分量节点数), names_to = "结构量", values_to = "数量")
selected_graph <- graph |> filter(选定阈值)

p1a <- ggplot(edge_long, aes(相关阈值, 边数, colour = 边类型, linetype = 边类型)) +
  geom_vline(xintercept = selected_graph$相关阈值, linewidth = 0.55, linetype = "22", colour = "#3F3F3F") +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  geom_point(data = filter(edge_long, 选定阈值), shape = 21, fill = "white", stroke = 1.0, size = 3.5) +
  annotate(
    "label", x = selected_graph$相关阈值 + 0.018, y = 2100,
    label = paste0("选定 |ρ| = 0.60\n总边数 ", selected_graph$总边数),
    hjust = 0, size = 3.05, family = FONT_PAPER, linewidth = 0.18, fill = "white"
  ) +
  scale_colour_manual(values = c("正相关边" = "#1F4E79", "负相关边" = "#C55A11")) +
  scale_linetype_manual(values = c("正相关边" = "solid", "负相关边" = "33")) +
  scale_x_continuous(breaks = graph$相关阈值) +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 1), breaks = c(0, 1, 10, 100, 1000, 10000),
    labels = label_number(big.mark = ",")
  ) +
  labs(x = "相关阈值 |ρ|", y = "边数（伪对数尺度）") +
  theme_mcup(legend = "right")

p1b <- ggplot(structure_long, aes(相关阈值, 数量, colour = 结构量, linetype = 结构量)) +
  geom_vline(xintercept = selected_graph$相关阈值, linewidth = 0.55, linetype = "22", colour = "#3F3F3F") +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  geom_point(data = filter(structure_long, 选定阈值), shape = 21, fill = "white", stroke = 1.0, size = 3.5) +
  annotate(
    "label", x = 0.405, y = 222,
    label = paste0(
      "|ρ| = 0.60：孤立节点 ", selected_graph$孤立节点,
      "\n连通分量 ", selected_graph$连通分量,
      "；最大分量 ", selected_graph$最大连通分量节点数
    ),
    hjust = 0, size = 2.9, family = FONT_PAPER, linewidth = 0.18, fill = "white"
  ) +
  scale_colour_manual(values = c(
    "孤立节点" = "#C55A11", "连通分量" = "#548235", "最大连通分量节点数" = "#1F4E79"
  )) +
  scale_linetype_manual(values = c("孤立节点" = "solid", "连通分量" = "33", "最大连通分量节点数" = "12")) +
  scale_x_continuous(breaks = graph$相关阈值) +
  scale_y_continuous(limits = c(0, 240), breaks = seq(0, 240, 40), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "相关阈值 |ρ|", y = "节点或分量数") +
  theme_mcup(legend = "right")

fig1 <- (p1a | p1b) + panel_annotation
save_q2(fig1, "图1_共表达图阈值敏感性_v3.png")

# 图2：折外性能与四路一致性。
model <- read_csv(file.path(source_dir, "图2a_模型性能.csv"), show_col_types = FALSE) |>
  mutate(方法 = factor(方法, levels = method_order))
agreement <- read_csv(file.path(source_dir, "图2b_四路一致性.csv"), show_col_types = FALSE) |>
  mutate(证据路径 = factor(证据路径, levels = c("总关联", "分层自助", "风格校正", "逐风格删除")))

p2a <- ggplot(model, aes(R2均值, 方法, colour = 方法)) +
  geom_errorbar(aes(xmin = R2均值 - R2标准差, xmax = R2均值 + R2标准差), orientation = "y", width = 0.18, linewidth = 0.65) +
  geom_point(size = 3.0) +
  geom_text(aes(label = sprintf("%.3f", R2均值)), hjust = -0.45, size = 2.85, family = FONT_PAPER, show.legend = FALSE) +
  scale_colour_manual(values = method_colors, guide = "none") +
  scale_x_continuous(limits = c(0.09, 0.285), breaks = seq(0.10, 0.25, 0.05), expand = expansion(mult = c(0.01, 0.01))) +
  labs(x = "宏平均 R²（均值 ± SD）", y = NULL) +
  theme_mcup(legend = "none")

p2b <- ggplot(model, aes(MAE均值, 方法, colour = 方法)) +
  geom_errorbar(aes(xmin = MAE均值 - MAE标准差, xmax = MAE均值 + MAE标准差), orientation = "y", width = 0.18, linewidth = 0.65) +
  geom_point(size = 3.0) +
  geom_text(aes(label = sprintf("%.3f", MAE均值)), hjust = -0.35, size = 2.85, family = FONT_PAPER, show.legend = FALSE) +
  scale_colour_manual(values = method_colors, guide = "none") +
  scale_x_continuous(limits = c(0.33, 0.395), breaks = seq(0.34, 0.39, 0.01), expand = expansion(mult = c(0.01, 0.01))) +
  labs(x = "宏平均 MAE（均值 ± SD，越低越好）", y = NULL) +
  theme_mcup(legend = "none") +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

p2c <- ggplot(agreement, aes(平均Jaccard, 证据路径, fill = 证据路径)) +
  geom_col(width = 0.60, colour = "black", linewidth = 0.25) +
  geom_text(
    aes(label = sprintf("Jaccard %.3f；完全一致 %d/50", 平均Jaccard, 完全一致目标数)),
    hjust = -0.04, size = 3.0, family = FONT_PAPER, fontface = "bold"
  ) +
  scale_fill_manual(values = c("总关联" = "#5B9BD5", "分层自助" = "#1F4E79", "风格校正" = "#ED7D31", "逐风格删除" = "#70AD47"), guide = "none") +
  scale_x_continuous(limits = c(0, 0.95), breaks = seq(0, 0.9, 0.1), expand = c(0, 0)) +
  labs(x = "最终 Top-3 与单一路径的平均 Jaccard", y = NULL) +
  theme_mcup(legend = "none")

fig2 <- ((p2a | p2b) / p2c) + plot_layout(heights = c(1.18, 0.82)) + panel_annotation
save_q2(fig2, "图2_模型性能与四路一致性_v3.png")

# 图3：候选稳定性与证据等级。
stability <- read_csv(file.path(source_dir, "图3_候选稳定性.csv"), show_col_types = FALSE) |>
  mutate(
    稳健等级 = factor(稳健等级, levels = c("R1 强", "R2 中", "R3 探索")),
    化学证据等级 = factor(化学证据等级, levels = c("A", "B", "C", "D"))
  )
grade_summary <- stability |>
  select(化学证据等级, 自助入选频率, 留一风格保留率) |>
  pivot_longer(c(自助入选频率, 留一风格保留率), names_to = "稳定性指标", values_to = "数值") |>
  group_by(化学证据等级, 稳定性指标) |>
  summarise(下四分位 = quantile(数值, 0.25), 中位数 = median(数值), 上四分位 = quantile(数值, 0.75), .groups = "drop")

p3a <- ggplot(stability, aes(自助入选频率, 留一风格保留率, colour = 稳健等级)) +
  annotate("rect", xmin = 0.60, xmax = Inf, ymin = 0.60, ymax = Inf, fill = "#DCE6F1", alpha = 0.28) +
  geom_vline(xintercept = 0.60, colour = "#666666", linetype = "22", linewidth = 0.5) +
  geom_hline(yintercept = 0.60, colour = "#666666", linetype = "22", linewidth = 0.5) +
  geom_point(size = 2.35, alpha = 0.84) +
  annotate("text", x = 0.98, y = 0.63, label = "双高稳定区", hjust = 1, vjust = 0, size = 3.0, family = FONT_PAPER, fontface = "bold") +
  scale_colour_manual(values = c("R1 强" = "#1F4E79", "R2 中" = "#C55A11", "R3 探索" = "#8C8C8C")) +
  scale_x_continuous(limits = c(0, 1.01), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1.01), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
  labs(x = "分层自助 Top-3 入选频率", y = "留一风格 Top-3 保留率") +
  theme_mcup(legend = "right")

p3b <- ggplot(grade_summary, aes(中位数, 化学证据等级, colour = 稳定性指标, shape = 稳定性指标)) +
  geom_errorbar(aes(xmin = 下四分位, xmax = 上四分位), orientation = "y", width = 0.12, linewidth = 0.75, position = position_dodge(width = 0.34)) +
  geom_point(size = 2.8, position = position_dodge(width = 0.34)) +
  scale_colour_manual(values = c("自助入选频率" = "#1F4E79", "留一风格保留率" = "#C55A11")) +
  scale_shape_manual(values = c("自助入选频率" = 16, "留一风格保留率" = 17)) +
  scale_x_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
  labs(x = "稳定性中位数与四分位区间", y = "化学证据等级") +
  theme_mcup(legend = "right")

fig3 <- (p3a | p3b) + plot_layout(widths = c(1.22, 0.78)) + panel_annotation
save_q2(fig3, "图3_候选稳定性与证据等级_v3.png")

# 图4：A 级关系四路名次与 OAV。
grade_a <- read_csv(file.path(source_dir, "图4_A级关系名次与OAV.csv"), show_col_types = FALSE) |>
  mutate(关系 = factor(关系, levels = rev(关系)))
rank_long <- grade_a |>
  select(关系, 总关联, 分层自助, 风格校正, 逐风格删除) |>
  pivot_longer(-关系, names_to = "证据路径", values_to = "名次") |>
  mutate(证据路径 = factor(证据路径, levels = c("总关联", "分层自助", "风格校正", "逐风格删除")))
oav <- grade_a |>
  mutate(
    横坐标 = case_when(OAV状态 == "N/A" ~ -4.0, OAV状态 == "零值" ~ -3.5, TRUE ~ log10(OAV中位数)),
    区间下界 = if_else(OAV状态 == "可计算", log10(OAV下四分位), NA_real_),
    区间上界 = if_else(OAV状态 == "可计算", log10(OAV上四分位), NA_real_),
    标签 = case_when(
      OAV状态 == "N/A" ~ "N/A",
      OAV状态 == "零值" ~ sprintf("0 [0, %.2f]", OAV上四分位),
      TRUE ~ sprintf("%.2f", OAV中位数)
    )
  )

p4a <- ggplot(rank_long, aes(证据路径, 关系)) +
  geom_path(aes(group = 关系), colour = "#BFBFBF", linewidth = 0.55) +
  geom_point(aes(fill = 名次), shape = 21, size = 4.5, colour = "black", stroke = 0.35) +
  geom_text(aes(label = 名次), size = 2.75, family = FONT_PAPER, fontface = "bold") +
  scale_fill_gradient(low = "white", high = "#5B9BD5", limits = c(1, 5), guide = "none") +
  labs(x = NULL, y = NULL) +
  theme_mcup(legend = "none") +
  theme(axis.text.x = element_text(face = "bold", margin = margin(t = 5)), axis.text.y = element_text(size = 8.4))

p4b <- ggplot(oav, aes(横坐标, 关系)) +
  annotate("rect", xmin = -4.20, xmax = -3.75, ymin = -Inf, ymax = Inf, fill = "#EFEFEF") +
  annotate("rect", xmin = -3.75, xmax = -3.25, ymin = -Inf, ymax = Inf, fill = "#F8F3E8") +
  geom_vline(xintercept = c(-3.75, -3.25), colour = "#888888", linewidth = 0.35) +
  geom_vline(xintercept = 0, colour = "#666666", linetype = "22", linewidth = 0.55) +
  geom_errorbar(
    data = filter(oav, OAV状态 == "可计算"),
    aes(xmin = 区间下界, xmax = 区间上界), orientation = "y", width = 0.14, linewidth = 0.65, colour = "#1F4E79"
  ) +
  geom_point(aes(shape = OAV状态, fill = OAV状态), size = 3.1, colour = "black", stroke = 0.4) +
  geom_text(aes(label = 标签), nudge_y = 0.24, size = 2.55, family = FONT_PAPER, show.legend = FALSE) +
  scale_shape_manual(values = c("可计算" = 21, "零值" = 22, "N/A" = 4), guide = "none") +
  scale_fill_manual(values = c("可计算" = "#5B9BD5", "零值" = "#F4B183", "N/A" = "white"), guide = "none") +
  scale_x_continuous(
    limits = c(-4.2, 1.05), breaks = c(-4, -3.5, -3, -2, -1, 0, 1),
    labels = c("N/A", "0", "0.001", "0.01", "0.1", "1", "10"), expand = c(0, 0)
  ) +
  labs(x = "中位 OAV（无量纲；正值区为对数尺度）", y = NULL) +
  theme_mcup(legend = "none") +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

fig4 <- (p4a | p4b) + plot_layout(widths = c(1.20, 0.80)) + panel_annotation
save_q2(fig4, "图4_A级关系四路排名与OAV_v3.png")

figure_list <- tibble::tribble(
  ~图号, ~路径, ~图题, ~说明,
  "图1", "04_图表/Q2_关键化合物/v3/图1_共表达图阈值敏感性_v3.png", "化合物共表达网络的阈值选择与结构敏感性", "突出 |ρ|=0.60 下正负边保留、孤立节点和连通结构。",
  "图2", "04_图表/Q2_关键化合物/v3/图2_模型性能与四路一致性_v3.png", "候选模型的折外性能与稳健排序的四路一致性", "分开呈现 R²、MAE 与四路 Jaccard，减少同一面板叠加。",
  "图3", "04_图表/Q2_关键化合物/v3/图3_候选稳定性与证据等级_v3.png", "最终候选的重采样稳定性、跨风格保留率与证据等级", "主图给出 150 条候选，辅图给出等级内中位数和四分位区间。",
  "图4", "04_图表/Q2_关键化合物/v3/图4_A级关系四路排名与OAV_v3.png", "A级化合物—感官关系的四路名次与气味活性值", "对单位不可比关系标 N/A，对零值单独标记，OAV 正值区使用对数尺度。"
)
write_csv(figure_list, file.path(figure_dir, "Q2_图清单_v3.csv"))

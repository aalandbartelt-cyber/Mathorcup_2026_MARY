suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(scales)
})

source("scripts/plot_style.R")

result_dir <- "03_结果/Q2_关键化合物"
figure_dir <- "04_图表/Q2_关键化合物"
data_dir <- file.path(result_dir, "figure_data")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

panel_theme <- theme(
  plot.title = element_text(face = "bold", size = 10.5, hjust = 0, margin = margin(b = 4)),
  plot.title.position = "plot"
)

save_q2 <- function(plot, filename, width = 8.8, height = 5.0) {
  save_paper(plot, file.path(figure_dir, filename), width = width, height = height, dpi = 300)
}

# Figure 1: graph threshold sensitivity.
graph <- read_csv(file.path(result_dir, "q2_graph_threshold_audit.csv"), show_col_types = FALSE)
edge_data <- graph |>
  select(abs_rho_threshold, positive_edges, negative_edges) |>
  pivot_longer(-abs_rho_threshold, names_to = "edge_type", values_to = "edges") |>
  mutate(edge_type = recode(edge_type,
    positive_edges = "正相关边", negative_edges = "负相关边"
  ))
structure_data <- graph |>
  select(abs_rho_threshold, isolated_nodes, n_components, largest_component) |>
  pivot_longer(-abs_rho_threshold, names_to = "metric", values_to = "nodes") |>
  mutate(metric = recode(metric,
    isolated_nodes = "孤立节点", n_components = "连通分量", largest_component = "最大分量节点"
  ))
write_csv(edge_data, file.path(data_dir, "图1a_边数量.csv"))
write_csv(structure_data, file.path(data_dir, "图1b_图结构.csv"))

p1a <- ggplot(edge_data, aes(abs_rho_threshold, edges, colour = edge_type)) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.2) +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 1),
    breaks = c(0, 1, 10, 100, 1000, 10000),
    labels = label_number(big.mark = ",")
  ) +
  scale_colour_manual(values = c("正相关边" = "#1F4E79", "负相关边" = "#C55A11")) +
  scale_x_continuous(breaks = graph$abs_rho_threshold) +
  labs(title = "(a) 边数量随阈值变化", x = expression("相关阈值  " * abs(rho)), y = "边数（对数尺度）") +
  theme_mcup(legend = "tr") + panel_theme

p1b <- ggplot(structure_data, aes(abs_rho_threshold, nodes, colour = metric, linetype = metric)) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 2.2) +
  scale_colour_manual(values = c("孤立节点" = "#C55A11", "连通分量" = "#70AD47", "最大分量节点" = "#1F4E79")) +
  scale_linetype_manual(values = c("孤立节点" = "solid", "连通分量" = "22", "最大分量节点" = "solid")) +
  scale_x_continuous(breaks = graph$abs_rho_threshold) +
  scale_y_continuous(limits = c(0, 240), breaks = seq(0, 240, 60), expand = expansion(mult = c(0, .04))) +
  annotate("label", x = 0.69, y = 224, label = "稳定模块 170 个\n单字段模块 155 个", size = 3.0,
           family = FONT_PAPER, linewidth = 0.2, fill = alpha("white", .9)) +
  labs(title = "(b) 连通结构对阈值敏感", x = expression("相关阈值  " * abs(rho)), y = "节点或分量数") +
  theme_mcup(legend = "bl") + panel_theme

fig1 <- p1a + p1b + plot_layout(widths = c(1, 1.08))
save_q2(fig1, "图1_共表达图阈值敏感性.png", width = 8.8, height = 4.6)

# Figure 2: predictive performance and lens agreement.
model <- bind_rows(
  read_csv(file.path(result_dir, "q2_model_summary_E101_E104.csv"), show_col_types = FALSE),
  read_csv(file.path(result_dir, "q2_model_summary_E105.csv"), show_col_types = FALSE)
) |>
  filter(method != "E100_GlobalMean") |>
  mutate(
    model = recode(method,
      E101_SpearmanTop3 = "Spearman Top-3",
      E102_PLS_VIP = "PLS/VIP",
      E103_ElasticNet = "ElasticNet",
      E104_GraphElasticNet = "图约束 ElasticNet",
      E105_ExtraTreesConditionalPermutation = "ExtraTrees"
    ),
    model = factor(model, levels = c("Spearman Top-3", "PLS/VIP", "ElasticNet", "图约束 ElasticNet", "ExtraTrees")),
    chosen = model == "ExtraTrees"
  )
agreement <- read_csv(file.path(result_dir, "q2_final_lens_agreement_E111.csv"), show_col_types = FALSE) |>
  group_by(lens) |>
  summarise(mean_jaccard = mean(jaccard), exact_targets = sum(exact_top3), .groups = "drop") |>
  mutate(
    lens_name = recode(lens,
      E106_total_association = "总关联",
      E107_bootstrap_stability = "Bootstrap",
      E108_style_adjusted = "风格内",
      E108_leave_one_style_survival = "逐风格删除"
    ),
    lens_name = factor(lens_name, levels = c("总关联", "Bootstrap", "风格内", "逐风格删除"))
  )
write_csv(model, file.path(data_dir, "图2ab_模型性能.csv"))
write_csv(agreement, file.path(data_dir, "图2c_四路一致性.csv"))

p2a <- ggplot(model, aes(model, macro_r2_mean, colour = chosen)) +
  geom_errorbar(aes(ymin = macro_r2_mean - macro_r2_sd, ymax = macro_r2_mean + macro_r2_sd), width = .18, linewidth = .55) +
  geom_point(size = 2.8) +
  scale_colour_manual(values = c(`FALSE` = "#7F7F7F", `TRUE` = "#1F4E79"), guide = "none") +
  scale_y_continuous(limits = c(.08, .25), breaks = seq(.10, .25, .05), expand = expansion(mult = c(.02, .08))) +
  labs(title = expression(bold("(a) OOF " * R^2)), x = NULL, y = expression("宏平均 " * R^2)) +
  theme_mcup(legend = "none") + panel_theme +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

p2b <- ggplot(model, aes(model, macro_mae_mean, colour = chosen)) +
  geom_errorbar(aes(ymin = macro_mae_mean - macro_mae_sd, ymax = macro_mae_mean + macro_mae_sd), width = .18, linewidth = .55) +
  geom_point(size = 2.8) +
  scale_colour_manual(values = c(`FALSE` = "#7F7F7F", `TRUE` = "#1F4E79"), guide = "none") +
  scale_y_continuous(limits = c(.335, .39), breaks = seq(.34, .39, .01), expand = expansion(mult = c(.02, .08))) +
  labs(title = "(b) OOF MAE", x = NULL, y = "宏平均 MAE（越低越好）") +
  theme_mcup(legend = "none") + panel_theme +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

p2c <- ggplot(agreement, aes(lens_name, mean_jaccard, fill = lens_name)) +
  geom_col(width = .64) +
  geom_text(aes(label = sprintf("%.3f\n%d/50", mean_jaccard, exact_targets)), vjust = -0.25, size = 3.0, fontface = "bold") +
  scale_fill_manual(values = c("总关联" = "#5B9BD5", "Bootstrap" = "#1F4E79", "风格内" = "#ED7D31", "逐风格删除" = "#70AD47"), guide = "none") +
  scale_y_continuous(limits = c(0, .78), breaks = seq(0, .7, .1), expand = c(0, 0)) +
  labs(title = "(c) 最终 Top-3 与四路证据一致性", x = NULL, y = "平均 Jaccard（标签下行为完全一致目标数）") +
  theme_mcup(legend = "none") + panel_theme +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

fig2 <- p2a + p2b + p2c + plot_layout(widths = c(1, 1, 1.15))
save_q2(fig2, "图2_模型性能与四路一致性.png", width = 10.0, height = 5.6)

# Figure 3: final Top-3 stability and evidence distribution.
top3 <- read_csv(file.path(result_dir, "q2_final_top3_E112.csv"), show_col_types = FALSE) |>
  mutate(
    statistical_robustness_tier = factor(statistical_robustness_tier,
      levels = c("R1_strong", "R2_moderate", "R3_exploratory"),
      labels = c("R1 强", "R2 中", "R3 探索")
    ),
    predictability_tier = factor(predictability_tier,
      levels = c("P1_supported", "P2_limited", "P3_weak"),
      labels = c("P1 支持", "P2 有限", "P3 较弱")
    ),
    chemical_evidence_grade = factor(chemical_evidence_grade, levels = c("A", "B", "C", "D"))
  )
grade_count <- top3 |> count(predictability_tier, chemical_evidence_grade, name = "n")
write_csv(top3 |> select(target_code, feature_code, bootstrap_selection_frequency, leave_style_top3_frequency,
                         statistical_robustness_tier, predictability_tier, chemical_evidence_grade, style_evidence),
          file.path(data_dir, "图3a_候选稳定性.csv"))
write_csv(grade_count, file.path(data_dir, "图3b_证据等级.csv"))

p3a <- ggplot(top3, aes(bootstrap_selection_frequency, leave_style_top3_frequency,
                        colour = statistical_robustness_tier)) +
  geom_vline(xintercept = .6, colour = "#888888", linetype = "22", linewidth = .45) +
  geom_hline(yintercept = .6, colour = "#888888", linetype = "22", linewidth = .45) +
  geom_point(aes(shape = style_evidence), size = 2.2, alpha = .82) +
  scale_colour_manual(values = c("R1 强" = "#1F4E79", "R2 中" = "#ED7D31", "R3 探索" = "#A5A5A5")) +
  scale_shape_manual(values = c(within_style_top3 = 16, within_style_top10 = 17, style_sensitive = 1),
                     labels = c(within_style_top3 = "风格内 Top-3", within_style_top10 = "风格内 Top-10", style_sensitive = "风格敏感")) +
  coord_cartesian(xlim = c(0, 1.03), ylim = c(0, 1.03), clip = "off") +
  scale_x_continuous(breaks = seq(0, 1, .2)) +
  scale_y_continuous(breaks = seq(0, 1, .2)) +
  labs(title = "(a) Bootstrap 与逐风格删除稳定性", x = "Bootstrap Top-3 入选频率", y = "留一风格 Top-3 保留率") +
  theme_mcup(legend = "right") + panel_theme +
  theme(legend.box = "vertical")

p3b <- ggplot(grade_count, aes(predictability_tier, n, fill = chemical_evidence_grade)) +
  geom_col(width = .62, colour = "white", linewidth = .3) +
  geom_text(aes(label = if_else(n > 0, as.character(n), "")), position = position_stack(vjust = .5),
            size = 3.0, fontface = "bold", colour = "white") +
  scale_fill_manual(values = c(A = "#1B7837", B = "#4D87C7", C = "#E69F00", D = "#7F7F7F"), name = "化学证据") +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  labs(title = "(b) 目标可预测性与化学证据", x = NULL, y = "候选数") +
  theme_mcup(legend = "tr") + panel_theme

fig3 <- p3a + p3b + plot_layout(widths = c(1.45, .75))
save_q2(fig3, "图3_候选稳定性与证据等级.png", width = 9.2, height = 5.2)

# Figure 4: eight grade-A relations across four lenses and OAV.
grade_a <- top3 |>
  filter(chemical_evidence_grade == "A") |>
  mutate(
    relation = paste0(target_code, " — ", chemical_name_short),
    relation = factor(relation, levels = rev(unique(relation)))
  )
rank_data <- grade_a |>
  select(relation, rank__E106_total_association, rank__E107_bootstrap_stability,
         rank__E108_style_adjusted, rank__E108_leave_one_style_survival) |>
  pivot_longer(-relation, names_to = "lens", values_to = "rank") |>
  mutate(
    lens = recode(lens,
      rank__E106_total_association = "总关联",
      rank__E107_bootstrap_stability = "Bootstrap",
      rank__E108_style_adjusted = "风格内",
      rank__E108_leave_one_style_survival = "逐风格删除"
    ),
    lens = factor(lens, levels = c("总关联", "Bootstrap", "风格内", "逐风格删除")),
    rank_for_fill = pmin(rank, 10)
  )
oav_data <- grade_a |>
  mutate(
    oav_plot = case_when(
      is.na(oav_median) ~ .001,
      oav_median <= 0 ~ .001,
      TRUE ~ oav_median
    ),
    oav_label = case_when(is.na(oav_median) ~ "NA", oav_median <= 0 ~ "0", TRUE ~ sprintf("%.2f", oav_median)),
    available = if_else(is.na(oav_median), "无同单位阈值", "可计算 OAV")
  )
write_csv(rank_data, file.path(data_dir, "图4a_A级关系四路排名.csv"))
write_csv(oav_data |> select(relation, oav_q25, oav_median, oav_q75, oav_above_1_fraction, oav_label, available),
          file.path(data_dir, "图4b_A级关系OAV.csv"))

p4a <- ggplot(rank_data, aes(lens, relation, fill = rank_for_fill)) +
  geom_tile(colour = "white", linewidth = .65) +
  geom_text(aes(label = rank), size = 3.0, fontface = "bold") +
  scale_fill_gradient(low = "#2F75B5", high = "#F2F2F2", limits = c(1, 10),
                      breaks = c(1, 3, 5, 10), name = "名次") +
  labs(title = "(a) A 级关系的四路排名", x = NULL, y = NULL) +
  theme_mcup(legend = "right") + panel_theme +
  theme(axis.text.x = element_text(face = "bold"), axis.text.y = element_text(size = 8.2))

p4b <- ggplot(oav_data, aes(oav_plot, relation, colour = available)) +
  geom_vline(xintercept = 1, linetype = "22", colour = "#666666", linewidth = .5) +
  geom_errorbar(aes(xmin = if_else(is.na(oav_q25), oav_plot, pmax(oav_q25, .001)),
                    xmax = if_else(is.na(oav_q75), oav_plot, pmax(oav_q75, .001))),
                orientation = "y", width = .16, linewidth = .55) +
  geom_point(size = 2.4) +
  geom_text(aes(label = oav_label), nudge_y = .23, size = 2.7, show.legend = FALSE) +
  scale_x_log10(limits = c(.0007, 10), breaks = c(.001, .01, .1, 1, 10),
                labels = c("0.001", "0.01", "0.1", "1", "10")) +
  scale_colour_manual(values = c("可计算 OAV" = "#1F4E79", "无同单位阈值" = "#A5A5A5")) +
  labs(title = "(b) A 级关系的中位 OAV", x = "OAV（对数尺度，虚线为 1）", y = NULL) +
  theme_mcup(legend = "right") + panel_theme +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        legend.position = "bottom", legend.direction = "horizontal",
        legend.justification = "center")

fig4 <- p4a + p4b + plot_layout(widths = c(1.25, .85))
save_q2(fig4, "图4_A级关系四路排名与OAV.png", width = 9.6, height = 5.4)

figure_list <- tibble::tribble(
  ~figure, ~path, ~caption, ~note, ~width_in, ~height_in,
  "图1", "04_图表/Q2_关键化合物/图1_共表达图阈值敏感性.png", "化合物共表达图的阈值敏感性", "硬阈值升高会使边数快速衰减并产生大量孤立节点，支持折内软权图和稳定模块口径。", 8.8, 4.6,
  "图2", "04_图表/Q2_关键化合物/图2_模型性能与四路一致性.png", "候选模型的折外性能与最终排名的一致性", "ExtraTrees 的 OOF 误差最低；最终 Top-3 与 bootstrap 最一致，与风格内排名差异最大。", 10.0, 5.6,
  "图3", "04_图表/Q2_关键化合物/图3_候选稳定性与证据等级.png", "最终候选的重采样稳定性、风格敏感性与证据等级", "R1 候选集中在高稳定区域；P3 弱预测目标产生更多 C/D 级候选。", 9.2, 5.2,
  "图4", "04_图表/Q2_关键化合物/图4_A级关系四路排名与OAV.png", "A 级化合物—感官关系的四路排名与气味活性值", "8 条 A 级关系同时展示总关联、bootstrap、风格内和逐风格删除排名，并标出可核验的中位 OAV。", 9.6, 5.4
)
write_csv(figure_list, file.path(figure_dir, "Q2_图清单.csv"))

message("Q2 figures written to ", figure_dir)

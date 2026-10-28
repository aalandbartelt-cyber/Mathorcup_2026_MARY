# 论文出图统一样式（R / ggplot2）
#
# 视觉规范来源：40 篇 M 杯优秀论文 615 张图表实测 + 逐页目检（docs/M杯优秀论文特点、题型特点.md §4.5）
#   - 白底、黑色细边框、无网格、图例内嵌且无边框、3:2 横版
#   - 中文宋体 + 拉丁 Times 风格（SimSun 自带）
#   - 训练曲线四色；PR 总曲线深蓝加粗；混淆矩阵 Blues；热力图 hot
# 项目强制规则：图片文件内不放整图标题/图号，图题在论文中置于图下；
# 分面标签无灰色底框并加粗，轴标题加粗，主信息需有明确视觉强调。
#
# 用法：
#   source("scripts/plot_style.R")
#   p <- ggplot(...) + theme_mcup()
#   save_paper(p, "figures/fig1.png")
#   plot_confusion_matrix(cm, c("dent","hole","rusty"), file = "figures/cm.png")
#   plot_pr_curves(curves, file = "figures/pr.png")
#   plot_corr_heatmap(cor_mat, file = "figures/corr.png")
#   plot_feature_importance(names, values, file = "figures/imp.png")
suppressPackageStartupMessages({
  library(ggplot2)
  library(ragg)
})

FONT_PAPER <- "SimSun"
FONT_SANS  <- "SimHei"

CYCLE <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
NAVY  <- "#00007f"

# 逐条重要性条用的冷暖色（近似 matplotlib Spectral，自上而下红→蓝）
SPECTRAL <- c("#D53E4F", "#F46D43", "#FDAE61", "#FEE08B",
              "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD")
spectral_n <- function(n) colorRampPalette(SPECTRAL)(n)

# 紧凑、干净的 M 杯主题
# legend: "br"=右下、"tr"=右上、"none"=不要图例
theme_mcup <- function(mode = c("paper", "sans"), base_size = 10, legend = "br",
                       title_family = NULL) {
  mode <- match.arg(mode)
  fam <- if (mode == "paper") FONT_PAPER else FONT_SANS
  lg <- switch(legend,
    br = list(pos = c(0.985, 0.02), just = c(1, 0)),
    tr = list(pos = c(0.985, 0.985), just = c(1, 1)),
    bl = list(pos = c(0.015, 0.02), just = c(0, 0)),
    right = list(pos = "right"),
    none = NULL,
    stop("legend 必须是 br / tr / bl / right / none")
  )
  th <- theme_bw(base_size = base_size, base_family = fam) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = 0.5),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      axis.text = element_text(colour = "black", size = base_size - 1),
      axis.title = element_text(colour = "black", size = base_size - 0.5,
                                face = "bold"),
      axis.ticks = element_line(colour = "black", linewidth = 0.35),
      axis.ticks.length = unit(1.6, "pt"),
      plot.title = element_blank(),
      plot.tag = element_text(colour = "black", face = "bold",
                              size = base_size),
      strip.background = element_blank(),
      strip.text = element_text(colour = "black", face = "bold",
                                size = base_size - 0.5,
                                margin = margin(2, 1, 3, 1)),
      plot.margin = margin(4, 5, 3, 3),
      legend.background = element_blank(),
      legend.box.background = element_blank(),
      legend.key = element_blank(),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 1.5),
      legend.key.size = unit(0.82, "lines"),
      legend.key.width = unit(1.15, "lines"),
      legend.spacing.y = unit(0.02, "cm"),
      legend.margin = margin(1.5, 2.5, 1.5, 2.5)
    )
  if (is.null(lg)) {
    th + theme(legend.position = "none")
  } else if (!is.null(lg$pos) && length(lg$pos) == 1 && lg$pos == "right") {
    th + theme(legend.position = "right",
               legend.justification = "center",
               legend.margin = margin(0, 0, 0, 4),
               legend.key.height = unit(0.6, "lines"))
  } else {
    th + theme(legend.position = "inside",
               legend.position.inside = lg$pos,
               legend.justification = lg$just)
  }
}

# 保存 300dpi PNG
save_paper <- function(plot, file, width = 6, height = 4, dpi = 300) {
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  agg_png(file, width = width, height = height, units = "in", res = dpi, background = "white")
  on.exit(invisible(dev.off()), add = TRUE)
  print(plot)
  invisible(file)
}

# 混淆矩阵：Blues + 每格数值（深色格白字），短类名不旋转、无留白
plot_confusion_matrix <- function(cm, classes, file = NULL,
                                  normalize = FALSE, width = 4.2, height = 3.6) {
  stopifnot(is.matrix(cm), nrow(cm) == ncol(cm), length(classes) == nrow(cm))
  df <- expand.grid(pred = classes, true = classes)
  df$n <- as.vector(t(cm))
  if (normalize) {
    df$value <- df$n / ave(df$n, df$true, FUN = sum)
    lab <- sprintf("%.2f", df$value)
  } else {
    df$value <- df$n
    lab <- as.character(df$n)
  }
  thresh <- max(df$value) / 2
  df$txt <- ifelse(df$value > thresh, "white", "black")
  angle <- if (max(nchar(classes)) <= 7) 0 else 35
  p <- ggplot(df, aes(pred, true, fill = value)) +
    geom_tile(colour = "white", linewidth = 0.45) +
    geom_text(aes(label = lab, colour = txt), size = 3.1) +
    scale_colour_identity() +
    scale_fill_gradient(low = "#F7FBFF", high = "#08306B",
                        guide = guide_colourbar(barwidth = unit(0.32, "cm"),
                                                barheight = unit(2.3, "cm"),
                                                frame.colour = NA)) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0), limits = rev(classes)) +
    labs(x = "预测标签", y = "真实标签") +
    coord_fixed() +
    theme_mcup(legend = "right") +
    theme(axis.text.x = element_text(angle = angle, hjust = if (angle == 0) 0.5 else 1),
          axis.title.x = element_text(margin = margin(t = 6)),
          axis.title.y = element_text(margin = margin(r = 6)))
  if (!is.null(file)) save_paper(p, file, width = width, height = height)
  p
}

# PR 曲线：各类细线 + 总曲线深蓝加粗，图例带 AP，无边框
plot_pr_curves <- function(curves, file = NULL, width = 5, height = 4) {
  dfs <- lapply(names(curves), function(nm) {
    d <- curves[[nm]]
    data.frame(recall = d$recall, precision = d$precision,
               cls = nm, ap = d$ap, stringsAsFactors = FALSE)
  })
  df <- do.call(rbind, dfs)
  df$is_all <- tolower(df$cls) %in% c("all", "all classes", "micro", "总体")
  others <- setdiff(unique(df$cls[!df$is_all]), character(0))
  lev <- c(others, if (any(df$is_all)) "all classes")
  lab <- vapply(lev, function(k) {
    ap <- df$ap[df$cls == k][1]
    if (is.na(ap)) ap <- df$ap[df$is_all][1]
    if (k == "all classes") sprintf("all classes %.3f mAP@0.5", ap) else sprintf("%s %.3f", k, ap)
  }, character(1))
  df$grp <- factor(ifelse(df$is_all, "all classes", df$cls), levels = lev, labels = lab)
  col_map <- setNames(c(CYCLE[seq_along(others)], NAVY), lab)
  lw_map <- setNames(ifelse(lev == "all classes", 1.15, 0.55), lab)
  p <- ggplot(df, aes(recall, precision, colour = grp, linewidth = grp)) +
    geom_line() +
    scale_colour_manual(values = col_map) +
    scale_linewidth_manual(values = lw_map, guide = "none") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.01), expand = FALSE) +
    labs(x = "Recall", y = "Precision", title = "Precision-Recall Curve") +
    theme_mcup(legend = "bl")
  if (!is.null(file)) save_paper(p, file, width = width, height = height)
  p
}

# 相关性热力图：hot（负值深、正值浅）+ 全格数值
plot_corr_heatmap <- function(mat, file = NULL, width = 5.2, height = 4.6) {
  stopifnot(is.matrix(mat), nrow(mat) == ncol(mat))
  nm <- rownames(mat); if (is.null(nm)) nm <- paste0("V", seq_len(nrow(mat)))
  df <- expand.grid(var2 = nm, var1 = nm)
  df$value <- as.vector(t(mat))
  df$txt <- ifelse(df$value < -0.2, "white", "black")
  p <- ggplot(df, aes(var2, var1, fill = value)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", value), colour = txt), size = 2.4) +
    scale_colour_identity() +
    scale_fill_gradientn(colours = c("black", "red", "orange", "yellow", "white"),
                         limits = c(-1, 1),
                         guide = guide_colourbar(barwidth = unit(0.3, "cm"),
                                                 barheight = unit(2.6, "cm"),
                                                 frame.colour = NA)) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0), limits = rev(nm)) +
    labs(x = NULL, y = NULL, title = "相关性热力图") +
    coord_fixed() +
    theme_mcup(legend = "right") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  if (!is.null(file)) save_paper(p, file, width = width, height = height)
  p
}

# 特征重要性：横向降序、逐条冷暖色、末端数值
plot_feature_importance <- function(names, values, file = NULL,
                                    width = 6, height = NULL, top_n = 15) {
  ord <- order(values, decreasing = TRUE)[seq_len(min(top_n, length(values)))]
  df <- data.frame(name = factor(names[ord], levels = rev(names[ord])),
                   value = values[ord])
  df$col <- factor(seq_len(nrow(df)))
  h <- if (is.null(height)) max(3.0, 0.30 * nrow(df) + 0.9) else height
  p <- ggplot(df, aes(value, name, fill = col)) +
    geom_col(width = 0.68) +
    geom_text(aes(label = sprintf("%.1f", value)), hjust = -0.15, size = 2.5) +
    scale_fill_manual(values = spectral_n(nrow(df)), guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.10))) +
    labs(x = "Relative Importance", y = NULL, title = "特征重要性") +
    theme_mcup(legend = "none") +
    theme(axis.text.y = element_text(size = 8.5))
  if (!is.null(file)) save_paper(p, file, width = width, height = h)
  p
}


# Read the frozen flat source tables; all outputs go to a separate replay folder.
.mcup_csv_map <- character()
.mcup_csv_map["Q1|all_target_deltas.csv"] <- "03_结果/Q1_逐目标性能变化.csv"
.mcup_csv_map["COM|all_target_deltas.csv"] <- "03_结果/Q1_逐目标性能变化.csv"
.mcup_csv_map["Q1|final_oof_predictions.csv.gz"] <- "03_结果/Q1_折外预测明细.csv.gz"
.mcup_csv_map["COM|final_oof_predictions.csv.gz"] <- "03_结果/Q1_折外预测明细.csv.gz"
.mcup_csv_map["Q1|final_oof_with_relative_grades.csv.gz"] <- "03_结果/Q1_相对评分明细.csv.gz"
.mcup_csv_map["COM|final_oof_with_relative_grades.csv.gz"] <- "03_结果/Q1_相对评分明细.csv.gz"
.mcup_csv_map["Q1|fixed_tail_13.csv"] <- "03_结果/Q1_尾部13目标.csv"
.mcup_csv_map["COM|fixed_tail_13.csv"] <- "03_结果/Q1_尾部13目标.csv"
.mcup_csv_map["Q1|model_comparison.csv"] <- "03_结果/Q1_模型对比.csv"
.mcup_csv_map["COM|model_comparison.csv"] <- "03_结果/Q1_模型对比.csv"
.mcup_csv_map["Q1|model_metrics_by_repeat.csv"] <- "03_结果/Q1_逐重复指标.csv"
.mcup_csv_map["COM|model_metrics_by_repeat.csv"] <- "03_结果/Q1_逐重复指标.csv"
.mcup_csv_map["Q1|model_summary.csv"] <- "03_结果/Q1_模型指标汇总.csv"
.mcup_csv_map["COM|model_summary.csv"] <- "03_结果/Q1_模型指标汇总.csv"
.mcup_csv_map["Q1|paired_beer_bootstrap.csv"] <- "03_结果/Q1_配对重采样.csv"
.mcup_csv_map["COM|paired_beer_bootstrap.csv"] <- "03_结果/Q1_配对重采样.csv"
.mcup_csv_map["Q1|preregistered_hard_targets.csv"] <- "03_结果/Q1_难目标检验.csv"
.mcup_csv_map["COM|preregistered_hard_targets.csv"] <- "03_结果/Q1_难目标检验.csv"
.mcup_csv_map["Q1|representative_oof.csv.gz"] <- "03_结果/Q1_代表目标折外预测.csv.gz"
.mcup_csv_map["COM|representative_oof.csv.gz"] <- "03_结果/Q1_代表目标折外预测.csv.gz"
.mcup_csv_map["Q1|scale_mapping_example.csv"] <- "03_结果/Q1_量表映射.csv"
.mcup_csv_map["COM|scale_mapping_example.csv"] <- "03_结果/Q1_量表映射.csv"
.mcup_csv_map["Q1|stacking_weights_by_fold.csv"] <- "03_结果/Q1_逐折堆叠权重.csv"
.mcup_csv_map["COM|stacking_weights_by_fold.csv"] <- "03_结果/Q1_逐折堆叠权重.csv"
.mcup_csv_map["Q1|stacking_weights_summary.csv"] <- "03_结果/Q1_堆叠权重汇总.csv"
.mcup_csv_map["COM|stacking_weights_summary.csv"] <- "03_结果/Q1_堆叠权重汇总.csv"
.mcup_csv_map["Q1|style_metrics_final.csv"] <- "03_结果/Q1_逐风格指标.csv"
.mcup_csv_map["COM|style_metrics_final.csv"] <- "03_结果/Q1_逐风格指标.csv"
.mcup_csv_map["Q1|styles_22.csv"] <- "03_结果/Q1_22风格指标.csv"
.mcup_csv_map["COM|styles_22.csv"] <- "03_结果/Q1_22风格指标.csv"
.mcup_csv_map["Q1|target_metrics_by_repeat.csv"] <- "03_结果/Q1_逐目标逐重复指标.csv"
.mcup_csv_map["COM|target_metrics_by_repeat.csv"] <- "03_结果/Q1_逐目标逐重复指标.csv"
.mcup_csv_map["Q1|target_metrics_final.csv"] <- "03_结果/Q1_逐目标指标.csv"
.mcup_csv_map["COM|target_metrics_final.csv"] <- "03_结果/Q1_逐目标指标.csv"
.mcup_csv_map["Q1|weight_structure_ablation.csv"] <- "03_结果/Q1_权重结构消融.csv"
.mcup_csv_map["COM|weight_structure_ablation.csv"] <- "03_结果/Q1_权重结构消融.csv"
.mcup_csv_map["Q2|全部候选稳健排名_当前版.csv"] <- "03_结果/Q2_全部候选稳健排名.csv"
.mcup_csv_map["COM|全部候选稳健排名_当前版.csv"] <- "03_结果/Q2_全部候选稳健排名.csv"
.mcup_csv_map["Q2|关键化合物Top3_当前版.csv"] <- "03_结果/Q2_关键化合物Top3.csv"
.mcup_csv_map["COM|关键化合物Top3_当前版.csv"] <- "03_结果/Q2_关键化合物Top3.csv"
.mcup_csv_map["Q2|化学证据核验_当前版.csv"] <- "03_结果/Q2_化学证据核验.csv"
.mcup_csv_map["COM|化学证据核验_当前版.csv"] <- "03_结果/Q2_化学证据核验.csv"
.mcup_csv_map["Q2|模型性能对比_当前版.csv"] <- "03_结果/Q2_模型性能对比.csv"
.mcup_csv_map["COM|模型性能对比_当前版.csv"] <- "03_结果/Q2_模型性能对比.csv"
.mcup_csv_map["Q3|三风格_900条生成化学谱.csv.gz"] <- "03_结果/Q3_三风格_900条生成化学谱.csv.gz"
.mcup_csv_map["COM|三风格_900条生成化学谱.csv.gz"] <- "03_结果/Q3_三风格_900条生成化学谱.csv.gz"
.mcup_csv_map["Q3|三风格_完整感官画像.csv"] <- "03_结果/Q3_三风格_完整感官画像.csv"
.mcup_csv_map["COM|三风格_完整感官画像.csv"] <- "03_结果/Q3_三风格_完整感官画像.csv"
.mcup_csv_map["Q3|跨风格迁移_33组状态.csv"] <- "03_结果/Q3_跨风格迁移_33组状态.csv"
.mcup_csv_map["COM|跨风格迁移_33组状态.csv"] <- "03_结果/Q3_跨风格迁移_33组状态.csv"
.mcup_csv_map["总论文|Q1_22风格样本数_可复用.csv"] <- "03_结果/Q1_22风格样本数.csv"
.mcup_csv_map["COM|Q1_22风格样本数_可复用.csv"] <- "03_结果/Q1_22风格样本数.csv"
.mcup_csv_map["总论文|Q1_风格中文映射与样本数.csv"] <- "03_结果/Q1_风格中文映射与样本数.csv"
.mcup_csv_map["COM|Q1_风格中文映射与样本数.csv"] <- "03_结果/Q1_风格中文映射与样本数.csv"
.mcup_csv_map["总论文|Q2_218候选成对相关.csv"] <- "03_结果/Q2_218候选成对相关.csv"
.mcup_csv_map["COM|Q2_218候选成对相关.csv"] <- "03_结果/Q2_218候选成对相关.csv"
.mcup_csv_map["总论文|Q2_排除13字段.csv"] <- "03_结果/Q2_排除13字段.csv"
.mcup_csv_map["COM|Q2_排除13字段.csv"] <- "03_结果/Q2_排除13字段.csv"
.mcup_csv_map["总论文|Q2_排除类型数量.csv"] <- "03_结果/Q2_排除类型数量.csv"
.mcup_csv_map["COM|Q2_排除类型数量.csv"] <- "03_结果/Q2_排除类型数量.csv"
.mcup_csv_map["总论文|Q2_输入与候选规模.csv"] <- "03_结果/Q2_输入与候选规模.csv"
.mcup_csv_map["COM|Q2_输入与候选规模.csv"] <- "03_结果/Q2_输入与候选规模.csv"
.mcup_csv_map["总论文|Q3_三风格六维实测长表.csv"] <- "03_结果/Q3_三风格六维实测长表.csv"
.mcup_csv_map["COM|Q3_三风格六维实测长表.csv"] <- "03_结果/Q3_三风格六维实测长表.csv"
.mcup_csv_map["总论文|Q3_实测与画像完整性.csv"] <- "03_结果/Q3_实测与画像完整性.csv"
.mcup_csv_map["COM|Q3_实测与画像完整性.csv"] <- "03_结果/Q3_实测与画像完整性.csv"
.mcup_csv_map["总论文|Q3_实测中位数与四分位.csv"] <- "03_结果/Q3_实测中位数与四分位.csv"
.mcup_csv_map["COM|Q3_实测中位数与四分位.csv"] <- "03_结果/Q3_实测中位数与四分位.csv"
.mcup_csv_map["总论文|q1_missing_nine_fields_zh.csv"] <- "03_结果/Q1_9字段缺失分布.csv"
.mcup_csv_map["COM|q1_missing_nine_fields_zh.csv"] <- "03_结果/Q1_9字段缺失分布.csv"
.mcup_csv_map["总论文|q1_style_counts.csv"] <- "03_结果/Q1_风格样本数.csv"
.mcup_csv_map["COM|q1_style_counts.csv"] <- "03_结果/Q1_风格样本数.csv"
.mcup_csv_map["总论文|q1_target_families.csv"] <- "03_结果/Q1_感官家族数量.csv"
.mcup_csv_map["COM|q1_target_families.csv"] <- "03_结果/Q1_感官家族数量.csv"
.mcup_csv_map["总论文|节点.csv"] <- "03_结果/路线节点.csv"
.mcup_csv_map["COM|节点.csv"] <- "03_结果/路线节点.csv"
.mcup_csv_map["总论文|连线.csv"] <- "03_结果/路线连线.csv"
.mcup_csv_map["COM|连线.csv"] <- "03_结果/路线连线.csv"
.mcup_csv_map["总论文|附件使用矩阵.csv"] <- "03_结果/附件使用矩阵.csv"
.mcup_csv_map["COM|附件使用矩阵.csv"] <- "03_结果/附件使用矩阵.csv"
.mcup_csv_map["总论文|附件使用矩阵_长表.csv"] <- "03_结果/附件使用矩阵_长表.csv"
.mcup_csv_map["COM|附件使用矩阵_长表.csv"] <- "03_结果/附件使用矩阵_长表.csv"
.mcup_csv_map["总论文|附件口径汇总.csv"] <- "03_结果/附件口径汇总.csv"
.mcup_csv_map["COM|附件口径汇总.csv"] <- "03_结果/附件口径汇总.csv"
.mcup_csv_map["Q1|q1_calibration_bins.csv"] <- "03_结果/Q1_校准分箱.csv"
.mcup_csv_map["COM|q1_calibration_bins.csv"] <- "03_结果/Q1_校准分箱.csv"
.mcup_csv_map["Q1|q1_calibration_summary.csv"] <- "03_结果/Q1_校准汇总.csv"
.mcup_csv_map["COM|q1_calibration_summary.csv"] <- "03_结果/Q1_校准汇总.csv"
.mcup_csv_map["Q1|q1_interval_coverage.csv"] <- "03_结果/Q1_区间覆盖率.csv"
.mcup_csv_map["COM|q1_interval_coverage.csv"] <- "03_结果/Q1_区间覆盖率.csv"
.mcup_csv_map["Q1|q1_repeat_metrics_summary.csv"] <- "03_结果/Q1_重复指标汇总.csv"
.mcup_csv_map["COM|q1_repeat_metrics_summary.csv"] <- "03_结果/Q1_重复指标汇总.csv"
.mcup_csv_map["Q1|q1_repeat_model_ranks.csv"] <- "03_结果/Q1_重复模型名次.csv"
.mcup_csv_map["COM|q1_repeat_model_ranks.csv"] <- "03_结果/Q1_重复模型名次.csv"
.mcup_csv_map["Q1|q1_target_reliability.csv"] <- "03_结果/Q1_目标可靠性.csv"
.mcup_csv_map["COM|q1_target_reliability.csv"] <- "03_结果/Q1_目标可靠性.csv"
.mcup_csv_map["Q1|q1_ablation_folds.csv"] <- "03_结果/Q1_逐折消融.csv"
.mcup_csv_map["COM|q1_ablation_folds.csv"] <- "03_结果/Q1_逐折消融.csv"
.mcup_csv_map["Q1|q1_ablation_summary.csv"] <- "03_结果/Q1_消融汇总.csv"
.mcup_csv_map["COM|q1_ablation_summary.csv"] <- "03_结果/Q1_消融汇总.csv"
.mcup_csv_map["Q1|q1_leave_style_out.csv"] <- "03_结果/Q1_留一风格检验.csv"
.mcup_csv_map["COM|q1_leave_style_out.csv"] <- "03_结果/Q1_留一风格检验.csv"
.mcup_csv_map["Q1|ablation_summary.csv"] <- "03_结果/Q1_消融汇总_图源.csv"
.mcup_csv_map["COM|ablation_summary.csv"] <- "03_结果/Q1_消融汇总_图源.csv"
.mcup_csv_map["Q1|calibration_bins.csv"] <- "03_结果/Q1_校准分箱_图源.csv"
.mcup_csv_map["COM|calibration_bins.csv"] <- "03_结果/Q1_校准分箱_图源.csv"
.mcup_csv_map["Q1|interval_summary.csv"] <- "03_结果/Q1_区间汇总.csv"
.mcup_csv_map["COM|interval_summary.csv"] <- "03_结果/Q1_区间汇总.csv"
.mcup_csv_map["Q1|leave_style_out.csv"] <- "03_结果/Q1_留一风格检验_图源.csv"
.mcup_csv_map["COM|leave_style_out.csv"] <- "03_结果/Q1_留一风格检验_图源.csv"
.mcup_csv_map["Q1|model_repeats.csv"] <- "03_结果/Q1_模型重复指标.csv"
.mcup_csv_map["COM|model_repeats.csv"] <- "03_结果/Q1_模型重复指标.csv"
.mcup_csv_map["Q1|target_reliability.csv"] <- "03_结果/Q1_目标可靠性_图源.csv"
.mcup_csv_map["COM|target_reliability.csv"] <- "03_结果/Q1_目标可靠性_图源.csv"
.mcup_csv_map["Q1|style_metrics_extra_trees.csv"] <- "03_结果/Q1_基线风格指标.csv"
.mcup_csv_map["COM|style_metrics_extra_trees.csv"] <- "03_结果/Q1_基线风格指标.csv"
.mcup_csv_map["Q1|target_dictionary.csv"] <- "03_结果/Q1_感官目标字典.csv"
.mcup_csv_map["COM|target_dictionary.csv"] <- "03_结果/Q1_感官目标字典.csv"
.mcup_csv_map["Q1|target_metrics_extra_trees.csv"] <- "03_结果/Q1_基线目标指标.csv"
.mcup_csv_map["COM|target_metrics_extra_trees.csv"] <- "03_结果/Q1_基线目标指标.csv"
.mcup_csv_map["Q2|最终模型_删除敏感性.csv"] <- "03_结果/Q2_最终模型_删除敏感性.csv"
.mcup_csv_map["COM|最终模型_删除敏感性.csv"] <- "03_结果/Q2_最终模型_删除敏感性.csv"
.mcup_csv_map["Q2|最终模型_条件重要性.csv.gz"] <- "03_结果/Q2_最终模型_条件重要性.csv.gz"
.mcup_csv_map["COM|最终模型_条件重要性.csv.gz"] <- "03_结果/Q2_最终模型_条件重要性.csv.gz"
.mcup_csv_map["Q2|最终模型_模块重要性.csv.gz"] <- "03_结果/Q2_最终模型_模块重要性.csv.gz"
.mcup_csv_map["COM|最终模型_模块重要性.csv.gz"] <- "03_结果/Q2_最终模型_模块重要性.csv.gz"
.mcup_csv_map["Q2|q2_validation_anchor.csv"] <- "03_结果/Q2_直接锚点核验.csv"
.mcup_csv_map["COM|q2_validation_anchor.csv"] <- "03_结果/Q2_直接锚点核验.csv"
.mcup_csv_map["Q2|q2_validation_completeness.csv"] <- "03_结果/Q2_完整性核验.csv"
.mcup_csv_map["COM|q2_validation_completeness.csv"] <- "03_结果/Q2_完整性核验.csv"
.mcup_csv_map["Q2|q2_validation_deletion.csv"] <- "03_结果/Q2_删除敏感性.csv"
.mcup_csv_map["COM|q2_validation_deletion.csv"] <- "03_结果/Q2_删除敏感性.csv"
.mcup_csv_map["Q2|q2_validation_negative_controls.csv"] <- "03_结果/Q2_负对照.csv"
.mcup_csv_map["COM|q2_validation_negative_controls.csv"] <- "03_结果/Q2_负对照.csv"
.mcup_csv_map["Q2|q2_validation_sensitivity.csv"] <- "03_结果/Q2_敏感性检验.csv"
.mcup_csv_map["COM|q2_validation_sensitivity.csv"] <- "03_结果/Q2_敏感性检验.csv"
.mcup_csv_map["Q2|q2_validation_stability.csv"] <- "03_结果/Q2_稳定性核验.csv"
.mcup_csv_map["COM|q2_validation_stability.csv"] <- "03_结果/Q2_稳定性核验.csv"
.mcup_csv_map["Q2|图1_阈值与网络结构.csv"] <- "03_结果/Q2_图1_阈值与网络结构.csv"
.mcup_csv_map["COM|图1_阈值与网络结构.csv"] <- "03_结果/Q2_图1_阈值与网络结构.csv"
.mcup_csv_map["Q2|图2a_模型性能.csv"] <- "03_结果/Q2_图2a_模型性能.csv"
.mcup_csv_map["COM|图2a_模型性能.csv"] <- "03_结果/Q2_图2a_模型性能.csv"
.mcup_csv_map["Q2|图2b_四路一致性.csv"] <- "03_结果/Q2_图2b_四路一致性.csv"
.mcup_csv_map["COM|图2b_四路一致性.csv"] <- "03_结果/Q2_图2b_四路一致性.csv"
.mcup_csv_map["Q2|图3_候选稳定性.csv"] <- "03_结果/Q2_图3_候选稳定性.csv"
.mcup_csv_map["COM|图3_候选稳定性.csv"] <- "03_结果/Q2_图3_候选稳定性.csv"
.mcup_csv_map["Q2|图3_稳定性分箱汇总.csv"] <- "03_结果/Q2_图3_稳定性分箱汇总.csv"
.mcup_csv_map["COM|图3_稳定性分箱汇总.csv"] <- "03_结果/Q2_图3_稳定性分箱汇总.csv"
.mcup_csv_map["Q2|图4_A级关系名次与OAV.csv"] <- "03_结果/Q2_图4_A级关系名次与OAV.csv"
.mcup_csv_map["COM|图4_A级关系名次与OAV.csv"] <- "03_结果/Q2_图4_A级关系名次与OAV.csv"
.mcup_csv_map["Q2|图5_目标置换阈值核验.csv"] <- "03_结果/Q2_图5_目标置换阈值核验.csv"
.mcup_csv_map["COM|图5_目标置换阈值核验.csv"] <- "03_结果/Q2_图5_目标置换阈值核验.csv"
.mcup_csv_map["Q2|图5a_影子字段入选槽.csv"] <- "03_结果/Q2_图5a_影子字段入选槽.csv"
.mcup_csv_map["COM|图5a_影子字段入选槽.csv"] <- "03_结果/Q2_图5a_影子字段入选槽.csv"
.mcup_csv_map["Q2|图5b_直接锚点.csv"] <- "03_结果/Q2_图5b_直接锚点.csv"
.mcup_csv_map["COM|图5b_直接锚点.csv"] <- "03_结果/Q2_图5b_直接锚点.csv"
.mcup_csv_map["Q3|三风格生成画像_逐样本.csv.gz"] <- "03_结果/Q3_三风格生成画像_逐样本.csv.gz"
.mcup_csv_map["COM|三风格生成画像_逐样本.csv.gz"] <- "03_结果/Q3_三风格生成画像_逐样本.csv.gz"
.mcup_csv_map["Q3|主模型_稳健性汇总.csv"] <- "03_结果/Q3_主模型_稳健性汇总.csv"
.mcup_csv_map["COM|主模型_稳健性汇总.csv"] <- "03_结果/Q3_主模型_稳健性汇总.csv"
.mcup_csv_map["Q3|主模型_逐折分布指标.csv"] <- "03_结果/Q3_主模型_逐折分布指标.csv"
.mcup_csv_map["COM|主模型_逐折分布指标.csv"] <- "03_结果/Q3_主模型_逐折分布指标.csv"
.mcup_csv_map["Q3|主模型_逐重复指标.csv"] <- "03_结果/Q3_主模型_逐重复指标.csv"
.mcup_csv_map["COM|主模型_逐重复指标.csv"] <- "03_结果/Q3_主模型_逐重复指标.csv"
.mcup_csv_map["Q3|固定主规则候选.csv"] <- "03_结果/Q3_固定主规则候选.csv"
.mcup_csv_map["COM|固定主规则候选.csv"] <- "03_结果/Q3_固定主规则候选.csv"
.mcup_csv_map["Q3|固定候选裕量敏感性.csv"] <- "03_结果/Q3_固定候选裕量敏感性.csv"
.mcup_csv_map["COM|固定候选裕量敏感性.csv"] <- "03_结果/Q3_固定候选裕量敏感性.csv"
.mcup_csv_map["Q3|跨风格迁移_候选明细.csv"] <- "03_结果/Q3_跨风格迁移_候选明细.csv"
.mcup_csv_map["COM|跨风格迁移_候选明细.csv"] <- "03_结果/Q3_跨风格迁移_候选明细.csv"
.mcup_csv_map["Q3|跨风格迁移_化学谱.csv.gz"] <- "03_结果/Q3_跨风格迁移_化学谱.csv.gz"
.mcup_csv_map["COM|跨风格迁移_化学谱.csv.gz"] <- "03_结果/Q3_跨风格迁移_化学谱.csv.gz"
.mcup_csv_map["Q3|跨风格迁移_感官画像.csv.gz"] <- "03_结果/Q3_跨风格迁移_感官画像.csv.gz"
.mcup_csv_map["COM|跨风格迁移_感官画像.csv.gz"] <- "03_结果/Q3_跨风格迁移_感官画像.csv.gz"
.mcup_csv_map["Q3|跨风格迁移_汇总.csv"] <- "03_结果/Q3_跨风格迁移_汇总.csv"
.mcup_csv_map["COM|跨风格迁移_汇总.csv"] <- "03_结果/Q3_跨风格迁移_汇总.csv"
.mcup_csv_map["Q3|主模型逐重复检验.csv"] <- "03_结果/Q3_主模型逐重复检验.csv"
.mcup_csv_map["COM|主模型逐重复检验.csv"] <- "03_结果/Q3_主模型逐重复检验.csv"
.mcup_csv_map["Q3|主模型逐风格检验.csv"] <- "03_结果/Q3_主模型逐风格检验.csv"
.mcup_csv_map["COM|主模型逐风格检验.csv"] <- "03_结果/Q3_主模型逐风格检验.csv"
.mcup_csv_map["Q3|主模型逐风格逐重复.csv"] <- "03_结果/Q3_主模型逐风格逐重复.csv"
.mcup_csv_map["COM|主模型逐风格逐重复.csv"] <- "03_结果/Q3_主模型逐风格逐重复.csv"
.mcup_csv_map["Q3|固定生成谱诊断.csv"] <- "03_结果/Q3_固定生成谱诊断.csv"
.mcup_csv_map["COM|固定生成谱诊断.csv"] <- "03_结果/Q3_固定生成谱诊断.csv"
.mcup_csv_map["Q3|独立置换汇总.csv"] <- "03_结果/Q3_独立置换汇总.csv"
.mcup_csv_map["COM|独立置换汇总.csv"] <- "03_结果/Q3_独立置换汇总.csv"
.mcup_csv_map["Q3|独立置换诊断.csv"] <- "03_结果/Q3_独立置换诊断.csv"
.mcup_csv_map["COM|独立置换诊断.csv"] <- "03_结果/Q3_独立置换诊断.csv"
.mcup_csv_map["Q3|独立置换逐重复差.csv"] <- "03_结果/Q3_独立置换逐重复差.csv"
.mcup_csv_map["COM|独立置换逐重复差.csv"] <- "03_结果/Q3_独立置换逐重复差.csv"
.mcup_csv_map["Q3|生成缓存完整性.csv"] <- "03_结果/Q3_生成缓存完整性.csv"
.mcup_csv_map["COM|生成缓存完整性.csv"] <- "03_结果/Q3_生成缓存完整性.csv"
.mcup_csv_map["Q3|结构消融汇总.csv"] <- "03_结果/Q3_结构消融汇总.csv"
.mcup_csv_map["COM|结构消融汇总.csv"] <- "03_结果/Q3_结构消融汇总.csv"
.mcup_csv_map["Q3|结构消融逐重复.csv"] <- "03_结果/Q3_结构消融逐重复.csv"
.mcup_csv_map["COM|结构消融逐重复.csv"] <- "03_结果/Q3_结构消融逐重复.csv"
.mcup_csv_map["Q3|表6_补充检验.csv"] <- "03_结果/Q3_表6_补充检验.csv"
.mcup_csv_map["COM|表6_补充检验.csv"] <- "03_结果/Q3_表6_补充检验.csv"
.mcup_csv_map["Q3|表7_结构消融.csv"] <- "03_结果/Q3_表7_结构消融.csv"
.mcup_csv_map["COM|表7_结构消融.csv"] <- "03_结果/Q3_表7_结构消融.csv"
.mcup_csv_map["Q3|迁移阶段分母核验.csv"] <- "03_结果/Q3_迁移阶段分母核验.csv"
.mcup_csv_map["COM|迁移阶段分母核验.csv"] <- "03_结果/Q3_迁移阶段分母核验.csv"
.mcup_csv_map["Q3|图1_十次重复稳健指标.csv"] <- "03_结果/Q3_图1_十次重复稳健指标.csv"
.mcup_csv_map["COM|图1_十次重复稳健指标.csv"] <- "03_结果/Q3_图1_十次重复稳健指标.csv"
.mcup_csv_map["Q3|图2_三风格六维感官画像.csv"] <- "03_结果/Q3_图2_三风格六维感官画像.csv"
.mcup_csv_map["COM|图2_三风格六维感官画像.csv"] <- "03_结果/Q3_图2_三风格六维感官画像.csv"
.mcup_csv_map["Q3|图3_各风格状态计数.csv"] <- "03_结果/Q3_图3_各风格状态计数.csv"
.mcup_csv_map["COM|图3_各风格状态计数.csv"] <- "03_结果/Q3_图3_各风格状态计数.csv"
.mcup_csv_map["Q3|图3_固定主规则迁移状态.csv"] <- "03_结果/Q3_图3_固定主规则迁移状态.csv"
.mcup_csv_map["COM|图3_固定主规则迁移状态.csv"] <- "03_结果/Q3_图3_固定主规则迁移状态.csv"
.mcup_csv_map["公共|compound_name_map.csv"] <- "02_数据_不含赛题原始数据/化合物名称映射.csv"
.mcup_csv_map["COM|compound_name_map.csv"] <- "02_数据_不含赛题原始数据/化合物名称映射.csv"
.mcup_csv_map["公共|感官目标中文与模板列对照.csv"] <- "02_数据_不含赛题原始数据/感官目标对照.csv"
.mcup_csv_map["COM|感官目标中文与模板列对照.csv"] <- "02_数据_不含赛题原始数据/感官目标对照.csv"
.mcup_csv_map["Q1|外层交叉验证划分.csv"] <- "02_数据_不含赛题原始数据/Q1_外层交叉验证划分.csv"
.mcup_csv_map["COM|外层交叉验证划分.csv"] <- "02_数据_不含赛题原始数据/Q1_外层交叉验证划分.csv"
.mcup_csv_map["Q1|特征尺度审计.csv"] <- "02_数据_不含赛题原始数据/Q1_特征尺度审计.csv"
.mcup_csv_map["COM|特征尺度审计.csv"] <- "02_数据_不含赛题原始数据/Q1_特征尺度审计.csv"
.mcup_csv_map["Q1|缺失结构.csv"] <- "02_数据_不含赛题原始数据/Q1_缺失结构.csv"
.mcup_csv_map["COM|缺失结构.csv"] <- "02_数据_不含赛题原始数据/Q1_缺失结构.csv"
.mcup_csv_map["Q1|预处理步骤.csv"] <- "02_数据_不含赛题原始数据/Q1_预处理步骤.csv"
.mcup_csv_map["COM|预处理步骤.csv"] <- "02_数据_不含赛题原始数据/Q1_预处理步骤.csv"
.mcup_csv_map["Q1|感官目标字典.csv"] <- "02_数据_不含赛题原始数据/Q1_感官目标字典.csv"
.mcup_csv_map["COM|感官目标字典.csv"] <- "02_数据_不含赛题原始数据/Q1_感官目标字典.csv"
.mcup_csv_map["Q2|严格单体候选字典.csv"] <- "02_数据_不含赛题原始数据/Q2_严格单体候选字典.csv"
.mcup_csv_map["COM|严格单体候选字典.csv"] <- "02_数据_不含赛题原始数据/Q2_严格单体候选字典.csv"
.mcup_csv_map["Q2|感官目标字典.csv"] <- "02_数据_不含赛题原始数据/Q2_感官目标字典.csv"
.mcup_csv_map["Q3|字段契约.csv"] <- "02_数据_不含赛题原始数据/Q3_字段契约.csv"
.mcup_csv_map["COM|字段契约.csv"] <- "02_数据_不含赛题原始数据/Q3_字段契约.csv"
.mcup_csv_map["Q3|预处理步骤.csv"] <- "02_数据_不含赛题原始数据/Q3_预处理步骤.csv"
.mcup_csv_map["Q3|缺陷字段登记.csv"] <- "02_数据_不含赛题原始数据/Q3_缺陷字段登记.csv"
.mcup_csv_map["COM|缺陷字段登记.csv"] <- "02_数据_不含赛题原始数据/Q3_缺陷字段登记.csv"
.mcup_csv_map["Q3|测量状态.csv"] <- "02_数据_不含赛题原始数据/Q3_测量状态.csv"
.mcup_csv_map["COM|测量状态.csv"] <- "02_数据_不含赛题原始数据/Q3_测量状态.csv"
.mcup_csv_map["Q3|源样本缺陷状态.csv"] <- "02_数据_不含赛题原始数据/Q3_源样本缺陷状态.csv"
.mcup_csv_map["COM|源样本缺陷状态.csv"] <- "02_数据_不含赛题原始数据/Q3_源样本缺陷状态.csv"
.mcup_csv_map["Q3|风格样本数.csv"] <- "02_数据_不含赛题原始数据/Q3_风格样本数.csv"
.mcup_csv_map["COM|风格样本数.csv"] <- "02_数据_不含赛题原始数据/Q3_风格样本数.csv"
.mcup_csv_path <- function(path) {
  key <- paste(.mcup_question, basename(path), sep="|")
  if (!(key %in% names(.mcup_csv_map))) key <- paste("COM", basename(path), sep="|")
  if (!(key %in% names(.mcup_csv_map))) stop(paste("缺少绘图输入:", path))
  file.path(.mcup_support, unname(.mcup_csv_map[[key]]))
}
.mcup_read_csv <- function(file, ...) utils::read.csv(.mcup_csv_path(file), ...)
.mcup_readr_csv <- function(file, ...) readr::read_csv(.mcup_csv_path(file), ...)
.mcup_read_lines <- function(path, ...) {
  name <- if (grepl("plot_validation_drafts",path)) "Q1_08_检验图.R" else "Q1_07_主图.R"
  readLines(file.path(.mcup_code, name), ...)
}

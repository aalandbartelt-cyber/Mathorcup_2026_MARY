# 论文出图统一样式（R / ggplot2 版，比赛主用）
#
# 风格依据：40 篇 M 杯优秀论文 615 张图表实测 + 逐页目检（见 docs/M杯优秀论文特点、题型特点.md §4.5）
#   - 白底、黑色四边框、无网格、图例内嵌、3:2 横版
#   - 中文宋体 + 拉丁 Times 风格（SimSun 自带，一次设置中英统一）
#   - 训练曲线四色：蓝/橙/绿/红；PR 总曲线深蓝加粗；混淆矩阵 Blues；热力图 hot
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

FONT_PAPER <- "SimSun"   # 中文宋体；拉丁字形近似 Times
FONT_SANS  <- "SimHei"   # 黑体（标题/演示用）

CYCLE <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
NAVY  <- "#00007f"

# M 杯风格主题
# legend: "br"=右下、"tr"=右上、"none"=不要图例
theme_mcup <- function(mode = c("paper", "sans"), base_size = 10, legend = "br",
                        title_family = NULL) {
  mode <- match.arg(mode)
  fam <- if (mode == "paper") FONT_PAPER else FONT_SANS
  lg <- switch(legend,
    br = list(pos = c(1, 0), just = c(1, 0)),
    tr = list(pos = c(1, 1), just = c(1, 1)),
    none = NULL,
    stop("legend 必须是 br / tr / none")
  )
  th <- theme_bw(base_size = base_size, base_family = fam) +
    theme(
      panel.grid = element_blank(),                        # 无网格（M 杯主流）
      panel.border = element_rect(colour = "black", linewidth = 0.6),
      axis.text = element_text(colour = "black"),
      axis.ticks = element_line(colour = "black", linewidth = 0.4),
      plot.title = element_text(size = base_size + 1, hjust = 0.5,
                                family = if (is.null(title_family)) fam else title_family),
      plot.subtitle = element_text(size = base_size, hjust = 0.5),
      legend.background = element_rect(fill = "white", colour = "grey70", linewidth = 0.2),
      legend.key = element_blank(),
      legend.title = element_blank()
    )
  if (is.null(lg)) {
    th <- th + theme(legend.position = "none")
  } else {
    th <- th + theme(
      legend.position = "inside",
      legend.position.inside = lg$pos,
      legend.justification = lg$just
    )
  }
  th
}

# 保存 300dpi PNG（论文印刷标准）
save_paper <- function(plot, file, width = 6, height = 4, dpi = 300) {
  dir.create(dirname(file), showWarnings = FALSE, recursive = TRUE)
  agg_png(file, width = width, height = height, units = "in", res = dpi, background = "white")
  on.exit(invisible(dev.off()), add = TRUE)
  print(plot)
  invisible(file)
}

# 混淆矩阵：Blues 配色 + 每格数值（深色格白字）
plot_confusion_matrix <- function(cm, classes, file = NULL,
                                  normalize = FALSE, width = 4.4, height = 3.8) {
  stopifnot(is.matrix(cm), nrow(cm) == ncol(cm), length(classes) == nrow(cm))
  df <- expand.grid(pred = classes, true = classes)
  df$n <- as.vector(t(cm))                       # 行=true, 列=pred
  if (normalize) {
    df$value <- df$n / ave(df$n, df$true, FUN = sum)
    lab <- sprintf("%.2f", df$value)
  } else {
    df$value <- df$n
    lab <- as.character(df$n)
  }
  thresh <- max(df$value) / 2
  df$txt <- ifelse(df$value > thresh, "white", "black")
  p <- ggplot(df, aes(pred, true, fill = value)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    geom_text(aes(label = lab, colour = txt), size = 3) +
    scale_colour_identity() +
    scale_fill_gradient(low = "#F7FBFF", high = "#08306B", name = NULL) +
    scale_y_discrete(limits = rev(classes)) +
    labs(x = "预测标签", y = "真实标签") +
    coord_fixed() +
    theme_mcup(legend = "none") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  if (!is.null(file)) save_paper(p, file, width = width, height = height)
  p
}

# PR 曲线：各类细线 + 总曲线深蓝加粗，图例带 AP
# curves: list(类名 = list(recall = ..., precision = ..., ap = 0.804), ...)
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
    ap <- df$ap[df$cls == if (k == "all classes") c("all") else k][1]
    if (is.na(ap)) ap <- df$ap[df$is_all][1]
    if (k == "all classes") sprintf("all classes %.3f mAP@0.5", ap) else sprintf("%s %.3f", k, ap)
  }, character(1))
  df$grp <- factor(ifelse(df$is_all, "all classes", df$cls),
                   levels = lev, labels = lab)
  col_map <- setNames(c(CYCLE[seq_along(others)], NAVY), lab)
  lw_map <- setNames(ifelse(lev == "all classes", 1.4, 0.7), lab)
  p <- ggplot(df, aes(recall, precision, colour = grp, linewidth = grp)) +
    geom_line() +
    scale_colour_manual(values = col_map) +
    scale_linewidth_manual(values = lw_map, guide = "none") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.01)) +
    labs(x = "Recall", y = "Precision", title = "Precision-Recall Curve",
         colour = NULL) +
    theme_mcup(legend = "tr")
  if (!is.null(file)) save_paper(p, file, width = width, height = height)
  p
}

# 相关性热力图：hot 配色（黑→红→橙→黄→白，即负值深、正值浅）+ 全格数值
plot_corr_heatmap <- function(mat, file = NULL, width = 5.4, height = 4.8) {
  stopifnot(is.matrix(mat), nrow(mat) == ncol(mat))
  nm <- rownames(mat); if (is.null(nm)) nm <- paste0("V", seq_len(nrow(mat)))
  df <- expand.grid(var2 = nm, var1 = nm)
  df$value <- as.vector(t(mat))
  df$txt <- ifelse(df$value < -0.2, "white", "black")
  p <- ggplot(df, aes(var2, var1, fill = value)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", value), colour = txt), size = 2.6) +
    scale_colour_identity() +
    scale_fill_gradientn(colours = c("black", "red", "orange", "yellow", "white"),
                         limits = c(-1, 1), name = NULL) +
    scale_y_discrete(limits = rev(nm)) +
    labs(x = NULL, y = NULL, title = "相关性热力图") +
    coord_fixed() +
    theme_mcup(legend = "none") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  if (!is.null(file)) save_paper(p, file, width = width, height = height)
  p
}

# 特征重要性：横向降序彩条 + 末端数值
plot_feature_importance <- function(names, values, file = NULL,
                                    width = 6, height = NULL, top_n = 15) {
  ord <- order(values, decreasing = TRUE)[seq_len(min(top_n, length(values)))]
  df <- data.frame(name = factor(names[ord], levels = rev(names[ord])),
                   value = values[ord])
  h <- if (is.null(height)) max(3.2, 0.32 * nrow(df) + 1) else height
  p <- ggplot(df, aes(value, name, fill = value)) +
    geom_col(width = 0.72) +
    geom_text(aes(label = sprintf("%.1f", value)), hjust = -0.12, size = 2.6) +
    scale_fill_gradientn(colours = c("#4575B4", "#91BFDB", "#FEE090", "#FC8D59", "#D73027"),
                         guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = "Relative Importance", y = NULL, title = "特征重要性") +
    theme_mcup(legend = "none")
  if (!is.null(file)) save_paper(p, file, width = width, height = h)
  p
}

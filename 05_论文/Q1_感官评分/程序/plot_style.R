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

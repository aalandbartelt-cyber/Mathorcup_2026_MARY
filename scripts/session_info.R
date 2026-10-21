#!/usr/bin/env Rscript
# 生成 R 环境快照（满足"代码可复现/依赖说明"的交付要求）
# 用法：在项目根目录执行 Rscript scripts/session_info.R

out_dir <- "docs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out <- file.path(out_dir, "env_r.txt")

con <- file(out, "w", encoding = "UTF-8")
sink(con)
cat("R 环境快照\n")
cat("生成时间:", format(Sys.time()), "\n\n")
print(sessionInfo())

cat("\n== 关键包版本 ==\n")
pkgs <- c("ggplot2", "dplyr", "tidyr", "readr", "data.table", "scales",
          "patchwork", "ggrepel", "ragg", "svglite", "showtext", "languageserver")
for (p in pkgs) {
  v <- tryCatch(as.character(packageVersion(p)), error = function(e) "未安装")
  cat(sprintf("%-16s %s\n", p, v))
}
sink()
close(con)
cat("已写入", out, "\n")

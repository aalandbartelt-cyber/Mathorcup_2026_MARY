#!/usr/bin/env Rscript
# Frozen plots, unchanged numeric data and geometry. Only textual dash glyphs.
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
source("scripts/plot_style.R")
OUT <- "05_论文/总论文/图表_v3_字符统一"
dir.create(OUT,recursive=TRUE,showWarnings=FALSE)

read_plot <- function(lines) {
  env <- new.env(parent=globalenv())
  for (expr in parse(text=lines)) {
    head <- if (is.call(expr)) as.character(expr[[1]])[1] else ""
    if (head %in% c("save_paper","save_figure","save_v4","write_csv","cat","dir.create")) next
    eval(expr,envir=env)
  }
  env
}
export <- function(p,stem,w,h) {
  save_paper(p,file.path(OUT,paste0(stem,".png")),width=w,height=h,dpi=300)
  ggsave(file.path(OUT,paste0(stem,".pdf")),p,width=w,height=h,device=cairo_pdf,
         family=FONT_PAPER,bg="white")
  ggsave(file.path(OUT,paste0(stem,".svg")),p,width=w,height=h,device=svglite::svglite,bg="white")
}

q1 <- readLines("02_代码/Q1_感官评分/src/plot_q1_final.R",warn=FALSE)
# Do not change categorical keys/levels: they match immutable source CSVs.
keys <- grepl("scale_x_discrete\\(labels|labs\\(x = .*y = ",q1)
q1[keys] <- gsub("[—–]","-",q1[keys])
q1env <- read_plot(q1)
q1original <- read_plot(readLines("02_代码/Q1_感官评分/src/plot_q1_final.R",warn=FALSE))
export(q1env$p1,"图2_模型比较",8.2,4.35)
supp <- q1env$p_s1_combined & theme(text=element_text(family="SimSun"),
       legend.text=element_text(family="SimSun",size=10.5,face="plain"))
export(supp,"附图S1_残差与量表",9.5,7.0)

q2src <- ".worktrees/q2-evidence-opt/04_图表/Q2_关键化合物/v4"
q2 <- readLines(file.path(q2src,"plot_q2_paper_v4.R"),warn=FALSE)
q2[grepl("^figure_dir <-",q2)] <- paste0('figure_dir <- "',q2src,'"')
q2[grepl("^relation_labels <-",q2)] <- gsub("[—–]","-",q2[grepl("^relation_labels <-",q2)])
q2env <- read_plot(q2)
q2original_lines <- readLines(file.path(q2src,"plot_q2_paper_v4.R"),warn=FALSE)
q2original_lines[grepl("^figure_dir <-",q2original_lines)] <- paste0('figure_dir <- "',q2src,'"')
q2original <- read_plot(q2original_lines)
export(q2env$fig4,"图11_A级关系",8.2,4.6)

q3src <- ".worktrees/q3-foundation/04_图表/Q3_风格生成/draft_v2"
q3 <- readLines(file.path(q3src,"plot_q3_figures.R"),warn=FALSE)
q3[grepl("^figure_dir <-",q3)] <- paste0('figure_dir <- "',q3src,'"')
q3 <- gsub("[—–]","-",q3)
q3env <- read_plot(q3)
q3original_lines <- readLines(file.path(q3src,"plot_q3_figures.R"),warn=FALSE)
q3original_lines[grepl("^figure_dir <-",q3original_lines)] <- paste0('figure_dir <- "',q3src,'"')
q3original <- read_plot(q3original_lines)
export(q3env$fig1,"图14_生成质量",8.4,5.35)
export(q3env$fig2,"图15_三风格画像",8.4,4.8)
numeric_layers <- function(p) {
  children <- lapply(p$patches$plots,numeric_layers)
  built <- ggplot_build(p)
  own <- lapply(built$data,function(d)d[,vapply(d,is.numeric,logical(1)),drop=FALSE])
  c(children,list(own))
}
before <- list(q1original$p1,q1original$p_s1_combined,q2original$fig4,q3original$fig1,q3original$fig2)
after <- list(q1env$p1,q1env$p_s1_combined,q2env$fig4,q3env$fig1,q3env$fig2)
checks <- vapply(seq_along(before),function(i)isTRUE(all.equal(numeric_layers(before[[i]]),numeric_layers(after[[i]]),tolerance=0)),logical(1))
write.csv(data.frame(figure=c("图2","附图S1","图11","图14","图15"),numeric_plot_layers_unchanged=checks),
          file.path(OUT,"数值图层核验.csv"),row.names=FALSE)
stopifnot(all(checks))
capture.output(sessionInfo(),file=file.path(OUT,"R_session_info.txt"))
cat("Five plots exported; source data and original PNGs remain unchanged.\n")

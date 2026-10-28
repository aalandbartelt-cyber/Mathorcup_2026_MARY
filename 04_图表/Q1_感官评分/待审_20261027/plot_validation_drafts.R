#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(patchwork); library(scales)
})
source("scripts/plot_style.R")
OUT <- "04_图表/Q1_感官评分/待审_20261027"
read_plot <- function(name) read.csv(file.path(OUT,name),check.names=FALSE)

BLUE <- "#1F5A94"; LIGHTBLUE <- "#88A9D3"; GREEN <- "#67A995"
GOLD <- "#D4A35B"; GREY <- "#8C8C8C"
paper_theme <- function(size=11,legend="bottom") {
  theme_mcup(base_size=size,legend="none") +
    theme(legend.position=legend,legend.direction="horizontal",
      legend.text=element_text(size=size-1.0),
      legend.key.width=unit(14,"pt"),legend.key.height=unit(11,"pt"),
      legend.spacing.x=unit(3,"pt"),legend.margin=margin(3,0,0,0),
      plot.subtitle=element_text(face="bold",size=size,hjust=0,margin=margin(b=6)),
      axis.title.x=element_text(margin=margin(t=5)),
      axis.title.y=element_text(margin=margin(r=5)),
      plot.margin=margin(5,6,5,5))
}
save_both <- function(p,stem,w,h) {
  save_paper(p,file.path(OUT,paste0(stem,".png")),width=w,height=h,dpi=300)
  ggsave(file.path(OUT,paste0(stem,".pdf")),p,width=w,height=h,units="in",
         device=cairo_pdf,family=FONT_PAPER,bg="white")
}

rep <- read_plot("model_repeats.csv") %>% rename(repeat_id="repeat") %>%
  mutate(model=factor(model,levels=c("基线","最终模型")))
model_colors <- c("基线"=LIGHTBLUE,"最终模型"=BLUE)
model_shapes <- c("基线"=1,"最终模型"=16)
model_lines <- c("基线"=2,"最终模型"=1)
repeat_panel <- function(metric,label,subtitle) {
  ggplot(rep,aes(repeat_id+1,.data[[metric]],colour=model,shape=model,linetype=model))+
    geom_line(linewidth=.55)+geom_point(size=2.2,stroke=.7)+
    scale_colour_manual(values=model_colors)+scale_shape_manual(values=model_shapes)+
    scale_linetype_manual(values=model_lines)+
    scale_x_continuous(breaks=1:10,expand=expansion(mult=c(.035,.04)))+
    scale_y_continuous(expand=expansion(mult=c(.12,.14)))+
    labs(x="重复序号",y=label,subtitle=subtitle)+paper_theme()+
    guides(colour=guide_legend(nrow=1))
}
p_a <- repeat_panel("r2",expression("宏平均 "*R^2),"(a) 重复验证解释度")
p_b <- repeat_panel("mae","宏平均 MAE","(b) 重复验证绝对误差")

cal <- read_plot("calibration_bins.csv") %>%
  mutate(group_plot=factor(group_plot,levels=c("香气","风味","味觉/口感","总体")))
group_colors <- c("香气"=BLUE,"风味"=GREEN,"味觉/口感"=GOLD,"总体"=GREY)
group_shapes <- c("香气"=16,"风味"=17,"味觉/口感"=15,"总体"=18)
p_c <- ggplot(cal,aes(mean_pred,mean_truth,colour=group_plot,shape=group_plot))+
  geom_abline(slope=1,intercept=0,colour="#555555",linetype=3,linewidth=.45)+
  geom_point(size=1.8,alpha=.82)+
  scale_colour_manual(values=group_colors,drop=FALSE)+
  scale_shape_manual(values=group_shapes,drop=FALSE)+
  labs(x="预测均值（五分位箱）",y="观测均值",subtitle="(c) 逐目标样本内校准")+
  paper_theme()+guides(colour=guide_legend(nrow=1,override.aes=list(alpha=1,size=2.3)))

cov <- read_plot("interval_summary.csv") %>%
  mutate(method=factor(method,levels=c("重复标准差","残差校准","名义=实际")))
ref <- data.frame(x=c(.78,.97),y=c(.78,.97),method="名义=实际")
cov_cols <- c("重复标准差"=GOLD,"残差校准"=BLUE,"名义=实际"="#555555")
cov_shapes <- c("重复标准差"=17,"残差校准"=16,"名义=实际"=NA)
cov_lines <- c("重复标准差"=2,"残差校准"=1,"名义=实际"=3)
p_d <- ggplot(cov,aes(nominal,mean_cov,colour=method,shape=method,linetype=method))+
  geom_errorbar(aes(ymin=q10,ymax=q90),width=.008,linewidth=.45)+
  geom_line(linewidth=.55)+geom_point(size=2.3)+
  geom_line(data=ref,aes(x,y,colour=method,linetype=method),inherit.aes=FALSE,linewidth=.45)+
  scale_colour_manual(values=cov_cols,limits=names(cov_cols),drop=FALSE)+
  scale_shape_manual(values=cov_shapes,limits=names(cov_shapes),drop=FALSE)+
  scale_linetype_manual(values=cov_lines,limits=names(cov_lines),drop=FALSE)+
  scale_x_continuous(breaks=c(.8,.9,.95),labels=percent_format(accuracy=1),limits=c(.78,.97))+
  scale_y_continuous(breaks=seq(0,1,.2),labels=percent_format(accuracy=1),limits=c(.08,1.02))+
  labs(x="名义覆盖率",y="实际覆盖率",subtitle="(d) 区间覆盖诊断")+
  paper_theme()+
  guides(colour=guide_legend(nrow=1,override.aes=list(shape=c(17,16,NA),linetype=c(2,1,3))),
         shape="none",linetype="none")
p5 <- ((p_a|p_b)/(p_c|p_d))+plot_layout(heights=c(1,1.15))
save_both(p5,"图5_重复验证稳定性与预测校准",8.4,5.6)

abl <- read_plot("ablation_summary.csv") %>%
  mutate(setting=factor(setting,levels=c("完整模型","去汇总特征","标签置换")))
setting_colors <- c("完整模型"=BLUE,"去汇总特征"=GREEN,"标签置换"=GOLD)
s2a <- ggplot(abl,aes(setting,r2,fill=setting))+
  geom_hline(yintercept=0,colour="#555555",linetype=3,linewidth=.4)+
  geom_col(width=.5,colour="#555555",linewidth=.2)+
  geom_text(aes(label=sprintf("%.4f",r2)),vjust=ifelse(abl$r2>=0,-.65,1.35),
            size=3.2,family=FONT_PAPER)+
  scale_fill_manual(values=setting_colors)+
  scale_y_continuous(limits=c(-.035,.30),expand=expansion(mult=c(.02,.02)))+
  labs(x=NULL,y=expression("宏平均 "*R^2),subtitle="(a) 结构消融与标签置换")+
  paper_theme(10.5,"none")
s2b <- ggplot(abl,aes(setting,mae,colour=setting,shape=setting))+
  geom_point(size=3,stroke=.8)+
  geom_text(aes(label=sprintf("%.4f",mae)),vjust=-1.0,size=3.2,family=FONT_PAPER,colour="#333333")+
  scale_colour_manual(values=setting_colors)+
  scale_shape_manual(values=c("完整模型"=16,"去汇总特征"=17,"标签置换"=15))+
  scale_y_continuous(limits=c(.32,.442),breaks=c(.32,.36,.40,.44))+
  labs(x=NULL,y="宏平均 MAE",subtitle="(b) 单次重复的误差比较")+
  paper_theme(10.5,"none")

leave <- read_plot("leave_style_out.csv") %>%
  arrange(spearman) %>%
  mutate(style_label=paste0(style_zh," (n=",n,")"),
         style_label=factor(style_label,levels=style_label),
         sample_size=factor(ifelse(n>=10,"n≥10","n<10"),levels=c("n≥10","n<10")))
size_colors <- c("n≥10"=BLUE,"n<10"=GOLD)
size_shapes <- c("n≥10"=16,"n<10"=2)
s2c <- ggplot(leave,aes(spearman,style_label,colour=sample_size,shape=sample_size))+
  geom_vline(xintercept=0,colour="#555555",linetype=3,linewidth=.4)+
  geom_point(size=2.2,stroke=.8)+
  geom_text(aes(x=ifelse(spearman<0,.02,spearman+.015),label=sprintf("%.2f",spearman)),
            hjust=0,size=2.8,family=FONT_PAPER,colour="#333333",show.legend=FALSE)+
  scale_colour_manual(values=size_colors)+scale_shape_manual(values=size_shapes)+
  scale_x_continuous(limits=c(-.085,.50),breaks=seq(0,.4,.1),expand=c(0,0))+
  scale_y_discrete(expand=expansion(add=.7))+
  labs(x="留一风格 Spearman",y=NULL,subtitle="(c) 新风格排序压力测试")+
  paper_theme(10.5)+theme(axis.text.y=element_text(size=9.5))
s2d <- ggplot(leave,aes(mae,style_label,colour=sample_size,shape=sample_size))+
  geom_point(size=2.2,stroke=.8)+
  geom_text(aes(label=sprintf("%.3f",mae)),nudge_x=.009,hjust=0,size=2.8,
            family=FONT_PAPER,colour="#333333",show.legend=FALSE)+
  scale_colour_manual(values=size_colors)+scale_shape_manual(values=size_shapes)+
  scale_x_continuous(limits=c(.24,.59),breaks=seq(.25,.55,.1),expand=c(0,0))+
  scale_y_discrete(expand=expansion(add=.7))+
  labs(x="留一风格 MAE",y=NULL,subtitle="(d) 新风格误差压力测试")+
  paper_theme(10.5)+theme(axis.text.y=element_blank())
s2bottom <- (s2c|s2d)+plot_layout(widths=c(1.4,1),guides="collect")&
  theme(legend.position="bottom")
s2 <- (s2a|s2b)/s2bottom+plot_layout(heights=c(1,2.1))
save_both(s2,"附图S2_消融与负对照",8.6,7.2)

rel <- read_plot("target_reliability.csv") %>% arrange(r2) %>%
  mutate(panel=ifelse(row_number()<=25,"left","right"),
         tier=factor(reliability_tier,levels=c("较低","中等","较高")))
tier_colors <- c("较低"=GREY,"中等"=GOLD,"较高"=BLUE)
tier_shapes <- c("较低"=16,"中等"=17,"较高"=15)
reliability_panel <- function(d,label) {
  d <- d %>% mutate(target_zh=factor(target_zh,levels=target_zh[order(r2)]))
  ggplot(d,aes(r2,target_zh,colour=tier,shape=tier))+
    geom_vline(xintercept=0,colour="#555555",linetype=3,linewidth=.4)+
    geom_segment(aes(x=r2_ci_low,xend=r2_ci_high,yend=target_zh),linewidth=.5,show.legend=TRUE)+
    geom_point(size=2.2,stroke=.6,show.legend=TRUE)+
    scale_colour_manual(values=tier_colors,name="相对可靠性",drop=FALSE)+
    scale_shape_manual(values=tier_shapes,name="相对可靠性",drop=FALSE)+
    scale_x_continuous(limits=c(-.05,.88),breaks=seq(0,.8,.2),expand=c(0,0))+
    scale_y_discrete(expand=expansion(add=.7))+
    labs(x=expression("平均OOF "*R^2),y=NULL,subtitle=label)+
    paper_theme(11)+theme(axis.text.y=element_text(size=10.5))
}
s3 <- (reliability_panel(filter(rel,panel=="left"),"(a) 较低解释度目标")|
       reliability_panel(filter(rel,panel=="right"),"(b) 较高解释度目标"))+
  plot_layout(guides="collect")&theme(legend.position="bottom")
save_both(s3,"附图S3_逐目标预测可靠性",8.6,7.0)
cat("Three drafts written to",OUT,"\n")

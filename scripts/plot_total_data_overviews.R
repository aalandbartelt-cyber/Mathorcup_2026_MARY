#!/usr/bin/env Rscript
# Data-only descriptive figures; no fitting, selection or experiment execution.
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(scales)})
source("scripts/plot_style.R")
OUT <- "05_论文/总论文/图稿_待审计/数据概览"
DATA <- file.path(OUT,"绘图数据")
BLUE <- "#1F5A94"; GREEN <- "#67A995"; ORANGE <- "#D4A35B"; GRAY <- "#8A929C"
FONT <- "SimSun"; PT <- 72.27/25.4
read_d <- function(name) read.csv(file.path(DATA,name),check.names=FALSE)
th <- function() theme_mcup(base_size=12,legend="none") + theme(
  panel.grid=element_blank(),panel.border=element_rect(colour="black",fill=NA,linewidth=.45),
  axis.text=element_text(size=11,colour="#202020"),axis.title=element_text(size=11.5,face="bold"),
  plot.subtitle=element_text(size=12,face="bold",hjust=0,margin=margin(b=7)),
  plot.margin=margin(7,13,6,8),legend.position="bottom",legend.text=element_text(size=11),
  legend.title=element_blank(),legend.box="horizontal")
save_both <- function(p,name,w,h) {
  save_paper(p,file.path(OUT,paste0(name,".png")),width=w,height=h,dpi=300)
  ggsave(file.path(OUT,paste0(name,".pdf")),p,width=w,height=h,device=cairo_pdf,family=FONT,bg="white")
}

missing <- read_d("q1_missing_nine_fields_zh.csv")
missing <- missing[order(missing$missing_cells,missing$feature_zh),]
missing$label <- factor(missing$feature_zh,levels=missing$feature_zh)
p1a <- ggplot(missing,aes(missing_cells,label)) +
  geom_col(fill=BLUE,width=.59) +
  geom_text(aes(label=missing_cells),hjust=0,nudge_x=.6,family=FONT,size=11/PT) +
  scale_x_continuous(limits=c(0,max(missing$missing_cells)*1.2),expand=expansion(mult=c(0,0)),breaks=pretty_breaks(4)) +
  labs(x="缺失单元数",y=NULL,subtitle="（a）缺失集中在9个字段") + th()
family <- read_d("q1_target_families.csv")
count_col <- intersect(c("n_targets","target_count","count","n"),names(family))[1]
name_col <- intersect(c("group_zh","family_cn","family_zh","family","group"),names(family))[1]
family$count <- family[[count_col]];family$label <- factor(family[[name_col]],levels=rev(family[[name_col]]))
p1b <- ggplot(family,aes(count,label)) + geom_col(fill=GREEN,width=.55) +
  geom_text(aes(label=count),hjust=0,nudge_x=.4,family=FONT,size=11/PT) +
  scale_x_continuous(limits=c(0,18),breaks=c(0,5,10,15),expand=expansion(mult=c(0,0))) +
  labs(x="感官目标数",y=NULL,subtitle="（b）50个目标分属五个家族") + th()
styles <- read_d("Q1_22风格样本数_可复用.csv")
style_names <- read.csv("04_图表/Q1_感官评分/待审_20261027/leave_style_out.csv",check.names=FALSE)
name_map <- setNames(style_names$style_zh,style_names$style)
styles$style_cn <- unname(name_map[styles$style])
styles$style_cn[styles$style %in% c("Tripel","Kriek","Lambic")] <- styles$style[styles$style %in% c("Tripel","Kriek","Lambic")]
stopifnot(!anyNA(styles$style_cn))
write.csv(styles,file.path(DATA,"Q1_风格中文映射与样本数.csv"),row.names=FALSE)
styles <- styles[order(styles$n,styles$style),];styles$label <- factor(styles$style_cn,levels=styles$style_cn)
p1c <- ggplot(styles,aes(n,label)) +
  geom_segment(aes(x=0,xend=n,yend=label),colour="#C3D1DE",linewidth=.6) +
  geom_point(colour=BLUE,size=2.5) +
  geom_text(aes(label=n),hjust=0,nudge_x=.75,family=FONT,size=10.5/PT) +
  scale_x_continuous(limits=c(0,max(styles$n)+4),breaks=c(0,10,20,30),expand=expansion(mult=c(0,0))) +
  labs(x="实测啤酒数",y=NULL,subtitle="（c）22种风格的样本量") + th() +
  theme(axis.text.y=element_text(size=10.5),plot.margin=margin(7,13,6,12))
p1 <- ((p1a / p1b) | p1c) + plot_layout(widths=c(1,1.04))
save_both(p1,"Q1_数据概览_待审",9,5.9)

elig <- data.frame(category=factor(c("严格单体","预先排除"),levels=c("严格单体","预先排除")),n=c(218,13),y="化学测量")
p2a <- ggplot(elig,aes(n,y,fill=category)) + geom_col(width=.5,position=position_stack(reverse=TRUE)) +
  geom_text(aes(label=n),position=position_stack(vjust=.5,reverse=TRUE),family=FONT,size=11.5/PT,
            colour=c("white","black")) +
  scale_fill_manual(values=c("严格单体"=BLUE,"预先排除"="#D8DEE5")) +
  scale_x_continuous(limits=c(0,240),breaks=c(0,50,100,150,200,231),expand=expansion(mult=c(0,0))) +
  labs(x="字段数",y=NULL,subtitle="（a）231个测量字段的候选资格") + th() +
  theme(legend.position="bottom",plot.margin=margin(6,12,1,8),axis.text.y=element_text(size=11))
pairs <- read_d("Q2_218候选成对相关.csv")
p2b <- ggplot(pairs,aes(spearman)) +
  geom_histogram(binwidth=.05,boundary=0,fill=BLUE,colour="white",linewidth=.18) +
  scale_x_continuous(limits=c(-1,1),breaks=seq(-1,1,.5),expand=expansion(mult=c(0,0))) +
  scale_y_continuous(labels=label_comma(),expand=expansion(mult=c(0,.05))) +
  labs(x="Spearman相关系数",y="字段对数量（对）",subtitle="（b）218个单体的成对相关分布") + th()
excluded <- read_d("Q2_排除类型数量.csv")
excluded$category_cn[excluded$category_cn=="混合或聚合组成"] <- "混合/聚合组成"
excluded$category_cn[excluded$category_cn=="派生或汇总指标"] <- "派生/汇总指标"
excluded$label <- factor(excluded$category_cn,levels=rev(excluded$category_cn))
p2c <- ggplot(excluded,aes(count,label)) + geom_col(fill=GRAY,width=.5) +
  geom_text(aes(label=count),hjust=0,nudge_x=.15,family=FONT,size=11.5/PT) +
  scale_x_continuous(limits=c(0,6),breaks=0:5,expand=expansion(mult=c(0,0))) +
  labs(x="排除字段数",y=NULL,subtitle="（c）13个字段的排除类别") + th()
p2 <- p2a / (p2b | p2c) + plot_layout(heights=c(.8,2.4))
save_both(p2,"Q2_数据概览_待审",9,5.3)

complete <- read_d("Q3_实测与画像完整性.csv")
style_order <- c("Tripel","Kriek","Lambic");cols <- c(Tripel=BLUE,Kriek=GREEN,Lambic=ORANGE)
complete$style <- factor(complete$style,levels=rev(style_order))
p3a <- ggplot(complete,aes(n_real_observed,style,fill=style)) + geom_col(width=.5,show.legend=FALSE) +
  geom_text(aes(label=n_real_observed),hjust=0,nudge_x=.7,family=FONT,size=11.5/PT) +
  scale_fill_manual(values=cols) + scale_x_continuous(limits=c(0,33),breaks=c(0,10,20,30),expand=expansion(mult=c(0,0))) +
  labs(x="真实啤酒数",y=NULL,subtitle="（a）目标风格的实测样本量") + th()
obs <- read_d("Q3_实测中位数与四分位.csv")
target_levels <- c("乳酸感","酸度","4VG","酒花味","苦味","酒精感")
obs$y <- match(obs$target_cn,target_levels) + c(Tripel=.20,Kriek=0,Lambic=-.20)[obs$style]
obs$style <- factor(obs$style,levels=style_order)
p3b <- ggplot(obs,aes(colour=style,shape=style)) +
  geom_segment(aes(x=q25,xend=q75,y=y,yend=y),linewidth=.8,show.legend=TRUE) +
  geom_point(aes(x=median,y=y),size=2.7) +
  scale_colour_manual(values=cols) + scale_shape_manual(values=c(Tripel=16,Kriek=17,Lambic=15)) +
  scale_y_continuous(breaks=1:6,labels=target_levels,limits=c(.5,6.5),expand=expansion(mult=c(0,0))) +
  scale_x_continuous(breaks=pretty_breaks(5)) +
  labs(x="标准化感官值（无量纲）",y=NULL,subtitle="（b）六维实测感官分布") + th() +
  theme(legend.position="bottom",plot.margin=margin(7,13,6,12))
p3c <- ggplot(complete,aes(finite_prediction_z,style,fill=style)) +
  geom_col(width=.5,show.legend=FALSE) +
  geom_text(aes(label=format(finite_prediction_z,big.mark=",",trim=TRUE)),hjust=0,nudge_x=300,family=FONT,size=11/PT) +
  scale_fill_manual(values=cols) +
  scale_x_continuous(limits=c(0,19000),breaks=c(0,5000,10000,15000),labels=label_comma(),expand=expansion(mult=c(0,0))) +
  labs(x="既有画像中有限的预测数值（个）",y=NULL,subtitle="（c）画像接口的数值完整性") + th()
p3 <- ((p3a / p3c) | p3b) + plot_layout(widths=c(1,1.06))
save_both(p3,"Q3_数据概览_待审",9,5.5)
cairo_pdf(file.path(OUT,"一页审计说明.pdf"),width=8.27,height=11.69,family=FONT)
grid::grid.newpage()
grid::grid.text("三问数据概览图稿审计说明",x=.5,y=.95,
  gp=grid::gpar(fontfamily=FONT,fontsize=16,fontface="bold"))
audit_lines <- c(
  "状态：待用户审计；三张新图尚未插入总论文。",
  "Q1：缺失结构、目标家族与实测风格样本量",
  "（a）9字段的73个缺失单元，涉及63款啤酒；来自附件1。",
  "（b）50目标分属15/15/15/4/1五家族；来自冻结目标字典。",
  "（c）22风格共250款实测啤酒；已复用验收版中文映射。",
  "附件8只核验尺度：37个log10、33个原尺度，不代表70字段全一致。",
  "Q2：候选资格、真实共线性和排除类别",
  "（a）231字段=218严格单体候选+13个预先排除字段。",
  "（b）218候选共23,653对Spearman；成对有效样本211—250。",
  "（c）排除类别为理化3、混合/聚合4、元素1、派生/汇总5。",
  "全量相关仅描述；245个强相关对与全231字段266边的分母不同。",
  "Q3：真实样本、实测感官分布与已有画像完整性",
  "（a）Tripel/Kriek/Lambic的真实样本量为28/10/11。",
  "（b）附件4六维294值：点为中位数，线为四分位距，非置信区间。",
  "（c）各300谱×50目标=15,000有限值，总900谱、45,000个数值。",
  "画像数量不增加真实样本量；标准化感官值不是0—6原评分。",
  "导出与核对",
  "R/ggplot2 theme_mcup；300dpi PNG和矢量PDF；白底黑框无网格。",
  "各图均有70%缩放及灰度版；独立数值与视觉QA已记录。",
  "同目录：附件使用矩阵、口径汇总、绘图数据、图稿合同与QA。",
  "审阅通过后再插入5.1、6.1、7.1，连续编号并更新Word目录。"
)
ys <- seq(.885,.115,length.out=length(audit_lines))
bold_rows <- c(2,7,12,17)
for(i in seq_along(audit_lines)) grid::grid.text(audit_lines[i],x=.075,y=ys[i],hjust=0,
  gp=grid::gpar(fontfamily=FONT,fontsize=if(i %in% bold_rows) 12 else 10.5,
               fontface=if(i %in% bold_rows) "bold" else "plain"))
dev.off()
capture.output(sessionInfo(),file=file.path(OUT,"R_session_info.txt"))
cat("Three descriptive drafts exported; no model fitting.\n")

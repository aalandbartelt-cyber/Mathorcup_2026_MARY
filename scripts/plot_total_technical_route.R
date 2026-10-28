#!/usr/bin/env Rscript
# Descriptive schematic of the frozen pipeline. No models or results are fitted.
suppressPackageStartupMessages({library(ggplot2);library(grid)})
source("scripts/plot_style.R")
OUT <- "05_论文/总论文/图稿_待审计/技术路线"
dir.create(OUT,recursive=TRUE,showWarnings=FALSE)
BLUE <- "#1F5A94"; GREEN <- "#67A995"; GOLD <- "#D4A35B"
FONT <- "SimSun"; PT <- 72.27/25.4
nodes <- data.frame(
 id=c("q1_m","q1_v","q1_o","q2_m","q2_v","q2_o","q3_m","q3_v","q3_o"),
 xmin=rep(c(1.55,4.85,8.15),3),xmax=rep(c(4.3,7.6,11.3),3),
 y=rep(c(4.6,3.0,1.4),each=3),
 title=c("四类基学习器","感官组收缩NNLS","50维相对评分",
         "218个严格单体","四路稳健秩融合","50维Top-3",
         "风格条件Copula","最小改动路径搜索","三风格谱及画像"),
 detail=c("ExtraTrees / HGB\nRidge / PLS2","10×3外层；内层3折","连续预测与0-6七档",
          "共表达描述与折内图","总关联 / 分层自助\n风格校正 / 留一风格","稳定性与化学核验",
          "经验边缘与收缩相关","直接浓度与代理约束","迁移候选与适用边界"),
 fill=rep(c("#EFF4F9","#F0F6F3","#FCF7EF"),each=3),stringsAsFactors=FALSE)
nodes$x <- (nodes$xmin+nodes$xmax)/2;nodes$ymin<-nodes$y-.5;nodes$ymax<-nodes$y+.5
main_edges <- data.frame(x=rep(c(4.3,7.6),3),xend=rep(c(4.85,8.15),3),y=rep(c(4.6,3,1.4),each=2))
main_edges$yend<-main_edges$y
data_edges <- data.frame(x=rep(1.20,3),xend=rep(1.55,3),y=c(4.6,3,1.4),yend=c(4.6,3,1.4))
lanes<-data.frame(y=c(4.6,3,1.4),label=c("问题一\n感官评分","问题二\n关键化合物","问题三\n风格生成"),colour=c(BLUE,GREEN,GOLD))
heads<-data.frame(x=c(2.925,6.225,9.725),label=c("方法构建","融合与约束","结果与解释"))
frozen_lines <- data.frame(x=c(6.225,6.225,11.8,11.8),y=c(4.1,3.78,3.78,.48),
                          xend=c(6.225,11.8,11.8,6.225),yend=c(3.78,3.78,.48,.48))
frozen_arrows <- data.frame(x=c(11.8,11.8,6.225),y=c(2.70,1.4,.48),
                           xend=c(11.3,11.3,6.225),yend=c(2.70,1.4,.9))
arrow_head<-arrow(length=unit(2.2,"mm"),type="closed")
p<-ggplot()+
  geom_rect(aes(xmin=1.55,xmax=11.3,ymin=5.78,ymax=6.43),fill="#F2F3F4",colour="#303030",linewidth=.45)+
  annotate("text",x=6.425,y=6.18,label="数据核验与折内处理",family=FONT,fontface="bold",size=12/PT)+
  annotate("text",x=6.425,y=5.93,label="250款啤酒  |  231项测量  |  50维感官目标",family=FONT,size=11/PT)+
  geom_segment(aes(x=1.55,y=6.105,xend=1.20,yend=6.105),colour="#505050",linewidth=.45)+
  geom_segment(aes(x=1.20,y=6.105,xend=1.20,yend=1.4),colour="#505050",linewidth=.45)+
  geom_segment(data=data_edges,aes(x=x,y=y,xend=xend,yend=yend),colour="#505050",linewidth=.45,arrow=arrow_head)+
  geom_text(data=heads,aes(x=x,y=5.42,label=label),family=FONT,fontface="bold",size=11.5/PT)+
  geom_rect(data=nodes,aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,fill=fill),colour="#303030",linewidth=.45)+
  scale_fill_identity()+
  geom_text(data=nodes,aes(x=x,y=y+.22,label=title),family=FONT,fontface="bold",size=11/PT)+
  geom_text(data=nodes,aes(x=x,y=y-.14,label=detail),family=FONT,size=9.5/PT,lineheight=1.05)+
  geom_text(data=lanes,aes(x=.42,y=y,label=label,colour=colour),family=FONT,fontface="bold",size=11.5/PT,lineheight=1.05)+
  scale_colour_identity()+
  geom_segment(data=main_edges,aes(x=x,y=y,xend=xend,yend=yend),colour="#303030",linewidth=.55,arrow=arrow_head)+
  geom_segment(data=frozen_lines,aes(x=x,y=y,xend=xend,yend=yend),colour=BLUE,linewidth=.6,linetype="22")+
  geom_segment(data=frozen_arrows,aes(x=x,y=y,xend=xend,yend=yend),colour=BLUE,linewidth=.6,linetype="22",arrow=arrow_head)+
  annotate("label",x=9.1,y=3.78,label="冻结感官接口",family=FONT,size=10.5/PT,
           label.padding=unit(.10,"lines"),label.r=unit(0,"pt"),fill="white",colour=BLUE,linewidth=0)+
  annotate("text",x=9.725,y=2.63,label="独立归因（不改排序）",family=FONT,size=9.5/PT,colour=BLUE)+
  annotate("text",x=9.725,y=1.03,label="感官画像",family=FONT,size=9.5/PT,colour=BLUE)+
  annotate("label",x=8.9,y=.48,label="感官代理检验",family=FONT,size=10.5/PT,
           label.padding=unit(.10,"lines"),label.r=unit(0,"pt"),fill="white",colour=BLUE,linewidth=0)+
  annotate("segment",x=1.55,xend=2.12,y=.06,yend=.06,linewidth=.55,colour="#303030",arrow=arrow(length=unit(1.7,"mm"),type="closed"))+
  annotate("text",x=2.3,y=.06,label="数据与求解流程",hjust=0,family=FONT,size=10/PT)+
  annotate("segment",x=6.5,xend=7.07,y=.06,yend=.06,linewidth=.6,linetype="22",colour=BLUE,arrow=arrow(length=unit(1.7,"mm"),type="closed"))+
  annotate("text",x=7.25,y=.06,label="冻结预测器的调用",hjust=0,family=FONT,size=10/PT)+
  coord_cartesian(xlim=c(-.30,12.08),ylim=c(-.15,6.58),expand=FALSE,clip="off")+
  theme_mcup(base_size=12,legend="none")+
  theme(axis.title=element_blank(),axis.text=element_blank(),axis.ticks=element_blank(),
        panel.border=element_blank(),panel.grid=element_blank(),plot.margin=margin(8,10,7,6))
save_paper(p,file.path(OUT,"技术路线图_待审.png"),width=7.4,height=4.6,dpi=300)
ggsave(file.path(OUT,"技术路线图_待审.pdf"),p,width=7.4,height=4.6,device=cairo_pdf,family=FONT,bg="white")
ggsave(file.path(OUT,"技术路线图_待审.svg"),p,width=7.4,height=4.6,bg="white")
write.csv(nodes,file.path(OUT,"节点.csv"),row.names=FALSE)
write.csv(rbind(transform(main_edges,type="数据与求解"),transform(data_edges,type="共同数据"),transform(frozen_lines,type="冻结调用总线"),transform(frozen_arrows,type="冻结调用箭头")),file.path(OUT,"连线.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(OUT,"R_session_info.txt"))
cat("Technical route draft exported; not embedded.\n")

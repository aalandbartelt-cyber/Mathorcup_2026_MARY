suppressPackageStartupMessages({
  library(ggplot2);library(readr);library(dplyr);library(patchwork);library(grid)
})
root <- "/home/mary/mathorcup"
source(file.path(root,"scripts/plot_style.R"))
out <- file.path(root,"04_图表/Q2_关键化合物/待审_20261027")
dat <- file.path(out,"source_data")
BLUE <- "#1F5A94";ORANGE <- "#D4A35B";GRAY <- "#8A929C"
PT <- 72.27/25.4

paper_theme <- function() {
  theme_mcup(base_size=11.5,legend="none") +
    theme(text=element_text(family=FONT_PAPER,colour="#252525"),
          axis.text=element_text(size=11,colour="#252525"),
          axis.title=element_text(size=11,face="bold"),
          axis.title.x=element_text(margin=margin(t=8)),
          panel.grid=element_blank(),
          panel.border=element_rect(colour="#222222",fill=NA,linewidth=.45),
          plot.subtitle=element_text(size=11.5,face="bold",hjust=0,margin=margin(b=8)),
          plot.margin=margin(9,12,8,7),
          legend.position="bottom",legend.direction="horizontal",
          legend.text=element_text(size=10.5),legend.key.width=unit(18,"pt"),
          legend.key.height=unit(12,"pt"),legend.margin=margin(t=7),
          legend.box.margin=margin(t=1))
}

slots <- read_csv(file.path(dat,"图5a_影子字段入选槽.csv"),show_col_types=FALSE) |>
  mutate(字段类型=factor(字段类型,levels=c("影子字段","真实字段")))
p_a <- ggplot(slots,aes(每记录平均入选槽数,字段类型,fill=字段类型)) +
  geom_col(width=.28,colour="#444444",linewidth=.4) +
  geom_text(aes(label=sprintf("%.3f",每记录平均入选槽数)),hjust=-.27,
            family=FONT_PAPER,size=11/PT) +
  scale_fill_manual(values=c("真实字段"=BLUE,"影子字段"=ORANGE),guide="none") +
  scale_x_continuous(limits=c(0,3),breaks=seq(0,3,.5),expand=c(0,0)) +
  scale_y_discrete(expand=expansion(add=.8)) +
  labs(subtitle="（a）真实字段与影子字段",x="每条记录平均入选槽数",y=NULL) +
  paper_theme() + theme(legend.position="none",plot.margin=margin(9,18,8,7))

anchor <- read_csv(file.path(dat,"图5b_直接锚点.csv"),show_col_types=FALSE) |>
  mutate(直接锚点=factor(直接锚点,levels=rev(直接锚点)),
         最终前三保留=factor(最终前三保留,levels=c("保留","未保留")))
p_b <- ggplot(anchor,aes(bootstrap名次,直接锚点)) +
  geom_vline(xintercept=3,linetype="22",colour="#777777",linewidth=.45) +
  geom_segment(aes(x=1,xend=bootstrap名次,yend=直接锚点),colour="#C6CDD4",linewidth=.65) +
  geom_point(aes(colour=最终前三保留,shape=最终前三保留),size=2.9,stroke=.9,fill="white") +
  geom_text(aes(x=bootstrap名次*1.17,label=bootstrap名次),hjust=0,
            family=FONT_PAPER,size=11/PT) +
  scale_x_log10(limits=c(.8,150),breaks=c(1,3,10,30,100),labels=c("1","3","10","30","100"),expand=c(0,0)) +
  scale_y_discrete(expand=expansion(add=.6)) +
  scale_colour_manual(values=c("保留"=BLUE,"未保留"=ORANGE),
                      labels=c("最终融合前三保留","未进入最终融合前三"),name=NULL) +
  scale_shape_manual(values=c("保留"=16,"未保留"=2),
                     labels=c("最终融合前三保留","未进入最终融合前三"),name=NULL) +
  guides(colour=guide_legend(nrow=1,override.aes=list(size=2.7)),shape=guide_legend(nrow=1)) +
  labs(subtitle="（b）直接锚点恢复",x="bootstrap 名次（对数轴；虚线为第3名）",y=NULL) +
  paper_theme()

figure <- (p_a | p_b) + plot_layout(widths=c(1,1.55))
save_paper(figure,file.path(out,"图5_负对照与直接锚点.png"),width=9.2,height=4.6,dpi=300)
ggsave(file.path(out,"图5_负对照与直接锚点.pdf"),figure,width=9.2,height=4.6,units="in",device=cairo_pdf,bg="white")
if(requireNamespace("svglite",quietly=TRUE)) {
  ggsave(file.path(out,"图5_负对照与直接锚点.svg"),figure,width=9.2,height=4.6,units="in",device=svglite::svglite,bg="white")
}
cat("figure5 written\n")

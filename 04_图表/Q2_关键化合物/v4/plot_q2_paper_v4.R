suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(patchwork); library(scales); library(grid)
})
source("scripts/plot_style.R")
figure_dir <- "04_图表/Q2_关键化合物/v4"
source_dir <- file.path(figure_dir, "source_data")
FONT <- FONT_PAPER  # Q1 定稿相同的宋体与拉丁字体风格
BLUE <- "#1F5A94"; GREEN <- "#67A995"; ORANGE <- "#D4A35B"; GRAY <- "#8A929C"
PT <- 72.27 / 25.4
qa_layout <- list()

paper_theme <- function(grid_axis = "x") {
  theme_mcup(base_size = 11, legend = "none") + theme(
    text = element_text(family = FONT, colour = "#252525"),
    axis.text = element_text(family = FONT, size = 10.5, colour = "#252525"),
    axis.title = element_text(family = FONT, size = 10.5, face = "bold"),
    axis.title.x = element_text(margin = margin(t = 7)),
    axis.title.y = element_text(margin = margin(r = 7)),
    axis.ticks.length = unit(2, "pt"),
    panel.grid.major.x = if (grid_axis == "x") element_line(colour = "#E7EAEC", linewidth = 0.25) else element_blank(),
    panel.grid.major.y = if (grid_axis == "y") element_line(colour = "#E7EAEC", linewidth = 0.25) else element_blank(),
    panel.border = element_rect(colour = "#222222", fill = NA, linewidth = 0.45),
    plot.title = element_blank(),
    plot.subtitle = element_text(family = FONT, face = "bold", size = 10.5, hjust = 0, margin = margin(b = 5)),
    plot.margin = margin(8, 10, 5, 8),
    legend.text = element_text(family = FONT, size = 10.5),
    legend.title = element_text(family = FONT, size = 10.5),
    legend.key.width = unit(22, "pt"), legend.key.height = unit(12, "pt"),
    legend.margin = margin(t = 4), legend.spacing.x = unit(5, "pt")
  )
}
save_v4 <- function(p, name, width, height) {
  save_paper(p, file.path(figure_dir, name), width, height, dpi = 300)
}

# Same horizontal geometry on both panels; no external right-hand legend.
graph <- read_csv(file.path(source_dir, "图1_阈值与网络结构.csv"), show_col_types = FALSE)
edge <- graph |> pivot_longer(c(正相关边, 负相关边), names_to = "类型", values_to = "数量")
p1a <- ggplot(edge, aes(相关阈值, 数量, colour = 类型, linetype = 类型, shape = 类型)) +
  geom_vline(xintercept = .6, colour = "#707070", linewidth = .5, linetype = "22") +
  geom_line(linewidth = .65) + geom_point(size = 1.9) +
  annotate("text", x = .615, y = 5000, label = "0.60", hjust = 0, family = FONT, size = 11/PT, fontface = "bold") +
  scale_colour_manual(values = c(正相关边 = BLUE, 负相关边 = ORANGE)) +
  scale_linetype_manual(values = c(正相关边 = "solid", 负相关边 = "22")) +
  scale_shape_manual(values = c(正相关边 = 16, 负相关边 = 17)) +
  scale_x_continuous(breaks = c(.2,.4,.6,.8), limits = c(.18,.82), expand = c(0,0)) +
  scale_y_continuous(trans = pseudo_log_trans(sigma = .2), breaks = c(0,1,10,100,1000,10000),
                     labels = label_number(big.mark = ","), limits = c(0,13000), expand = expansion(mult=c(.025,.01))) +
  labs(subtitle = "（a）正、负相关边", x = "相关阈值 |ρ|", y = "边数") +
  paper_theme("y") + theme(legend.position = "bottom", legend.title = element_blank())
struct <- graph |> select(相关阈值, 最大连通分量节点数, 孤立节点) |>
  rename(最大分量 = 最大连通分量节点数) |> pivot_longer(-相关阈值, names_to = "类型", values_to = "数量")
p1b <- ggplot(struct, aes(相关阈值, 数量, colour = 类型, linetype = 类型, shape = 类型)) +
  geom_vline(xintercept = .6, colour = "#707070", linewidth = .5, linetype = "22") +
  geom_line(linewidth = .65) + geom_point(size = 1.9) +
  scale_colour_manual(values = c(最大分量 = BLUE, 孤立节点 = ORANGE)) +
  scale_linetype_manual(values = c(最大分量 = "solid", 孤立节点 = "22")) +
  scale_shape_manual(values = c(最大分量 = 16, 孤立节点 = 17)) +
  scale_x_continuous(breaks = c(.2,.4,.6,.8), limits = c(.18,.82), expand = c(0,0)) +
  scale_y_continuous(breaks = seq(0,250,50), limits = c(0,250), expand = expansion(mult=c(.025,.01))) +
  labs(subtitle = "（b）连通结构", x = "相关阈值 |ρ|", y = "节点数") +
  paper_theme("y") + theme(legend.position = "bottom", legend.title = element_blank())
fig1 <- p1a + p1b + plot_layout(ncol=2, widths=c(1,1))
save_v4(fig1, "图1_共表达图阈值敏感性_v4.png", 7.4, 4.05)

model <- read_csv(file.path(source_dir, "图2a_模型性能.csv"), show_col_types = FALSE) |>
  mutate(短名 = recode(方法, "Spearman Top-3"="Spearman", "有符号图ElasticNet"="图ElasticNet",
                      "ExtraTrees条件归因"="ExtraTrees", "感官组收缩NNLS"="NNLS"),
         短名=factor(短名, levels=c("Spearman","PLS/VIP","ElasticNet","图ElasticNet","ExtraTrees","NNLS")),
         字重=if_else(短名=="NNLS","bold","plain"))
colors <- c("Spearman"=GRAY,"PLS/VIP"=GRAY,"ElasticNet"=GREEN,"图ElasticNet"=GREEN,"ExtraTrees"="#88A9D3","NNLS"=BLUE)
metric_panel <- function(mean_col, sd_col, title, lim, ticks, number_x) {
  ggplot(model, aes(x=.data[[mean_col]], y=短名)) +
    geom_errorbar(aes(xmin=.data[[mean_col]]-.data[[sd_col]], xmax=.data[[mean_col]]+.data[[sd_col]],colour=短名),
                  orientation="y",width=.18,linewidth=.45) +
    geom_point(aes(colour=短名),size=2.15) +
    geom_text(aes(x=number_x,label=sprintf("%.3f",.data[[mean_col]]),fontface=字重),
              family=FONT,colour="#252525",size=10.5/PT,hjust=1) +
    scale_colour_manual(values=colors,guide="none") +
    scale_x_continuous(limits=lim,breaks=ticks,expand=c(0,0)) +
    scale_y_discrete(expand=expansion(add=.65)) +
    labs(subtitle=title,x=NULL,y=NULL) + paper_theme("x")
}
p2a <- metric_panel("R2均值","R2标准差","（a）解释度",c(.075,.325),c(.1,.15,.2,.25),.319) + labs(x="宏平均 R²")
p2b <- metric_panel("MAE均值","MAE标准差","（b）绝对误差",c(.325,.41),c(.33,.35,.37,.39),.408) + labs(x="宏平均 MAE（z 分数）")
p2c <- metric_panel("Spearman均值","Spearman标准差","（c）排序一致性",c(.27,.485),c(.3,.35,.4,.45),.482) + labs(x="宏平均 Spearman")
agreement <- read_csv(file.path(source_dir,"图2b_四路一致性.csv"),show_col_types=FALSE) |>
  mutate(证据路径=factor(证据路径,levels=rev(c("总关联","分层自助","风格校正","逐风格删除"))))
p2d <- ggplot(agreement,aes(平均Jaccard,证据路径)) +
  geom_col(fill="#88A9D3",colour="#4F719B",linewidth=.38,width=.55) +
  geom_text(aes(label=sprintf("%.3f",平均Jaccard)),nudge_x=.025,hjust=0,family=FONT,size=10.5/PT) +
  scale_x_continuous(limits=c(0,.85),breaks=c(0,.2,.4,.6,.8),expand=c(0,0)) +
  scale_y_discrete(expand=expansion(add=.6)) +
  labs(subtitle="（d）四路证据一致性",x="平均 Jaccard",y=NULL) + paper_theme("x")
fig2 <- wrap_plots(p2a,p2b,p2c,p2d,ncol=2,widths=c(1,1),heights=c(1,1))
save_v4(fig2,"图2_模型性能与四路一致性_v4.png",7.4,4.6)

# Fixed bins summarize all 150 rows. Counts are labels; within-grade fractions
# define the common colour scale, so unequal grade sizes are explicit.
stability <- read_csv(file.path(source_dir,"图3_候选稳定性.csv"),show_col_types=FALSE)
ns <- stability |> count(化学证据等级)
bin_labels <- c("0–0.2","0.2–0.4","0.4–0.6","0.6–0.8","0.8–1")
stable_bins <- stability |> select(化学证据等级,自助入选频率,留一风格保留率) |>
  pivot_longer(-化学证据等级,names_to="指标",values_to="稳定度") |>
  mutate(分箱=cut(稳定度,breaks=c(0,.2,.4,.6,.8,1.00000001),right=FALSE,labels=bin_labels)) |>
  count(指标,化学证据等级,分箱,.drop=FALSE) |>
  complete(指标,化学证据等级,分箱,fill=list(n=0)) |>
  group_by(指标,化学证据等级) |> mutate(组内占比=n/sum(n)) |> ungroup()
write_csv(stable_bins,file.path(source_dir,"图3_稳定性分箱汇总.csv"))
heat_panel <- function(metric,title,xlab) {
  dat <- stable_bins |> filter(指标==metric) |> mutate(箱中点=.1+.2*(as.integer(分箱)-1))
  ggplot(dat,aes(箱中点,化学证据等级,fill=组内占比)) +
    geom_tile(width=.2,colour="white",linewidth=.45) +
    geom_text(aes(label=n),family=FONT,size=11/PT,colour="black") +
    scale_colour_manual(values=c("FALSE"="#202020","TRUE"="white"),guide="none") +
    scale_fill_gradient(low="white",high="#88A9D3",limits=c(0,1),labels=label_percent(accuracy=1),
                        name="等级内占比",breaks=c(0,.5,1),
                        guide=guide_colourbar(direction="horizontal",barwidth=unit(52,"mm"),barheight=unit(3,"mm"),title.position="top")) +
    scale_y_discrete(limits=rev(c("A","B","C","D")),
      labels=setNames(paste0(ns$化学证据等级,"  (n=",ns$n,")"),ns$化学证据等级),expand=c(0,0)) +
    scale_x_continuous(breaks=seq(0,1,.2),limits=c(0,1),expand=c(0,0)) +
    labs(subtitle=title,x=xlab,y=NULL) + paper_theme("none") +
    theme(legend.position="bottom",axis.ticks=element_blank(),axis.text.x=element_text(family=FONT,size=10.5),
          plot.margin=margin(8,9,4,6))
}
p3a <- heat_panel("自助入选频率","（a）样本扰动","分层自助入选频率")
p3b <- heat_panel("留一风格保留率","（b）风格扰动","留一风格保留率")
fig3 <- (p3a | p3b) + plot_layout(guides="collect") & theme(legend.position="bottom")
save_v4(fig3,"图3_候选稳定性与证据等级_v4.png",7.4,4.05)

# One explicit row coordinate per relation: every name, rank, and OAV stays bound
# to the same row even for zero and unavailable measurements.
grade <- read_csv(file.path(source_dir,"图4_A级关系名次与OAV.csv"),show_col_types=FALSE) |>
  mutate(行=9-row_number(),
    感官=c("指甲油香气","指甲油风味","4VG","酸度","酒精感","酒体","碳酸感","乳酸感"),
    化合物=c("乙酸乙酯","乙酸乙酯","4VG","乳酸","乙醇","乙醇","二氧化碳","乳酸"),
    OAV显示=case_when(OAV状态=="N/A"~"N/A",OAV状态=="零值"~sprintf("0 [0, %.2f]",OAV上四分位),
      TRUE~sprintf("%.2f [%.2f, %.2f]",OAV中位数,OAV下四分位,OAV上四分位)),
    字重=if_else(!is.na(OAV中位数)&OAV中位数>=1,"bold","plain"))
rank <- grade |> select(关系,行,总关联,分层自助,风格校正,逐风格删除) |>
  pivot_longer(c(总关联,分层自助,风格校正,逐风格删除),names_to="路径",values_to="名次") |>
  mutate(列=match(路径,c("总关联","分层自助","风格校正","逐风格删除")),横坐标=2.92+(列-1)*.70)
relation_labels <- setNames(paste0(grade$感官,"—",grade$化合物),grade$行)
p4a <- ggplot(rank,aes(列,行,fill=名次)) +
  geom_tile(width=.92,height=.88,colour="white",linewidth=.5) +
  geom_text(aes(label=名次),family=FONT,size=11/PT,colour="black") +
  scale_fill_gradient(low="#88A9D3",high="#F4F7FB",limits=c(1,5),breaks=1:5,name="名次",
    guide=guide_colourbar(order=1,direction="horizontal",barwidth=unit(32,"mm"),
      barheight=unit(3,"mm"),title.position="top")) +
  scale_x_continuous(breaks=1:4,labels=c("总关联","分层\n自助","风格\n校正","逐风格\n删除"),
                     limits=c(.5,4.5),expand=c(0,0)) +
  scale_y_continuous(breaks=1:8,labels=relation_labels[as.character(1:8)],limits=c(.5,8.5),expand=c(0,0)) +
  labs(subtitle="（a）四路名次",x=NULL,y=NULL) + paper_theme("none") +
  theme(axis.ticks=element_blank(),plot.margin=margin(8,12,5,6))
p4b <- ggplot(grade,aes(OAV中位数,行)) +
  geom_vline(aes(xintercept=1,linetype="OAV＝1"),colour="#707070",linewidth=.45) +
  geom_errorbar(data=filter(grade,OAV状态!="N/A"),
    aes(xmin=OAV下四分位,xmax=OAV上四分位,colour="四分位区间"),orientation="y",width=.16,linewidth=.65,key_glyph="path") +
  geom_point(data=filter(grade,OAV状态!="N/A"),aes(colour="中位数"),size=2.3) +
  geom_label(data=filter(grade,OAV状态=="N/A"),aes(x=.18,label="N/A"),
    family=FONT,size=10.5/PT,colour="#555555",fill="white",linewidth=0,label.padding=unit(.08,"lines")) +
  scale_x_continuous(trans=pseudo_log_trans(sigma=.12),breaks=c(0,.1,.5,1,2,5),
    labels=c("0","0.1","0.5","1","2","5"),limits=c(-.02,6.5),expand=c(0,0)) +
  scale_y_continuous(breaks=1:8,labels=NULL,limits=c(.5,8.5),expand=c(0,0)) +
  scale_colour_manual(values=c("中位数"=BLUE,"四分位区间"=GREEN),breaks=c("中位数","四分位区间"),name=NULL,
    guide=guide_legend(order=2,nrow=1,override.aes=list(shape=c(16,NA),linetype=c(0,1)))) +
  scale_linetype_manual(values=c("OAV＝1"="22"),name=NULL,guide=guide_legend(order=3)) +
  labs(subtitle="（b）气味活性与离散程度",x="OAV（无量纲）",y=NULL) + paper_theme("y") +
  theme(axis.ticks.y=element_blank(),plot.margin=margin(8,10,5,7))
fig4 <- (p4a + p4b + plot_layout(ncol=2,widths=c(1,1.18),guides="collect")) &
  theme(legend.position="bottom",legend.box="horizontal",legend.box.just="center")
save_v4(fig4,"图4_A级关系四路排名与OAV_v4.png",8.2,4.6)
write_csv(grade |> select(关系,行,感官,化合物,总关联,分层自助,风格校正,逐风格删除,OAV显示),
          file.path(source_dir,"图4_逐行排版核验.csv"))

checklist <- tibble::tribble(
  ~图号,~路径,~图题,~说明,
  "图1","04_图表/Q2_关键化合物/v4/图1_共表达图阈值敏感性_v4.png","化合物共表达网络的阈值选择与结构敏感性","两个等宽面板：正负边数与连通结构；全量描述图，阈值0.60。",
  "图2","04_图表/Q2_关键化合物/v4/图2_模型性能与四路一致性_v4.png","候选模型的折外性能与稳健排序的四路一致性","2×2：R²、MAE、Spearman与四路Jaccard；预测指标为十次重复均值±SD。",
  "图3","04_图表/Q2_关键化合物/v4/图3_候选稳定性与证据等级_v4.png","不同化学证据等级的样本与风格扰动稳定性","全部150条关系按稳定性分箱；格内为关系数，底色为等级内比例。",
  "图4","04_图表/Q2_关键化合物/v4/图4_A级关系四路排名与OAV_v4.png","A级化合物—感官关系的四路名次与气味活性值","四路名次热图与OAV点区间等行对齐；OAV采用含零伪对数轴，虚线为1；N/A与0分别保留。"
)
write_csv(checklist,file.path(figure_dir,"Q2_图清单_v4.csv"))

# Geometry audit of rendered text. This only renders figures from frozen CSVs.
source("04_图表/Q2_关键化合物/v4/plot_q2_paper_v4.R")
out <- "outputs/qa/q2_v4"
dir.create(out, recursive=TRUE, showWarnings=FALSE)
collect_text <- function(plot, number, width, height) {
  ragg::agg_png(file.path(out,paste0("layout_",number,".png")),width=width,height=height,units="in",res=300)
  print(plot); grid.force()
  ls <- grid.ls(print=FALSE,viewports=TRUE)
  rows <- list()
  for(i in grep("text",ls$name)) {
    g <- grid.get(paste0(ls$gPath[i],"::",ls$name[i]))
    if(!inherits(g,"text") || length(g$label)==0) next
    upViewport(0)
    parts <- strsplit(ls$vpPath[i],"::",fixed=TRUE)[[1]]
    parts <- parts[parts!="ROOT"]
    if(length(parts)) downViewport(do.call(vpPath,as.list(parts)))
    for(j in seq_along(g$label)) {
      lab <- as.character(g$label[j]); if(!nzchar(lab)) next
      pick <- function(v) v[1L+(j-1L)%%length(v)]
      gp <- do.call(gpar,lapply(g$gp,pick))
      hjust <- pick(g$hjust); vjust <- pick(g$vjust)
      if(is.null(hjust)||is.null(vjust)) {
        just <- grid:::valid.just(g$just)
        hjust <- grid:::resolveHJust(g$just,g$hjust)
        vjust <- grid:::resolveVJust(g$just,g$vjust)
      }
      one <- textGrob(lab,gp=gp)
      w <- convertWidth(grobWidth(one),"in",valueOnly=TRUE)
      h <- convertHeight(grobHeight(one),"in",valueOnly=TRUE)
      xy <- deviceLoc(pick(g$x),pick(g$y),valueOnly=TRUE)
      corners <- expand.grid(x=c(-hjust*w,(1-hjust)*w),y=c(-vjust*h,(1-vjust)*h))
      theta <- pick(g$rot)*pi/180
      xx <- xy$x+corners$x*cos(theta)-corners$y*sin(theta)
      yy <- xy$y+corners$x*sin(theta)+corners$y*cos(theta)
      rows[[length(rows)+1]] <- data.frame(figure=number,label=lab,
        grob=ls$name[i],viewport=ls$vpPath[i],family=gp$fontfamily,size=gp$fontsize,
        xmin=min(xx),xmax=max(xx),ymin=min(yy),ymax=max(yy),canvas_width=width,canvas_height=height)
    }
  }
  dev.off()
  bind_rows(rows)
}
boxes <- bind_rows(collect_text(fig1,1,7.4,4.05),collect_text(fig2,2,7.4,4.6),
                  collect_text(fig3,3,7.4,4.05),collect_text(fig4,4,8.2,4.6))
write_csv(boxes,file.path(out,"text_boxes.csv"))
overlap <- list()
for(f in 1:4) {
  dat <- boxes |> filter(figure==f)
  for(i in seq_len(nrow(dat)-1)) for(j in seq.int(i+1,nrow(dat))) {
    dx <- min(dat$xmax[i],dat$xmax[j])-max(dat$xmin[i],dat$xmin[j])
    dy <- min(dat$ymax[i],dat$ymax[j])-max(dat$ymin[i],dat$ymin[j])
    if(dx>0.006 && dy>0.006) overlap[[length(overlap)+1]] <- data.frame(
      figure=f,label_a=dat$label[i],label_b=dat$label[j],dx=dx,dy=dy)
  }
}
if(length(overlap)) {
  write_csv(bind_rows(overlap),file.path(out,"text_overlaps.csv"))
} else {
  write_csv(data.frame(figure=integer(),label_a=character(),label_b=character(),dx=numeric(),dy=numeric()),file.path(out,"text_overlaps.csv"))
}
outside <- boxes |> filter(xmin < -0.01 | ymin < -0.01 | xmax > canvas_width+.01 | ymax > canvas_height+.01)
write_csv(outside,file.path(out,"text_outside_canvas.csv"))
cat("Text boxes:",nrow(boxes),"overlaps:",length(overlap),"outside:",nrow(outside),"\n")

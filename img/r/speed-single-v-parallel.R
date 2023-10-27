library(ggplot2)
require(scales)
require(dplyr)

data=read.csv("/Users/sebas/Desktop/results.csv",header = TRUE ,sep = "\t")


#vergleichsfunktionen
#n <- seq(from=0.1,to=1250,by=0.1)
#f <- function(a){
#  a*a
#}
#g <- function(a){
#  a
#}
#t<-c(f(n),g(n))
#type<-c(rep("x*x",times=length(n)), 
#        rep("x",times=length(n)))
#density<-c(rep("n",times=length(n)), 
#           rep("1",times=length(n)))
#n<-c(n,n)
#d = data.frame(n,t,type,density)





p <- ggplot(data,aes(x=matrix.size,y=GMAC.s,color=threads,group=threads))+
  geom_point(aes(shape = threads)) + 
  #geom_path(aes(group = type))+
  geom_smooth(size=.8)+ # argument se=F schaltet konvidenzintervall aus
  theme_bw() +
  labs(color = "Core count",group="Core count",linetype="Core count",shape="Core count")+
  theme(
    legend.position = c(.97, .03),
    legend.justification = c("right", "bottom"),
    legend.box.just = "right",
    legend.margin = margin(6, 6, 6, 6)
  )+
  scale_color_hue(labels = c("6 (12 threads)", "1 (1 thread)"))+
  scale_shape(labels = c("6 (12 threads)", "1 (1 thread)"), solid = TRUE)+
  ylab("GMAC/s") +
  xlab("Matrix Size")+
  scale_y_log10(minor_breaks = rep(1:9, 21)*(10^rep(-10:10, each=9)))+
  scale_x_continuous(breaks = seq(150,750,by=50))+
  ggtitle(label="Matrix-Matrix Multiplication Speed")
  
  #+
  #vergleichsfunktionen
  #geom_line(data = d, aes(x=n, y=t, group=density, colour=density)
  #          ,show_guide = FALSE)


p


#ggsave("cg-dense-vs-sparse.png", units="in", width=5, height=4, dpi=300)

library(kernlab)
library(npreg)
library(gss)
library(ggplot2)
library(readxl)

setwd('.ARMDDesign/')
data <- read_xlsx('ARMADesign_dri50_epi50_sim50_num6_p2q2.xlsx')

delta <- 3
nn <- 50 # number of drivers
n_sim <- 50 # can increase to 50
p_s <- 0

method_list <- unique(data$Method)
mse_list <- c()
method_rank <- c('ATE_AT','ATE_UR', 'ATE_AD','ATE_greedy','ATE_Switch5','ATE_TMDP','ATE_NMDP','ATE_Markov','ATE_MDP') # num_sim 50

for (i in 1:length(method_rank)){
  print(method_rank[i])
  mse_list <- c(mse_list, (data$ATE_estimator[data$Method==method_rank[i]] - 2.24)^2)
}

method_plotlist <- c('AT','UR','AD','Greedy','Switch','TMDP','NMDP','CO','RL') # num_sim 50
ddd_05 <- data.frame(Design=rep(method_plotlist, each=n_sim),
                     MSE =mse_list)

ddd_05$Design <- factor(ddd_05$Design, levels=method_plotlist)


p <- ggplot(ddd_05, aes(x=Design, y=MSE, fill=Design)) + 
  geom_violin()
p
data_summary <- function(x) {
  m <- mean(x)
  ymin <- m-sd(x)
  ymax <- m+sd(x)
  return(c(y=m,ymin=ymin,ymax=ymax))
}
p + stat_summary(fun.data=data_summary, color='black') +
  # xlab(NULL) +
  xlab('Treatment Allocation Strategy') +
  coord_cartesian(ylim = c(0, 20)) +
  ggtitle('Average Treatment Effect (ATE)') +
  # geom_boxplot(width=0.1, fill="white")+
    theme(
      plot.title = element_text(size = 40,hjust = 0.5), #face=bold, hjust to center the title
      legend.title = element_text(size = 30),
      legend.key = element_blank(),
      legend.key.width = unit(1, "cm"),
      legend.position = "none",#"bottom",
      legend.text = element_text(size = 30),
      axis.title.x = element_text(size = 40),
      axis.text.x  = element_text(size = 20, angle = 0),
      axis.title.y = element_text(size = 40),
      axis.text.y  = element_text(size = 20)
    )



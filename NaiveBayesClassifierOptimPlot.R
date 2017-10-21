#!/usr/bin/env Rscript

library(ggplot2)
library(reshape2)
library(tidyverse)
library(cowplot)
library(dplyr)

x = read.table("NaiveBayesClassifierOptim.csv", header = T, sep = "\t")

x %<>% dplyr::rename("Accuracy" = test_accuracy,
                     "Precision" = test_precision,
                     "Recall" = test_recall,
                     "Average" = test_average)

xx = data.frame("idx" = 1:nrow(x), x[4:7]) #%>% melt(key = "metric", value = "score")
xx = cbind(idx = xx[, 1], xx[, -1] %>% melt) %>% dplyr::select(idx, "metric" = variable, "score" = value)
yy = data.frame("idx" = 1:nrow(x), x[1:3])

z = yy %>% left_join(xx) %>% dplyr::select(-idx)
z$metric = factor(z$metric, levels = c("Accuracy", "Precision", "Recall", "Average"))
z$lam = paste0("lambda=", z$lam)
z$c = paste0("c=", z$c)
z$n = z$n 

print("stil lgood")

gg = ggplot(data = z,
            mapping = aes(x = n,
                          y = score,
                          color = c))
gg = gg + geom_line()
gg = gg + facet_grid(metric ~ lam)
gg = gg + scale_y_continuous(limits = c(.8, 1))
gg = gg + scale_x_continuous(breaks = 1:3)
gg = gg + ylab("Score")
gg = gg + xlab("n")
#gg = gg + geom_segment(mapping = aes(x = 1, xend = 3,
#                                     y = max_score, yend = max_score),
#                       data = z %>% group_by(metric, lam, x) %>% summarize(max_score = max(score)), linetype="dotted")
#gg = gg + xlab(expression(lambda))

png("NaiveBayesClassifierOptimPlot.png", width = 7.5, height = 5)
print(gg)
dev.off()

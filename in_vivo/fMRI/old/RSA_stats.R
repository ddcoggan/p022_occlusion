# Title     : TODO
# Objective : TODO
# Created by: dave
# Created on: 3/22/22

# p008 exp2
# reads in timeseries.csv and plots response profiles to 6 different conditions across various brain regions
# produces text file with relevant stats

packages <- c('ggplot2', 'plyr', 'multcomp', 'broom', 'lsr', 'ez', 'plotrix', 'nlme', 'grid', 'Dict', 'stringr')
lapply(packages, require, character.only = TRUE)

sig_code <- function(x){
  y <- c()
  y[x > .05] = ''
  y[x <= .05] = '*'
  return(y)
}

# results directory
for (localizer in c('allConds', 'FHOloc')){
    for (norm in c('allConds', 'occluder')){
      analysisDir <- sprintf('fMRI/v3/analysis/results/MVPA/splitHalf/topUp/noB0/doubleGamma/%s/%s', localizer, norm)
      data <- data.frame(read.csv(sprintf('%s/MVPAcontrasts.csv', analysisDir), header = T, sep=',', strip.white=T)) # get data
      outFile = file.path(analysisDir, "ANOVA.txt")

      # analysis 1: 4x2x2x2 anova (region, attn, occPos, exemplar) for each of the occPositionExemplar analysis
      theseData <- data[data$contrast == 'occPositionExemplar',]
      theseData$exemplar = substr(theseData$level, 1, 4)
      theseData$occluder = substr(theseData$level, 9, 12)
      theseData <- droplevels(theseData[,c(2,3,6,8,9)])
      model <- ezANOVA(data = theseData, dv = mean, wid = subject, within = .(region,exemplar,occluder), detailed = TRUE)
      model$ANOVA$pes = model$ANOVA$SSn/(model$ANOVA$SSn + model$ANOVA$SSd) # add partial eta squared
      sink(outFile, append=F)
      cat('### ANOVA occPositionExemplar###\n')
      cat(capture.output(model), sep = '\n')
      sink()

      # analysis 2: 4x2x2x2 anova (region, attn, occPres, exemplar) for each of the occVunocc analysis
      theseData <- data[data$contrast == 'occVunocc',]
      theseData$exemplar = substr(theseData$level, 1, 4)
      theseData$occluder = substr(theseData$level, 9, 12)
      theseData <- droplevels(theseData[,c(2,3,6,8,9)])
      model <- ezANOVA(data = theseData, dv = mean, wid = subject, within = .(region,exemplar,occluder), detailed = TRUE)
      model$ANOVA$pes = model$ANOVA$SSn/(model$ANOVA$SSn + model$ANOVA$SSd) # add partial eta squared
      sink(outFile, append=T)
      cat('\n\n### ANOVA occVunocc###\n')
      cat(capture.output(model), sep = '\n')
      sink()
      
    }
}
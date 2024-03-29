# Se precisar carregar pacotes adicionais, siga os exemplos abaixo 
#install.packages("psych")

library("ggplot2")
#library("ggalt")
library("gridExtra")
library("plyr")
library("stringr")
library("forcats")
library("scales")
library("forcats")
library("ExpDes")
library("dplyr")
library("ExpDes.pt")
library(tidyr)
library("Metrics")
library(data.table)


options(scipen = 999)

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# BOXPLOT DO DESEMPENHO ENTRE TÉCNICAS
#
dados <- read.table('./dataset/results.csv',sep=',',header=TRUE)

metricas <- list("mAP50","mAP75","mAP","precision","recall","fscore","MAE","RMSE","r")
graficos <- list()
i <- 1

for (metrica in metricas) {

   print(metrica)
   TITULO = sprintf("Boxplot for %s",metrica)
   g <- ggplot(dados, aes_string(x="ml", y=metrica,fill="ml")) + 
   geom_boxplot()+
   scale_fill_brewer(palette="Purples")+
   labs(title=TITULO,x="Models", y = metrica)+
   theme(legend.position="none")+
   theme(plot.title = element_text(hjust = 0.5))
   
   graficos[[i]] <- g
   i = i + 1
}

g <- grid.arrange(grobs=graficos, ncol = 3)
ggsave(paste("./dataset/boxplot.png", sep=""),g, width = 12, height = 10)
print(g)



# -------------------------------------------------------------------
# -------------------------------------------------------------------
# CURVAS DE APRENDIZAGEM
#

nets <- levels(as.factor(dados$ml))
contaDobras <- dados[dados$ml == nets[1], ]

DOBRAS=nrow(contaDobras)

# ALERTA DE GAMBIARRA 
# Pega o total de épocas de dentro do arquivo experimento.py 
log <- readLines('experimento.py')
log <-log[grepl('EPOCAS=',log)]
logTable <- read.table(text = log,sep='=')
EPOCAS=logTable[1,2]

# ALERTA DE GAMBIARRA
# Pega dentro do arquivo de LOG os valores de perda no conjunto
# de validação (loss_val) para cada época da primeira dobra

logFile <- list.files(".", "log$", recursive=TRUE, full.names=TRUE, include.dirs=TRUE)
tail(file.info(logFile)$ctime) #mostrando a data de modificação deles
ultimoLog <- logFile[length(logFile)] #extraindo o ultimo elemento(ultima posição do vetor)
log <- readLines(ultimoLog)
epocas <-log[grepl('- mmdet - INFO - Epoch\\(',log)]
epocas <- gsub(".*loss_val: ([0-9.]+).*", "\\1", epocas)

epocasVal <- read.table(text = epocas,sep=' ')

colnames(epocasVal) <- c("loss")

folds <- sprintf("fold_%d",seq(1:DOBRAS))
epochs <- 1:EPOCAS

novasColunas <- tidyr::crossing(nets,folds,epochs)

# Testa se epocasVal tem mais linhas que novasColunas e remove
# as linhas adicionais 
if (nrow(epocasVal) > nrow(novasColunas)) {
   epocasVal <- epocasVal[1:nrow(novasColunas),]
} else if (nrow(epocasVal) < nrow(novasColunas)) {
   novasColunas <- novasColunas[1:nrow(epocasVal),]
}

dadosEpocas <- cbind(novasColunas,epocasVal)

# Troca o nome da coluna epocasVal por loss
colnames(dadosEpocas)[4] <- "loss"
write.csv(dadosEpocas,'./dataset/epocas.csv')

# Pegando apenas dados da primeira dobra 
filtrado <- dadosEpocas[dadosEpocas$folds == "fold_1",]
filtrado <- filtrado[filtrado$nets != "NA", ]
# Para evitar que valores muito grandes de perda na validação deixem
# o gráfico difícil de ver, os valores maiores que 5 são transformados
# em 5
filtrado$loss[filtrado$loss > 5] <- 5
TITULO = sprintf("Validation loss evolution during training")
g <- ggplot(filtrado, aes(x=epochs, y=loss, colour=nets, group=nets)) +
    geom_line() +
    ggtitle(TITULO)+
    theme(plot.title = element_text(hjust = 0.5))


ggsave(paste("./dataset/history.png", sep=""),g)
print(g)



# -------------------------------------------------------------------
# -------------------------------------------------------------------
# XY CONTAGEM MANUAL X AUTOMÁTICA - JUNTANDO TODAS AS DOBRAS
#
dadosContagem <- read.table('./dataset/counting.csv',sep=',',header=TRUE)

graficos <- list()
i <- 1

print(nets)
for (net in nets) {

   filtrado <- dadosContagem[dadosContagem$ml == net, ]

   RMSE = rmse(filtrado$groundtruth,filtrado$predicted)
   MAE = mae(filtrado$groundtruth,filtrado$predicted)
   MAPE = mape(filtrado$groundtruth,filtrado$predicted)
   R = cor(filtrado$groundtruth,filtrado$predicted,method = "pearson")
   TITULO = sprintf("%s RMSE=%.3f MAE=%.3f MAPE=%.3f r = %.3f",net,RMSE,MAE,MAPE,R)
   MAX <- max(filtrado$groundtruth, filtrado$predicted)
   
   g <- ggplot(filtrado, aes(x=groundtruth, y=predicted)) + 
        geom_point()+
        geom_smooth(method='lm')+
        labs(title=TITULO ,x="Measured", y = "Predicted")+ theme(plot.title = element_text(size = 10))+
        xlim(0,MAX)+
        ylim(0,MAX)

   print(g)
   graficos[[i]] <- g
   i = i + 1
}

g <- grid.arrange(grobs=graficos, ncol = 2)
ggsave(paste("./dataset/counting.png", sep=""),g, width = 10, height = 12)
print(g)


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# XY CONTAGEM MANUAL X AUTOMÁTICA - APENAS PARA A PRIMEIRA DOBRA
#
dadosContagem <- read.table('./dataset/counting.csv',sep=',',header=TRUE)

graficos <- list()
i <- 1

dadosContagem <- subset(dadosContagem,fold == 'fold_1')
print(nets)
for (net in nets) {
  
  filtrado <- dadosContagem[dadosContagem$ml == net, ]
  
  RMSE = rmse(filtrado$groundtruth,filtrado$predicted)
  MAE = mae(filtrado$groundtruth,filtrado$predicted)
  MAPE = mape(filtrado$groundtruth,filtrado$predicted)
  R = cor(filtrado$groundtruth,filtrado$predicted,method = "pearson")
  TITULO = sprintf("%s RMSE=%.3f MAE=%.3f MAPE=%.3f r = %.3f",net,RMSE,MAE,MAPE,R)
  MAX <- max(filtrado$groundtruth, filtrado$predicted)
  
  g <- ggplot(filtrado, aes(x=groundtruth, y=predicted)) + 
    geom_point()+
    geom_smooth(method='lm')+
    labs(title=TITULO ,x="Measured", y = "Predicted")+ theme(plot.title = element_text(size = 10))+
    xlim(0,MAX)+
    ylim(0,MAX)
  
  print(g)
  graficos[[i]] <- g
  i = i + 1
}

g <- grid.arrange(grobs=graficos, ncol = 2)
ggsave(paste("./dataset/counting_FOLD_1.png", sep=""),g, width = 10, height = 12)
print(g)

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Encontra as imagens com as maiores e menores diferenças na contagem
# -------------------------------------------------------------------
# -------------------------------------------------------------------

quantos <- 3

dadosContagem <- read.table('./dataset/counting.csv',sep=',',header=TRUE)

# Encontra as 3 maiores diferenças (coluna dif) para cada uma das
# técnicas (coluna ml), considerando o módulo da diferença (abs(dif))
# e ordenando em ordem decrescente
maiores <- dadosContagem %>% group_by(ml) %>% top_n(quantos, abs(dif))
# Copia as imagens (fileName) com as maiores diferenças de cada ténica (ml)
# para um diretório separado dentro da pasta dataset/prediction_X onde
# X é o nome da técnica (coluna ml sem os dois primeiros caracteres)
for (net in nets) {
  print(substr(net,3,100))
  print(maiores[maiores$ml == net, ]$fileName)
  pastaDaRede <- paste("./dataset/prediction_",substr(net,3,100),"/",sep="")
  novaPasta <- paste("./dataset/prediction_",substr(net,3,100),"/maiores/",sep="")
  dir.create(novaPasta, showWarnings = TRUE)
  file.copy(paste(pastaDaRede,maiores[maiores$ml == net, ]$fileName,sep=""),novaPasta,overwrite = TRUE)
}
# A mesma coisa para as 3 menores diferenças
menores <- dadosContagem %>% group_by(ml) %>% top_n(-quantos, abs(dif))
for (net in nets) {
  print(substr(net,3,100))
  print(menores[menores$ml == net, ]$fileName)
  pastaDaRede <- paste("./dataset/prediction_",substr(net,3,100),"/",sep="")
  novaPasta <- paste("./dataset/prediction_",substr(net,3,100),"/menores/",sep="")
  dir.create(novaPasta, showWarnings = TRUE)
  file.copy(paste(pastaDaRede,menores[menores$ml == net, ]$fileName,sep=""),novaPasta,overwrite = TRUE)
}











# -------------------------------------------------------------------
# -------------------------------------------------------------------
# HISTOGRAMA DA DISTRIBUIÇÃO DOS DADOS DO CONJUNTO DE TESTE
# (CONTAGENS MANUAIS)

g <- ggplot(filtrado, aes(x=groundtruth))+
   geom_histogram(color="darkblue", fill="lightblue")+
   xlab("Objects Countings")+
   ylab("Density")+
   ggtitle("Histogram for Ground Truth Countings (Test Set)")

ggsave(paste("./dataset/histogram.png", sep=""),g)
print(g)


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# GERA ESTATÍSTICAS PARA mAP, fscore e R
# Formatado para tabela em Latex


sink('./dataset/statistics.txt')


metricas <- list("mAP50","mAP75","mAP","precision","recall","fscore","MAE","RMSE","r")

dt <- data.table(dados)
cat("\n[ Estatísticas para mAP50]-----------------------------\n")
dt[,list(median=median(mAP50),IQR=IQR(mAP50),mean=mean(mAP50),sd=sd(mAP50)),by=ml]
cat("\n[ Estatísticas para mAP75]-----------------------------\n")
dt[,list(median=median(mAP75),IQR=IQR(mAP75),mean=mean(mAP75),sd=sd(mAP75)),by=ml]
cat("\n[ Estatísticas para mAP]-----------------------------\n")
dt[,list(median=median(mAP),IQR=IQR(mAP),mean=mean(mAP),sd=sd(mAP)),by=ml]
cat("\n[ Estatísticas para precision]-----------------------------\n")
dt[,list(median=median(precision),IQR=IQR(precision),mean=mean(precision),sd=sd(precision)),by=ml]
cat("\n[ Estatísticas para recall]-----------------------------\n")
dt[,list(median=median(recall),IQR=IQR(recall),mean=mean(recall),sd=sd(recall)),by=ml]
cat("\n[ Estatísticas para fscore]-----------------------------\n")
dt[,list(median=median(fscore),IQR=IQR(fscore),mean=mean(fscore),sd=sd(fscore)),by=ml]
cat("\n[ Estatísticas para MAE]-----------------------------\n")
dt[,list(median=median(MAE),IQR=IQR(MAE),mean=mean(MAE),sd=sd(MAE)),by=ml]
cat("\n[ Estatísticas para RMSE]-----------------------------\n")
dt[,list(median=median(RMSE),IQR=IQR(RMSE),mean=mean(RMSE),sd=sd(RMSE)),by=ml]
cat("\n[ Estatísticas para r]-----------------------------\n")
try(dt[,list(median=median(r),IQR=IQR(r),mean=mean(r),sd=sd(r)),by=ml])

sink()


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# APLICA TESTES DE HIPÓTESE E PÓS-TESTE
# Anova e Tukey para mAP, fscore e r


sink('./dataset/anova.txt')

cat("[ Teste para mAP50]-----------------------------","\n")
dados.anova <- aov(dados$mAP50 ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey

cat("[ Teste para mAP75]-----------------------------","\n")
dados.anova <- aov(dados$mAP75 ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey



cat("[ Teste para mAP]-----------------------------","\n")
dados.anova <- aov(dados$mAP ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey

cat("[ Teste para precision]-----------------------------","\n")
dados.anova <- aov(dados$precision ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey

cat("[ Teste para recall]-----------------------------","\n")
dados.anova <- aov(dados$recall ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey


cat("[ Teste para fscore]-----------------------------","\n")
dados.anova <- aov(dados$fscore ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey

cat("[ Teste para MAE]-----------------------------","\n")
dados.anova <- aov(dados$MAE ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey

cat("[ Teste para RMSE]-----------------------------","\n")
dados.anova <- aov(dados$RMSE ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey


cat("[ Teste para r]-----------------------------","\n")
dados.anova <- aov(dados$r ~ dados$ml)
summary(dados.anova)
tukey <- TukeyHSD(dados.anova,'dados$ml',conf.level=0.95)
tukey

sink()


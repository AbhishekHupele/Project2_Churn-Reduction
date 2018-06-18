
rm(list = ls())
# install.packages("corrgram", lib="/Library/Frameworks/R.framework/Versions/3.4/Resources/library")
# install.packages("sampling", lib="/Library/Frameworks/R.framework/Versions/3.4/Resources/library")
# install.packages("DataCombine", lib="/Library/Frameworks/R.framework/Versions/3.4/Resources/library")
# install.packages("caret", lib="/Library/Frameworks/R.framework/Versions/3.4/Resources/library")
library(psych)
library(ggplot2)
library(corrgram)
library(sampling)
library(corrgram)
library(class)
library(e1071)
library(caret)
library(DataCombine)
library(caret)
library(randomForest)
library(inTrees)
library(C50)

train = read.csv('Train_data.csv', header = TRUE)
test = read.csv('Test_data.csv', header = TRUE)

numeric_index = sapply(train,is.numeric) 
numeric_data = train[,numeric_index]
cnames = colnames(numeric_data)
# multi.hist(train[,numeric_index], main = NA, dcol = c("blue", "red"), dlty = c("solid", "solid"), bcol = "linen")

train$area.code = factor(train$area.code, labels = (1:length(levels(factor(train$area.code)))))
train$phone.number = as.factor(substr(train$phone.number,2,4))
numeric_index = sapply(train,is.numeric) 
numeric_data = train[,numeric_index]
cnames = colnames(numeric_data)
factor_index = sapply(train, is.factor)
summary(train[,factor_index])
factor_index = sapply(train,is.factor)
factor_data = train[,factor_index]
cnames_f = colnames(factor_data)

for (i in 1:ncol(train)) {
  if (class(train[,i]) == 'factor') {
    train[,i] = factor(train[,i],labels = 1:length(levels(factor(train[,i]))))
  }
}

for (i in 1:length(cnames)){
  assign(paste0('plot',i), ggplot(aes_string(y = cnames[i], x = 'Churn'), data = subset(train)) +
           stat_boxplot(geom = 'errorbar', width = 0.5) +
           geom_boxplot(outlier.colour = 'red', fill = 'grey', outlier.shape = 18, outlier.size = 1, notch = FALSE) +
           theme(legend.position = 'bottom') +
           labs(y = cnames[i], x = 'Churn') +
           ggtitle(paste('Boxplot of Churn for', cnames[i])))
}
gridExtra::grid.arrange(plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8, ncol = 4)
gridExtra::grid.arrange(plot9,plot10,plot11,plot12,plot13,plot14,plot15,ncol = 4)

for (i in cnames) {
  outlier =  train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
  train = train[which(!train[,i] %in% outlier),]
}
corrgram(train[,numeric_index], order = F, upper.panel = panel.pie, text.panel = panel.txt, main = 'CorrelationPlot')
symnum(cor(train[,numeric_index]))

for (i in cnames_f[1:5]) {
  print(i)
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}
train_deleted = subset(train, select = -c(area.code, phone.number, total.day.minutes, total.eve.minutes, total.night.minutes, total.intl.minutes))
numeric_index = sapply(train_deleted,is.numeric) 
numeric_data = train_deleted[,numeric_index]
cnames = colnames(numeric_data)

for ( i in cnames) {
  train_deleted[,i] = (train_deleted[,i] - min(train_deleted[,i]))/(max(train_deleted[,i]) - min(train_deleted[,i]))
}

test = read.csv('Test_data.csv', header = TRUE)
test$area.code = factor(test$area.code, labels = (1:length(levels(factor(test$area.code)))))
test$phone.number = as.factor(substr(test$phone.number,2,4))
numeric_index = sapply(test,is.numeric)
factor_index = sapply(test, is.factor)
for (i in 1:ncol(test)) {
  if (class(test[,i]) == 'factor') {
    test[,i] = factor(test[,i],labels = 1:length(levels(factor(test[,i]))))
  }
}
numeric_data = test[,numeric_index]
cnames = colnames(numeric_data)
factor_index = sapply(test,is.factor)
factor_data = test[,factor_index]
cnames_f = colnames(factor_data)
test_deleted = subset(test, select = -c(area.code,phone.number, total.day.minutes, total.eve.minutes, total.night.minutes, total.intl.minutes))
numeric_index = sapply(test_deleted,is.numeric)
cnames = colnames(test_deleted[,numeric_index])
for ( i in cnames) {
  test_deleted[,i] = (test_deleted[,i] - min(test_deleted[,i]))/(max(test_deleted[,i]) - min(test_deleted[,i]))
}


rmExcept(c('test_deleted','train_deleted'))
train = train_deleted
test = test_deleted

knn_pred = knn(train[,1:14],test[,1:14],train$Churn,k=1)
conf_mat = table(knn_pred, test$Churn)

nb_model = naiveBayes(Churn~.,data = train)
nb_pred = predict(nb_model, test[,1:14], type = 'class')
conf_mat = table(observed = test[,15], predicted = nb_pred)

rf_model = randomForest(Churn~., train, importance = TRUE, ntree = 500)
rf_pred = predict(rf_model, test[,-15])
conf_mat = table(test$Churn,rf_pred)

c50_model = C5.0(Churn~., train, trials = 50, rules = TRUE)
c50_pred = predict(c50_model,test[,-15],type = 'class')
conf_mat = table(test$Churn,c50_pred)

logis_model= glm(Churn~., data = train, family = 'binomial')
logis_pred = predict(logis_model, newdata = test, type = 'response')
logis_pred = ifelse(logis_pred > 0.5,1,0)
conf_mat = table(test$Churn, logis_pred)

conf_mat
confusionMatrix(conf_mat)
TN = conf_mat[1,1]
FN = conf_mat[2,1]
TP = conf_mat[2,2]
FP = conf_mat[1,2]
(FN*100)/(FN+TP)
(TP+TN)*100/(TN+FN+TP+FP)

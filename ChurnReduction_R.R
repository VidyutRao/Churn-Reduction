rm(list = ls())
setwd("D:/DS_New/Project 2")
getwd()

library(ggplot2)
library(corrgram)
library(randomForest)
library(DMwR)
library(randomForest)
library(C50)
library(e1071)
library(class)
library(usdm)
library(caret)
library(stats)

df = read.csv("Train_data.csv", header = T)
test = read.csv("Test_data.csv", header = T)

#Univariate Pre-Processing
df$area.code = as.factor(df$area.code)
df = subset(df, select = -c(phone.number))

#Encoding String Factors
for(i in 1:ncol(df)){
  if(class(df[,i]) == 'factor'){
    df[,i] = factor(df[,i], labels=(1:length(levels(factor(df[,i])))))
  }
}

#Changing Churn values to 0 and 1
df$Churn = as.numeric(df$Churn)
df$Churn = df$Churn-1
df$Churn = as.factor(df$Churn)

#Missing Value Analysis
miss_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
miss_val$Predictors = row.names(miss_val)
names(miss_val)[1] =  "Missing_Percentage"
miss_val$Missing_Percentage = (miss_val$Missing_Percentage/nrow(df)) * 100
miss_val = miss_val[order(-miss_val$Missing_Percentage),]
row.names(miss_val) = NULL
miss_val = miss_val[,c(2,1)]

#Creating separate Data Frames for numeric and categorical variables
numeric_index = sapply(df, is.numeric)
num_dt = df[,numeric_index]
num_nm = colnames(num_dt)

cat_index = sapply(df, is.factor)
cat_dt = df[,cat_index]
cat_name = colnames(cat_dt)

#Boxplot Analysis
#Account Length
ggplot(df, aes(x = Churn, y = account.length)) + 
  geom_boxplot(fill = "steelblue", color = "red") +
  ylab("Account Length") + xlab("Churn") +
  theme_minimal()

#Total Day Minutes
ggplot(df, aes(x = Churn, y = total.day.minutes)) + 
  geom_boxplot(fill = "darkgreen", color = "steelblue") +
  ylab("Total Day Minutes") + xlab("Churn") +
  theme_minimal()

#Histograms
qplot(account.length, data = df, geom = "histogram", fill =Churn)
qplot(total.day.calls, data = df, geom = "histogram", fill =Churn)
qplot(total.intl.charge, data = df, geom = "histogram", fill =Churn)

ggplot(df, aes(x = account.length)) +
 geom_area(aes(fill = Churn), stat = "bin", alpha = 0.8) +
  theme_classic()
 
ggplot(df, aes(x = account.length)) +
 geom_histogram(aes(colour = Churn), fill = "white", position  = "dodge")

#Bar Graphs
ggplot(df, aes(x = area.code)) +
geom_bar(fill = "red",colour = "darkgreen") +
  theme_classic()

#Outlier Removal
for(i in num_nm){
  out = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% out] = NA
}

#Imputing Missing Values
colnames(df)
df_test = df
df_test$number.customer.service.calls[20] 
df_test$number.customer.service.calls[20] = NA

#Test
df_test$number.customer.service.calls[is.na(df_test$number.customer.service.calls)] =median(df_test$number.customer.service.calls, na.rm = T)
df_test$number.customer.service.calls[is.na(df_test$number.customer.service.calls)] =mean(df_test$number.customer.service.calls, na.rm = T) 
df_test = knnImputation(df_test, k = 3)

#Freezing Median Method
for(i in num_nm){
  df[,i][is.na(df[,i])] = median(df[,i], na.rm = T)
}



#Validation  
boxplot(df$number.vmail.messages)
qplot(total.intl.charge, data = df, geom = "histogram", fill =Churn)
qplot(number.vmail.messages, data = df, geom = "histogram", fill =Churn)
qplot(number.customer.service.calls, data = df, geom = "histogram", fill =Churn)
qplot(total.eve.calls, data = df, geom = "histogram", fill =Churn)


#Feature Selection
#Variable Importance
rm(PredImp)
PredImp = randomForest(Churn ~ ., data = df, ntree = 1000, keep.forest = FALSE, importance = TRUE)
importance(PredImp, type = 1)

#Correlation Check
corrgram(df[,numeric_index], order = F,upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")
symnum(cor(df[,numeric_index]))

#Chi-Square Test
cat_index = sapply(df, is.factor)
cat_dt = df[,cat_index]
cat_name = colnames(cat_dt)

for(i in 1:4){
  print(cat_name[i])
  print(chisq.test(table(cat_dt$Churn,cat_dt[,i])))
}

#Dimension Reduction
df= subset(df, select = -c(total.night.calls, total.eve.calls, total.day.calls,account.length, total.day.charge, total.eve.charge, total.night.charge, total.intl.charge, area.code))

#Normalization
numeric_index = sapply(df, is.numeric)
num_dt = df[,numeric_index]
num_nm = colnames(num_dt)

for(i in num_nm){
  print(i)
  df[,i] = (df[,i] - min(df[,i]))/(max(df[,i]) - min(df[,i]))
}

#Clean the Test Dataset similarly
test = subset(test, select = -c(total.night.calls, total.eve.calls, total.day.calls, phone.number, account.length, total.day.charge, total.eve.charge, total.night.charge, total.intl.charge, area.code))

for(i in 1:ncol(test)){
  if(class(test[,i]) == 'factor'){
    test[,i] = factor(test[,i], labels=(1:length(levels(factor(test[,i])))))
  }
}

test$Churn = as.numeric(test$Churn)
test$Churn = test$Churn-1
test$Churn = as.factor(test$Churn)

numeric_index = sapply(test, is.numeric)
num_dt = test[,numeric_index]
num_nm = colnames(num_dt)

for(i in num_nm){
  out = test[,i][test[,i] %in% boxplot.stats(test[,i])$out]
  test[,i][test[,i] %in% out] = NA
}

for(i in num_nm){
  test[,i][is.na(test[,i])] = median(test[,i], na.rm = T)
}


for(i in num_nm){
  if(i != "number.vmail.messages")
  test[,i] = (test[,i] - min(test[,i]))/(max(test[,i]) - min(test[,i]))
}

#Model Development and Evaluation
#Decision Trees
C50_model = C5.0(Churn ~., df, trials = 100, rules = TRUE)
summary(C50_model)
c50_pred = predict(C50_model, test[, -11], type = "class")

confusionMatrix(table(test$Churn, c50_pred))
FN = 127
TP = 97
FNR = FN/(FN+TP)
FNR #0.5669643

#Random Forest
RF_model = randomForest(Churn ~ ., df, importance = TRUE, ntree = 1000)
RF_Pred = predict(RF_model, test[,-11])

confusionMatrix(table(test$Churn, RF_Pred))
FN = 114
TP = 110
FNR = FN/(FN+TP)
FNR #0.5089286

#Logistic Regression
vif(df[-11])
LR_model = glm(Churn ~ ., data = df, family = "binomial")
summary(LR_model)
LR_Pred = predict(LR_model, newdata = test[,-11], type = "response")
LR_Pred = ifelse(LR_Pred > 0.5, 1, 0)

confusionMatrix(table(test$Churn, LR_Pred))
FN = 193 
TP = 31
FNR = FN/(FN+TP)
FNR #0.8616071

df1 = subset(df, select = -c(number.vmail.messages))
test1 = subset(test, select = -c(number.vmail.messages))
LR_model = glm(Churn ~ ., data = df1, family = "binomial")

summary(LR_model)
LR_Pred = predict(LR_model, newdata = test1[,-10], type = "response")
LR_Pred = ifelse(LR_Pred > 0.5, 1, 0)
confusionMatrix(table(test1$Churn, LR_Pred))
FN = 190 
TP = 34
FNR = FN/(FN+TP)
FNR #0.8482143

#knn Imputation
knn_Pred = knn(df[,-11], test[,-11], df$Churn, k = 1)
confusionMatrix(table(test$Churn, knn_Pred))
FN = 167 
TP = 57
FNR = FN/(FN+TP)
FNR #0.7455357

knn_Pred = knn(df[,-11], test[,-11], df$Churn, k = 5)
confusionMatrix(table(test$Churn, knn_Pred))
FN = 186 
TP = 38
FNR = FN/(FN+TP)
FNR #0.8303571

#Naive Bayes
NB_model = naiveBayes(Churn ~., data = df)  
NB_Pred = predict(NB_model, test[-11], type = 'class') 
confusionMatrix(table(test$Churn, NB_Pred))

FN = 175
TP = 49
FNR = FN/(FN+TP)
FNR #0.78125

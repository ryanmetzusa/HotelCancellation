#load libraries used
library(caret)
library(tidyverse)
library(lubridate)
library(skimr)
library(ROCR)
library(nnet)
library(e1071)
library(glmnet)
library(Matrix)
library(countrycode)
library(randomForest)
library(rpart)
library(rpart.plot)
library(xgboost)
library(SHAPforxgboost)
library(partykit)
library(dplyr)
library(ggplot2)

#change from scientific notation
options(scipen=999)


#load in dataset
data <- read.csv("Hospitality.csv", header=T)
#holdout <- read.csv("Hospitality_holdout_noresponse.csv")
###Pre-Processing###

#Changing characters to factors
data<- data%>% mutate_if(is.character,as.factor)
#Removing Duplicates
data<-data %>%
  distinct(.keep_all = TRUE)
#Find null values
null_vars<-apply(data, 2, function(x) any(grepl('NULL', x)))==T
null_vars[null_vars == TRUE]

#Remove reservation_status because high carnality with is_canceled
data <- subset(data, select = -reservation_status)
#Formatting dates
data <- subset(data, select = -reservation_status_date)

#Added babeis to childrens column then removed babies column to reduce variables
data$children <- data$children + data$babies
data <- subset(data, select = -babies)



#Created continent variables for corresponding customer country origin, removed country column
data$continent <- countrycode(data$country, "iso3c", "continent")
#check for number of country unique types.
sum(data$country == "NULL", na.rm = TRUE)
sum(data$country == "ATA", na.rm = TRUE)
sum(data$country == "ATF", na.rm = TRUE)
sum(data$country == "TMP", na.rm = TRUE)
sum(data$country == "UMI", na.rm = TRUE)


data$continent[data$country == "CN"] <- "Asia"
data <- data[!(data$country %in% c("ATA", "ATF", "TMP", "UMI")), ]
data <- subset(data, select = -country)




#Feature engineering - roomtype diff
data$reserved_room_type <- as.character(data$reserved_room_type)
data$assigned_room_type <- as.character(data$assigned_room_type)

data$roomtype_diff <- ifelse(data$reserved_room_type == data$assigned_room_type, 1, 0)
#data <- subset(data, select = -reserved_room_type)
#data <- subset(data, select = -assigned_room_type)

#Changing response variable to character
data$is_canceled<-as.factor(data$is_canceled)
data$is_canceled<-fct_recode(data$is_canceled, not_canceled = "0", is_canceled = "1")
data$is_canceled<-relevel(data$is_canceled,ref="is_canceled")

#$roomtype_diff<-as.factor(data$roomtype_diff)
#data$roomtype_diff<-fct_recode(data$roomtype_diff, not_changed = "0", room_change = "1")
#data$roomtype_diff<-relevel(data$roomtype_diff,ref="room_change")



#Changed the levels to two, company or no company - Ryan
levels(data$company)[levels(data$company)!='NULL'] <- 'company'
data$company<-fct_recode(data$company, company = "company")
levels(data$company)

levels(data$company)[levels(data$company)=='NULL'] <- 'nocompany'
data$company<-fct_recode(data$company, nocompany = "nocompany")
levels(data$company)

#Change the levels for agent to two, agent or self - George
levels(data$agent)[levels(data$agent)!='NULL'] <- 'agent'
data$agent<-fct_recode(data$agent, agent = "agent")
levels(data$agent)

levels(data$agent)[levels(data$agent)=='NULL'] <- 'noagent'
data$agent<-fct_recode(data$agent, noagent = "noagent")
levels(data$agent)



#round adr column to 2 decimals - Kadin
data$adr <- round(data$adr, 2)

#change any characters to factors for dummy
data<- data%>% mutate_if(is.character,as.factor)
#Create dummy variables
dummies_model <- dummyVars(is_canceled~ ., data = data)
#if the response is a factor may get a warning that you can ignore
#provide only predictors that are now converted to dummy variables
predictors_dummy<- data.frame(predict(dummies_model, newdata = data)) 
#recombine predictors including dummy variables with response
data <- cbind(is_canceled=data$is_canceled, predictors_dummy) 


#Split testing and training data for imputation and fitting
set.seed(99)
index <- createDataPartition(data$is_canceled, p = .8,list = FALSE)
train_data <- data[index,]
test_data <- data[-index,]

# Impute missing numerical predictor variables using the rows for training data
# Create the median imputation model on the training data
preProcess_missingdata_model <- preProcess(train_data, method='medianImpute')
preProcess_missingdata_model
train_data <-predict(preProcess_missingdata_model,newdata=train_data)
test_data <-predict(preProcess_missingdata_model,newdata=test_data)

#Checking and removing NA values
sum(apply(is.na(train_data), 1, any))
data <- data[complete.cases(data), ]
sum(apply(is.na(data), 1, any))


library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)


set.seed(10)


###NNET###

ctrlnnet <- trainControl(method = "cv",    # Method for resampling
                     number = 5,                # Number of folds               
                     classProbs = TRUE,         # Enable class probabilities
                     summaryFunction = twoClassSummary)  # Performance metric for binary classification

nnetGrid <-  expand.grid(size = seq(from = 1, to = 10, by = 1),
                         decay = seq(from = 0.1, to = 2, by = 0.1))

nnet_model <- train(is_canceled~.,                       # Model formula
                       data = train_data,           # Training data
                       method = "nnet",                # Method (nnet)
                       trControl = ctrlnnet,               # Train control parameters
                       tuneGrid=nnetGrid,
                       metric="ROC")    

nnet_model
varImp(nnet_model$finalModel)
plot(varImp(nnet_model))
varImpnnet <- varImp(nnet_model$finalModel)
coef(nnet_model$finalModel, nnet_model$bestTune$size)
nnet_model$bestTune


predicted_model_nnet <- predict(nnet_model,test_data)
pred_nnet = prediction(predicted_model_nnet$is_canceled, test_data$is_canceled,label.ordering =c("not_canceled","is_canceled")) 
perf_nnet = performance(pred_nnet, "tpr", "fpr")
plot(perf_nnet, colorize=TRUE)
unlist(slot(performance(pred_nnet, "auc"), "y.values"))

confusionMatrix(predicted_model_nnet,test_data$is_canceled)

####LASSO####

lassoGrid =  expand.grid(alpha = 1, lambda = c(seq(0.1, 1.5, by = 0.1), seq(2,5, by=1), seq(5,20,by=5)))
ctrllasso <- trainControl(method = "cv",    # Method for resampling
                         number = 5,                # Number of folds               
                         classProbs = TRUE,         # Enable class probabilities
                         summaryFunction = twoClassSummary)  # Performance metric for binary classification

lasso_model <- train(is_canceled ~ .,
                      data = train_data,
                      method = "glmnet",
                      standardize =T,
                      trControl =ctrllasso,
                      metric="ROC")
lasso_model     
varImp(lasso_model$finalModel)
plot(varImp(lasso_model), top = 12)
varimplasso<-varImp(lasso_model$finalModel, lasso_model$bestTune$lambda)
coef(lasso_model$finalModel, lasso_model$bestTune$lambda)


predicted_model_lasso <- predict(lasso_model,test_data, type="prob")
pred_lasso = prediction(predicted_model_lasso$is_canceled, test_data$is_canceled,label.ordering =c("not_canceled","is_canceled")) 
perf_lasso = performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, colorize=TRUE)
unlist(slot(performance(pred_lasso, "auc"), "y.values"))



### RANDOM FOREST###
rf_grid <- expand.grid( mtry = sqrt(ncol(train_data)), ntree = 500, maxdepth = c(5,10,15,20))

model_rf <- train(is_canceled ~ .,#####
                  data = train_data,#####
                  method = "rf",
                  tuneGrid= rf_grid,
                  trControl =trainControl(method = "cv", 
                                          number = 5, 
                                          #Estimate class probabilities
                                          classProbs = TRUE,
                                          ## Evaluate performance using the following function
                                          summaryFunction = twoClassSummary),
                  metric="ROC")
model_rf
model_rf$finalModel
plot(model_rf$finalModel)
varImp(model_rf$finalModel)
varimprf <- varImp(model_rf$finalModel)
plot(varImp(model_rf), top=12)

model_rf$finalModel

predicted_model_rf <- predict(model_rf,test_data)
pred_rf = prediction(predicted_model_rf$is_canceled, test_data$is_canceled,label.ordering =c("not_canceled","is_canceled")) 
perf_rf = performance(pred_rf, "tpr", "fpr")
plot(perf_rf, colorize=TRUE)
unlist(slot(performance(pred_rf, "auc"), "y.values"))

confusionMatrix(predicted_model_rf, test_data$is_canceled)


trees <- getTree(model_rf$finalModel, k = 1, labelVar = TRUE)
pruned_tree <- prune(trees$tree)


model_rf$finalModel$err.rate
model_gbm

### rpart ###

model_rpart <- train(is_canceled ~ .,
                     data = train_data,
                     method = "rpart",
                     trControl =trainControl(method = "cv",
                                             number = 5,
                                             classProbs = TRUE,
                                             summaryFunction = twoClassSummary),
                     metric="ROC") 

model_rpart
varImp(model_rpart)
plot(varImp(model_rpart))
rpart.plot(model_rpart$finalModel, type=5)



predicted_model_rpart <- predict(model_rpart,test_data)
pred_rpart = prediction(predicted_model_rpart$is_canceled, test_data$is_canceled,label.ordering =c("not_canceled","is_canceled")) 
perf_rpart = performance(pred_gbm, "tpr", "fpr")
plot(perf_rpart, colorize=TRUE)
unlist(slot(performance(pred_rpart, "auc"), "y.values"))



confusionMatrix(predicted_model_rpart, test_data$is_canceled)

### XGBOOST ###

gbmctrld <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
gbmgrid <- expand.grid(
  nrounds = c(50,100,150,200),
  eta = c(0.01,.02,.03,.04,.05), #learning rate, higher = overfitting, lower= underfitting
  max_depth = c(6), #higher = overfitting, lower = underfitting
  gamma = 0,
  colsample_bytree = 1, #higher = underfitting/high bias, lower= overfitting/low bias
  min_child_weight = 1, #higher = underfitting/high bias, lower = overfitting/low bias
  subsample = 1)
model_gbm <- train(is_canceled ~ .,
                   data = train_data,
                   method = "xgbTree",
                   trControl =gbmctrld,
                   verbose=TRUE,
                   metric="ROC")


model_gbm
plot(model_gbm)
varImp(model_gbm)
plot(varImp(model_gbm), top=12)
model_gbm$bestTune
model_gbm$finalModel


predicted_model_gbm <- predict(model_gbm,test_data)
pred_gbm = prediction(predicted_model_gbm$is_canceled, test_data$is_canceled,label.ordering =c("not_canceled","is_canceled")) 
perf_gbm = performance(pred_gbm, "tpr", "fpr")
plot(perf_gbm, colorize=TRUE)
unlist(slot(performance(pred_gbm, "auc"), "y.values"))

confusionMatrix(predicted_model_gbm, test_data$is_canceled)

### XGBOOST 5-CV###
model_gbm_5cv <- train(is_canceled ~ .,
                       data = train_data,
                       method = "xgbTree",
                       trControl =gbmctrld,
                       tuneGrid = gbmgrid,
                       verbose=TRUE,
                       metric="ROC")


model_gbm_5cv
plot(model_gbm_5cv)
varImp(model_gbm_5cv)
plot(varImp(model_gbm_5cv), top=12)
model_gbm_5cv$bestTune
model_gbm_5cv$finalModel

predicted_model_gbm_5cv <- predict(model_gbm_5cv,test_data, type="prob")
pred_gbm_5cv = prediction(predicted_model_gbm_5cv$is_canceled, test_data$is_canceled,label.ordering =c("not_canceled","is_canceled")) 
perf_gbm_5cv = performance(pred_gbm_5cv, "tpr", "fpr")
plot(perf_gbm_5cv, colorize=TRUE)
unlist(slot(performance(pred_gbm_5cv, "auc"), "y.values"))



### SHAP PLOT###
xdata<-as.matrix(select(train_data,-is_canceled))
shap <- shap.prep(model_gbm$finalModel, X_train = Xdata)
shap_values <- shap.values(xgb_model = model_gbm$finalModel, X_train=Xdata)


shap_values$shap_score

shap.plot.summary.wrap2(shap_score = shap_values$shap_score, X = Xdata, top_n = 12)


shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = Xdata)

shap.plot.summary(shap_long)

shap.plot.summary(shap)

shap.plot.summary.wrap1(model_gbm$finalModel, X = Xdata, top_n = 10)

noshowdiff <- sum_adr_cancelled/sum_adr_cancelled1






###EDA###
#create new dataframe to check for how much dollars are lost from no shows##
sum_adr_cancelled <- data %>%
  filter(reservation_status == "No-Show") %>%
  group_by(arrival_date_year) %>%
  summarize(total_adr = sum(adr * (stays_in_week_nights + stays_in_weekend_nights)))


## code to transform data, have to refresh original dataframe for this, this is bad code##
cancelled_reservations <- data %>%
  filter(is_canceled == 1, datecheckdiff > 0)

lookup_table <- c(January = 1, February = 2, March = 3, April = 4, May = 5, June = 6,
                  July = 7, August = 8, September = 9, October = 10, November = 11, December = 12)

data$month_numeric <- lookup_table[data$arrival_date_month]

data$arrival_date <- as.Date(sprintf("%04d-%02d-%02d", data$arrival_date_year, data$month_numeric, data$arrival_date_day))

data$reservation_status_date <- as.Date(data$reservation_status_date, format = "%m/%d/%Y")

data$total_nightstays <- data$stays_in_week_nights + data$stays_in_weekend_nights

data$datecheckdiff <- as.numeric(data$reservation_status_date - data$arrival_date)


#creating various plots for lelin
plot(data$is_canceled, data$lead_time, ylab="lead_time", xlab="is_canceled", main="Is_Canceled v Lead_Time")
plot(data$is_canceled, data$adr, ylab="adr", xlab="is_canceled", main="Is_Canceled v adr", ylim=c(0,600))
plot(data$is_canceled, data$required_car_parking_spaces, ylab="# of Parking", xlab="is_canceled", main="Is_Canceled v # of Parking")


# Create a stacked bar plot
#deposit_type graph
ggplot(data, aes(x = is_canceled, fill = deposit_type)) +
  geom_bar() +
  labs(x = "is_canceled", y = "# of Reservations", fill = "deposit_type")
table(data$is_canceled, data$deposit_type)
prop.table(table(data$is_canceled, data$deposit_type),2)*100

#continent graph
ggplot(data, aes(x = is_canceled, fill = continent)) +
  geom_bar() +
  labs(x = "is_canceled", y = "# of Reservations", fill = "continent")



table(data$is_canceled, data$continent)
prop.table(table(data$is_canceled, data$continent),2)*100

#tables for various variables
table(data$is_canceled, data$required_car_parking_spaces)
table(data$is_canceled, data$total_of_special_requests)
table(data$is_canceled, data$adr)

#plot for customer_type
ggplot(data, aes(x = is_canceled, fill = customer_type)) +
  geom_bar() +
  labs(x = "is_canceled", y = "# of Reservations", fill = "customer_type")

table(data$is_canceled, data$customer_type)
      prop.table(table(data$is_canceled, data$customer_type),2)*100

#plot for roomtype_diff
ggplot(data, aes(x = is_canceled, fill = roomtype_diff)) +
  geom_bar() +
  labs(x = "is_canceled", y = "Count", fill = "roomtype_diff")

table(data$is_canceled, data$roomtype_diff)
prop.table(table(data$is_canceled, data$roomtype_diff),2)*100


#plot for market_segment
ggplot(data, aes(x = is_canceled, fill = market_segment)) +
  geom_bar() +
  labs(x = "is_canceled", y = "Count", fill = "market_segment")

table(data$is_canceled, data$market_segment)
prop.table(table(data$is_canceled, data$market_segment),2)*100

#splot for total_of_special requests
ggplot(data, aes(x = factor(is_canceled), y = total_of_special_requests)) +
  geom_boxplot() +
  xlab("Booking Cancellation") +
  ylab("Total Special Requests") +
  ggtitle("Boxplot of Total Special Requests by Booking Cancellation")



#plot for arrival day of month
data$arrival_date_day_of_month <- as.numeric(data$arrival_date_day_of_month)

ggplot(data, aes(fill = is_canceled, x = arrival_date_day_of_month)) +
  geom_bar() +
  labs(x = "is_canceled", y = "Count", fill = "arrival_date_day_of_month")

table(data$is_canceled, data$arrival_date_day_of_month)
prop.table(table(data$is_canceled, data$arrival_date_day_of_month),2)*100







### Holdout ####

hdata <- read.csv("Hospitality_holdout_noresponse.csv", header=T)
#holdout <- read.csv("Hospitality_holdout_noresponse.csv")
###Pre-Processing###
skim(hdata)
#Changing characters to factors
hdata<- hdata%>% mutate_if(is.character,as.factor)
#Removing Duplicates
hdata<-hdata %>%
  distinct(.keep_all = TRUE)
#Find null values
null_vars<-apply(hdata, 2, function(x) any(grepl('NULL', x)))==T
null_vars[null_vars == TRUE]

#Remove reservation_status because high carnality with is_canceled
hdata <- subset(hdata, select = -reservation_status)
#Formatting dates
hdata <- subset(hdata, select = -reservation_status_date)

#Added babeis to childrens column then removed babies column to reduce variables
hdata$children <- hdata$children + hdata$babies
hdata <- subset(hdata, select = -babies)



#Created continent variables for corresponding customer country origin, removed country column
hdata$continent <- countrycode(hdata$country, "iso3c", "continent")
#check for number of country unique types.
sum(hdata$country == "NULL", na.rm = TRUE)
sum(hdata$country == "ATA", na.rm = TRUE)
sum(hdata$country == "ATF", na.rm = TRUE)
sum(hdata$country == "TMP", na.rm = TRUE)
sum(hdata$country == "UMI", na.rm = TRUE)


hdata$continent[hdata$country == "CN"] <- "Asia"
hdata <- hdata[!(hdata$country %in% c("ATA", "ATF", "TMP", "UMI")), ]
hdata <- subset(hdata, select = -country)




#Feature engineering - roomtype diff
hdata$reserved_room_type <- as.character(hdata$reserved_room_type)
hdata$assigned_room_type <- as.character(hdata$assigned_room_type)

hdata$roomtype_diff <- ifelse(hdata$reserved_room_type == hdata$assigned_room_type, 1, 0)






#Changed the levels to two, company or no company - Ryan
levels(hdata$company)[levels(hdata$company)!='NULL'] <- 'company'
hdata$company<-fct_recode(hdata$company, company = "company")
levels(hdata$company)

levels(hdata$company)[levels(hdata$company)=='NULL'] <- 'nocompany'
hdata$company<-fct_recode(hdata$company, nocompany = "nocompany")
levels(hdata$company)

#Change the levels for agent to two, agent or self - George
levels(hdata$agent)[levels(hdata$agent)!='NULL'] <- 'agent'
hdata$agent<-fct_recode(hdata$agent, agent = "agent")
levels(hdata$agent)

levels(hdata$agent)[levels(hdata$agent)=='NULL'] <- 'noagent'
hdata$agent<-fct_recode(hdata$agent, noagent = "noagent")
levels(hdata$agent)



#round adr column to 2 decimals - Kadin
hdata$adr <- round(hdata$adr, 2)

#change any characters to factors for dummy
hdata<- hdata%>% mutate_if(is.character,as.factor)


skim(hdata)

hdata$children[is.na(hdata$children)]<-0

skim(hdata)

#Create dummy variables
dummies_model <- dummyVars(~., data = hdata)
#if the response is a factor may get a warning that you can ignore
#provide only predictors that are now converted to dummy variables
predictors_dummy<- data.frame(predict(dummies_model, newdata = hdata)) 
#recombine predictors including dummy variables with response
hdata <- cbind(predictors_dummy) 

hdata$market_segment.Undefined<-0
hdata$reserved_room_type.L<-0
hdata$reserved_room_type.P<-0
hdata$assigned_room_type.L<-0
hdata$assigned_room_type.P<-0

skim(hdata)

hdata$continent.Africa[is.na(hdata$continent.Africa)]<-0
hdata$continent.Americas[is.na(hdata$continent.Americas)]<-0
hdata$continent.Asia[is.na(hdata$continent.Asia)]<-0
hdata$continent.Europe[is.na(hdata$continent.Europe)]<-0
hdata$continent.Oceania[is.na(hdata$continent.Oceania)]<-0

skim(hdata)

#holdout prediction#
case_holdoutprob<- predict(model_rf, hdata)


case_holdout_scored<- cbind(hdata, case_holdoutprob)
case_holdout_scored[1:5,]

write.csv(case_holdout_scored, file="g1_caseholdout.csv")

## stop parallel processing###
stopCluster(cl)
registerDoSEQ()






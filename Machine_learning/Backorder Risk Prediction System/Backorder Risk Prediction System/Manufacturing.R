library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)
library(dgof)
library(dplyr)
library(magrittr)

wob_train=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\MANUFACTURING\\product_train.csv",stringsAsFactors = FALSE)
wob_test=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\MANUFACTURING\\product_test.csv",stringsAsFactors = FALSE)

wob_test$went_on_backorder=NA

wob_train$data='train'
wob_test$data='test'
wob_all=rbind(wob_train,wob_test)

apply(wob_all,2,function(x) length(unique(x)))

wob_all=wob_all %>% select(-sku)

glimpse(wob_all)

##Next we'll create dummy variables for remaining categorical variables
##using sapply for creating dummies
CreateDummies=function(data,var,freq_cutoff=100){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    name=gsub("/","_",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
} 
#potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop
wob_all=CreateDummies(wob_all,"rev_stop",20)

glimpse(wob_train)
head(wob_all)
glimpse(wob_all)

###we can go ahead and separate training and test data BUT first we check NA values
wob_all=wob_all[!((is.na(wob_all$went_on_backorder)) & wob_all$data=='train'), ]

for(col in names(wob_all)){
  if(sum(is.na(wob_all[,col]))>0 & !(col %in% c("data","went_on_backorder"))){
    wob_all[is.na(wob_all[,col]),col]=mean(wob_all[wob_all$data=='train',col],na.rm=T)
  }
}

wob_train = wob_all %>% filter(data == 'train') %>% select(-data) 
wob_test= wob_all %>% filter(data == 'test') %>% select(-data) 
wob_test=wob_test %>% select(-went_on_backorder)

any(is.na(wob_train))
any(is.na(wob_test))

#Using Random forest model to predict
library(randomForest)
fit = randomForest(as.factor(went_on_backorder)~., data = wob_train)


### Make predictions on test and submit 
test.predictions = predict(fit, newdata = wob_test)
ks.test(as.numeric(test.predictions),"pnorm")
write.csv(test.predictions,file = "F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\MANUFACTURING\\mnft.csv", row.names = F)


t_test()

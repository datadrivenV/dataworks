setwd("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\MANUFACTURING")

library(dplyr)
library(car)

p_train=read.csv("product_train.csv")
p_test=read.csv("product_test.csv")

p_test$went_on_backorder=NA

p_train$data="train"
p_test$data="test"
p_all=rbind(p_train,p_test)

glimpse(p_all)

p_all=p_all %>% 
  select(-ppap_risk,-oe_constraint,-deck_risk,-stop_auto_buy,-rev_stop)

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

char_logical=sapply(p_all,is.character)
cat_cols=names(p_all)[char_logical]
cat_cols

cat_cols=cat_cols[!(cat_cols %in% c('data','went_on_backorder'))]
cat_cols

for(col in cat_cols){
  p_all=CreateDummies(p_all,col,50)
}

glimpse(p_all)

p_all=p_all[!(is.na(p_all$went_on_backorder) & p_all$data=="train"),]
for(col in names(p_all)){
  if(sum(is.na(p_all[,col]))>0 & !(col %in% c("data","went_on_backorder"))){
    p_all[is.na(p_all[,col]),col]=mean(p_all[p_all$data=='train',col],na.rm=T)
  }
}

p_all$went_on_backorder=ifelse(p_all$went_on_backorder=="No",0,1)
p_all$went_on_backorder=as.numeric(p_all$went_on_backorder)

glimpse(p_all)
p_train=p_all %>% 
  filter(data=="train") %>% 
  select(-data)

p_test=p_all %>% 
  filter(data=="test") %>% 
  select(-data)

glimpse(p_train)

fit=lm(went_on_backorder~.,data=p_train)
sort(vif(fit),decreasing=T)[1:3]

fit=lm(went_on_backorder~. -forecast_6_month,data=p_train)
sort(vif(fit),decreasing=T)[1:3]

fit=lm(went_on_backorder~. -forecast_6_month -sales_6_month,data=p_train)
sort(vif(fit),decreasing=T)[1:3]

fit=lm(went_on_backorder~. -forecast_6_month -sales_6_month -sales_9_month,
       data=p_train)
sort(vif(fit),decreasing=T)[1:3]

fit=lm(went_on_backorder~. -forecast_6_month 
       -sales_6_month -sales_9_month -forecast_9_month,data=p_train)
sort(vif(fit),decreasing=T)[1:3]

fit=lm(went_on_backorder~. -forecast_6_month 
       -sales_6_month -sales_9_month -forecast_9_month -sales_1_month,data=p_train)
sort(vif(fit),decreasing=T)[1:3]

formula(fit)

p_train=p_train %>% 
  select(-forecast_6_month, 
         -sales_6_month, -sales_9_month, -forecast_9_month, -sales_1_month)

library(randomForest)
fit=randomForest(as.factor(went_on_backorder)~.,data=p_train,classwt=c(0.99,0.01),do.trace=T)
fit

train_pred=round(predict(fit,new_data = p_train,type="prob")[,2],1)
test_pred=round(predict(fit,new_data = p_test,type="prob")[,2],1)

ks.test(test_pred,"pnorm")







train_pred1=predict(fit,newdata = p_train,type= "response")
test_pred1=predict(fit,newdata = p_test,type= "response")
test_pred1=as.numeric(test_pred1)
test_pred1=data.frame(test_pred1)
test_pred1$test_pred1=ifelse(test_pred1$test_pred1==1,"No","Yes")
ks.test(test_pred1,"pnorm")
names(test_pred1)="went_on_backorder"

write.csv(test_pred1,"F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\MANUFACTURING\\Vanshika_Parkar_P3_part2.csv",row.names=FALSE)

library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)
library(tidymodels)
library(visdat)

bank_train=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\BANKING\\bank-full_train.csv",stringsAsFactors = FALSE)
bank_test=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\BANKING\\bank-full_test.csv",stringsAsFactors = FALSE)
glimpse(bank_train)

to_num_func=function(x){
  x=as.numeric(x)
  return(x)
}

month_func=function(x){
  x=recode(x,
           jan=1,
           feb=2,
           mar=3,
           apr=4,
           may=5,
           jun=6,
           jul=7,
           aug=8,
           sep=9,
           oct=10,
           nov=11,
           dec=12)
  return(x)
}

cont_func=function(x){
  x=ifelse((x %in% c('telephone','cellular')),1,0)
  return(x)
}


bank_train$y=as.factor(as.numeric(bank_train$y=='yes'))

dp_pipe=recipe(y~.,data = bank_train) %>% 
  update_role(ID,new_role = "drop_var") %>% 
  update_role(job,marital,education,poutcome,new_role = "to_dummies") %>% 
  step_rm(has_role("drop_var")) %>% 
  step_mutate_at(default,fn=to_num_func) %>%
  step_mutate_at(housing,fn=to_num_func) %>% 
  step_mutate_at(loan,fn=to_num_func) %>% 
  step_mutate_at(month,fn=month_func) %>% 
  step_mutate_at(contact,fn=cont_func) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())


dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=bank_test)

## Random Forest

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(10,20)),levels = 3)


my_res=tune_grid(
  rf_model,
  y~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(y~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

ks.test(test_pred,"pnorm")
### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$y

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)
test_preds=data.frame(test_hard_class)
names(test_preds)="y"
write.csv(test_preds,"F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\BANKING\\Vanshika_Parkar_P5_part2.csv",row.names=FALSE)






library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)
library(tidymodels)
library(visdat)
prod_train=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\MANUFACTURING\\product_train.csv",stringsAsFactors = FALSE)
prod_test=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\MANUFACTURING\\product_test.csv",stringsAsFactors = FALSE)
glimpse(prod_train)
to_num_func=function(x){
  x=ifelse(x=="Yes",1,0)
  return(x)
}

chisq.test(prod_train$went_on_backorder,prod_train$deck_risk)

prod_train$went_on_backorder=as.factor(as.numeric(prod_train$went_on_backorder=='Yes'))

dp_pipe=recipe(went_on_backorder~.,data=prod_train) %>% 
  update_role(sku,new_role = "drop_vars") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_mutate_at(potential_issue,fn=to_num_func) %>%
  step_mutate_at(deck_risk,fn=to_num_func) %>%
  step_mutate_at(oe_constraint,fn=to_num_func) %>%
  step_mutate_at(ppap_risk,fn=to_num_func) %>%
  step_mutate_at(stop_auto_buy,fn=to_num_func) %>%
  step_mutate_at(rev_stop,fn=to_num_func) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=prod_test)
table(is.na(test))

## Random Forest

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,15)), trees(c(100,500)),
                       min_n(c(10,20)),levels = 3)


my_res=tune_grid(
  rf_model,
  went_on_backorder~.,
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

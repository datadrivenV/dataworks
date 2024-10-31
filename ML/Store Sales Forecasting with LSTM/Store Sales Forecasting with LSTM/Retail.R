library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)

st_train=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\RETAIL\\store_train.csv",stringsAsFactors = FALSE)
st_test=read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\RETAIL\\store_test.csv",stringsAsFactors = FALSE)

glimpse(st_train)

st_train$store=ifelse(st_train$store==1,"Yes","No")
st_train$store=as.factor(as.numeric(st_train$store=="Yes"))

dp_pipe=recipe(store~.,data=st_train) %>% 
  update_role(Id,State,countyname,Areaname,countytownname,storecode,new_role = "drop_vars") %>%
  update_role(country,store_Type,state_alpha,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)
train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=st_test)


glimpse(test)
##randomforest

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,30)), trees(c(100,500)),
                       min_n(c(10,20)),levels = 3)


my_res=tune_grid(
  rf_model,
  store~.,
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
  fit(store~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=round(predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1),2)
test_pred=round(predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1),2)
View(test_pred)
names(test_pred)="store"

write.csv(test_pred,"F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\RETAIL\\Vanshika_Parkar_P2_part2.csv",row.names=FALSE)

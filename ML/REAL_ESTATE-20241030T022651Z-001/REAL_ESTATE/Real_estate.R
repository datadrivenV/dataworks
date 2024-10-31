library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)


hsg_train= read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\REAL_ESTATE\\housing_train.csv",
             stringsAsFactors = F,sep=',',na.strings = c("","NA"))
hsg_test= read.csv("F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\REAL_ESTATE\\housing_test.csv",
                    stringsAsFactors = F,sep=',',na.strings = c("","NA"))

dp_pipe=recipe(Price~.,data=hsg_train) %>% 
  update_role(Address,new_role = "drop_vars") %>%
  update_role(Postcode,new_role = "to_numeric") %>% 
  update_role(CouncilArea,new_role = "to_char") %>% 
  step_mutate_at(has_role("to_char"),fn=as.character) %>% 
  update_role(Suburb,Type,Method,SellerG,CouncilArea,new_role = "to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_mutate_at(has_role("to_numeric"),fn=as.numeric) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(),-all_outcomes()) 
  
dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)
test=bake(dp_pipe,new_data=hsg_test)  

#random_forest
rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("regression") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(2,10)),levels = 3)
doParallel::registerDoParallel()

my_res=tune_grid(
  rf_model,
  Price~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(rmse,mae),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"rmse")) %>% 
  fit(Price~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "darkblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=round(predict(final_rf_fit,new_data = train),2)
test_pred=round(predict(final_rf_fit,new_data = test),2)
names(test_pred)="Price"

## partial dependence plots


model_explainer =explain_tidymodels(
  final_rf_fit,
  data = dplyr::select(train, -Price),
  y = as.integer(train$Price),
  verbose = FALSE
)

pdp = model_profile(
  model_explainer,
  variables = "Postcode",
  N = 1000
)

plot(pdp)

write.csv(test_pred,"F:\\Vanshika\\DATA\\Rdata\\PROJECTS\\REAL_ESTATE\\Vanshika_Parkar_P1_part2.csv",row.names=FALSE)



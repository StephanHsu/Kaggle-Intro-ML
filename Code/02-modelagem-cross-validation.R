library(tidyverse)
library(tidymodels)
library(recipes)
library(rsample)

# Leitura da base
adults <- readRDS('Data/adult.rds')
adults$resposta <- factor(adults$resposta)

# Separando a base em treino e teste
set.seed(1)
adults_split <- initial_split(adults, prop = 0.8)

# Pegando as bases de treino e teste
adults_train <- training(adults_split)
adults_test <- testing(adults_split)

# Base para cross-validation
adults_cv <- vfold_cv(adults_train, v = 5)

# Feature Engine ----------------------------------------------------------
adults_recipe <- recipe(resposta ~ ., data = adults_train) %>%
  step_rm(education, native_country, capital_loss, fnlwgt, id, skip = TRUE) %>%
  step_center(all_predictors(), -all_nominal()) %>%
  step_unknown(all_predictors(), -all_numeric()) %>%
  step_dummy(all_predictors(), -all_numeric())
  # step_log(capital_gain)


# Modelagem ---------------------------------------------------------------

# Especificando o modelo de regressão logística
modelo_log_reg <- logistic_reg(
    penalty = tune(),
    mixture = 1) %>% # 1-LASSO/0-RIDGE
  set_engine("glmnet") %>%
  set_mode("classification")

# Criando um workflow
log_reg_workflow <- workflow() %>%
  add_recipe(adults_recipe) %>%
  add_model(modelo_log_reg)

# Variando penalty
logit_grid <- modelo_log_reg %>%
  parameters() %>%
  grid_max_entropy(size = 100)

# Ajustando o modelo com cross-validation
adults_tune_grid <-  tune_grid(
    log_reg_workflow,
    resamples = adults_cv,
    grid = logit_grid,
    metrics = metric_set(accuracy),
    control = control_grid(verbose = TRUE, allow_par = FALSE))

# Verificando o ajuste do modelo
autoplot(adults_tune_grid)
collect_metrics(adults_tune_grid)
show_best(adults_tune_grid, "accuracy")

# Selecionando o melhor modelo
adults_best_params <- select_best(adults_tune_grid, "accuracy")
adults_model <- log_reg_workflow %>% finalize_workflow(adult_best_params)

adults_last_fit <- last_fit(adults_model, adults_split)


# Incluindo as previsões na base
adults_train_pred <- adults_train %>%
  mutate(
    pred = predict(adults_model, new_data = .)$.pred_class,
    prob = predict(adults_model, new_data = ., type = "prob")$`.pred_>50K`
  )

# Métricas de ajuste do modelo
multimetric <- metric_set(accuracy,
                          kap,
                          bal_accuracy,
                          sens,
                          yardstick::spec,
                          precision,
                          recall,
                          ppv,
                          npv)

adults_train_pred %>% multimetric(truth = resposta, estimate = pred)
adults_train_pred %>% roc_auc(truth = resposta, prob)

# aplicando na base teste
adults_test <- adults_recipe %>% bake(adults_test)
adults_test_pred <- adults_test %>%
  mutate(
    pred = predict(adults_model, new_data = .)$.pred_class,
    prob = predict(adults_model, new_data = ., type = "prob")$`.pred_>50K`
  )

adults_test_pred %>% multimetric(truth = resposta, estimate = pred)
adults_test_pred %>% roc_auc(truth = resposta, prob)


# Salvando workflow e modelo
saveRDS(adults_model, "adults_model.RDS")


# Fazendo previsão --------------------------------------------------------

# Lendo a base para previsão
submit <- readRDS("intro-ml-mestre-master/dados/dados_kaggle/adult_val.rds")
id <- submit$id
resposta <- submit$resposta
submit <- adults_recipe %>% bake(submit %>% select(-resposta))
submit_pred <- submit %>%
  mutate(
    pred = predict(adults_model, new_data = .)$.pred_class,
    prob = predict(adults_model, new_data = ., type = "prob")$`.pred_>50K`
  )

submit_final <- tibble(id = id,
       more_than_50K = submit_pred$prob,
       resposta_pred = factor(submit_pred$pred),
       resposta = factor(resposta))

submit_final %>% accuracy(truth = resposta, estimate = resposta_pred)

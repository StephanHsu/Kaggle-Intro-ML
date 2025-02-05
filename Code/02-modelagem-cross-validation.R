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


# Métricas de ajuste do modelo
# multimetric <- metric_set(accuracy,
#                           kap,
#                           bal_accuracy,
#                           sens,
#                           yardstick::spec,
#                           precision,
#                           recall,
#                           ppv,
#                           npv,
#                           roc_auc)


# Ajustando o modelo com cross-validation
adults_tune_grid <-  tune_grid(
    log_reg_workflow,
    resamples = adults_cv,
    grid = logit_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE, allow_par = FALSE))

# Verificando o ajuste do modelo
autoplot(adults_tune_grid)
collect_metrics(adults_tune_grid)
show_best(adults_tune_grid, "roc_auc")

# Selecionando o melhor modelo
adults_best_params <- select_best(adults_tune_grid, "roc_auc")
adults_wf <- log_reg_workflow %>% finalize_workflow(adults_best_params)

# Construindo o modelo
adults_last_fit <- last_fit(adults_wf, adults_split)

# Variáveis importantes
adults_model <- adults_last_fit$.workflow[[1]]$fit$fit
vip::vip(adults_model)
collect_metrics(adults_last_fit) # roc_auc - 0.906

# Salvando o modelo
write_rds(adults_last_fit, "Output/adults_last_fit.rds")
write_rds(adults_model, "Output/adults_model.rds")

# Predições
adults_pred <- collect_predictions(adults_last_fit)

# Curva ROC
adults_pred %>%
  roc_curve(resposta, `.pred_>50K`) %>%
  autoplot()

# Matriz de confusão
adults_pred %>% yardstick::conf_mat(resposta, .pred_class)

# risco por faixa de score
adults_pred %>%
  mutate(
    score =  factor(ntile(.pred_class, 10))
  ) %>%
  count(score, resposta) %>%
  ggplot(aes(x = score, y = n, fill = resposta)) +
  geom_col(position = "fill") +
  geom_label(aes(label = n), position = "fill") +
  coord_flip()

# gráfico sobre os da classe "bad"
percentis = 20
adults_pred %>%
  mutate(
    score = factor(ntile(.pred_class, percentis))
  ) %>%
  filter(resposta == ">50K") %>%
  group_by(score) %>%
  summarise(
    n = n(),
    media = mean(.pred_class)
  ) %>%
  mutate(p = n/sum(n)) %>%
  ggplot(aes(x = p, y = score)) +
  geom_col() +
  geom_label(aes(label = scales::percent(p))) +
  geom_vline(xintercept = 1/percentis, colour = "red", linetype = "dashed", size = 1)


# Salvando o modelo final
adults_final_model <- fit(adults_wf, adults)
write_rds(adults_final_model, "Output/adults_final_model.rds")

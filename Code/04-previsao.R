library(tidyverse)
library(tidymodels)
library(recipes)
library(rsample)


# Previsão numa nova base -------------------------------------------------

# Definindo o modelo
# adults_final_model.rds -> regressão logística
# adults_final_model_xgb.rds -> xgboost
adults_final_model <- readRDS("Output/adults_final_model_xgb_v2.rds")

# Lendo a base para previsão
submit <- readRDS("Data/adult_val.rds")
id <- submit$id
resposta <- submit$resposta
submit_pred <- submit %>% select(-resposta) %>%
  mutate(
    pred = predict(adults_final_model, new_data = .)$.pred_class,
    prob = predict(adults_final_model, new_data = ., type = "prob")$`.pred_>50K`
  )

submit_final <- tibble(id = id,
                       more_than_50K = submit_pred$prob,
                       resposta_pred = factor(submit_pred$pred),
                       resposta = factor(resposta))

submit_final %>% accuracy(truth = resposta, estimate = resposta_pred)
submit_final %>% roc_auc(truth = resposta, more_than_50K)

# Salvando as previsões
submit_final %>%
  select(id,more_than_50K) %>%
  write.table("Output/submission_xgb_v2.csv", row.names = FALSE, dec = ".", sep = ",")

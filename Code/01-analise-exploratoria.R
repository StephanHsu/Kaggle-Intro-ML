library(tidyverse)

# Leitura das bases
adults <- readRDS('Data/adult.rds')
adults %>% head() %>% View()

# Avaliando a variável resposta
adults %>%
  count(resposta) %>%
  mutate(freq = n / sum(n))

# Criando uma variável resposta dummy
adults$resp_dummy = adults$resposta == '>50K'

# Análise Exploratória ----------------------------------------------------

# Análise Univariada
adults %>%
  summarytools::dfSummary() %>%
  summarytools::view()

adults %>%
  skimr::skim() # Outro jeito de fazer

# Análise Bivariada

# Categóricas
vars_cat <- adults %>% select_if(is.character) %>% names()
vars_cat <- c(vars_cat, 'resp_dummy')

# Quantidade de categorias
adults[vars_cat] %>%
  map_df(n_distinct)

# Cruzando com a resposta
tab_cat <- adults[vars_cat] %>%
  gather(var, categ, -resp_dummy) %>%
  group_by(var, categ) %>%
  summarise(tx_resp = mean(resp_dummy),
            qtd = n()) %>%
  arrange(var, -tx_resp)

# Alguns gráficos
graf_cat <- function(nome_var){
  if(!(nome_var %in% c('resp_dummy','resposta'))){
    print(tab_cat %>%
      filter(var == nome_var) %>%
      ggplot() +
      geom_col(aes(x = reorder(categ, -tx_resp), y = tx_resp)) +
      labs(x = nome_var))
  }
}

for(var in vars_cat) graf_cat(var)


# Numéricas
vars_num <- adults %>% select_if(is.numeric) %>% names()
vars_num <- c(vars_num, 'resposta')

# Missings
adults[vars_num] %>%
  map_df(function(x) is.na(x) %>% sum())

# Correlação entre as variáveis
adults[vars_num] %>% select(-resposta, -id) %>%
  cor()

# Calculando a média de cada variável pela resposta
adults[vars_num] %>%
  group_by(resposta) %>%
  summarise_all(list(mean = mean))

# Alguns gráficos
graf <- function(nome_var){

  if(!(var %in% c('resposta','id'))){
    print(adults[vars_num] %>%
      ggplot() +
      geom_density(aes(x = get(nome_var), fill = resposta, alpha = .8)) +
      labs(x = nome_var))
}}

for(var in vars_num) graf(var)

# Outro jeito
# GGally::ggpairs(adults[vars_num])

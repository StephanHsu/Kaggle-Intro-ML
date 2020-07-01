library(devtools)
library(usethis)

# Configurações do repositório no Github ----------------------------------

# Criar um novo token
# usethis::browse_github_pat()
# usethis::edit_r_environ()

# Comando para criar um novo projeto
usethis::create_project("C:/Users/steph/Documents/2. Estudos/Projetos/Github/Entrega Final - Introdução ao ML R/")

# Inicializar o Git
usethis::use_git()
usethis::use_github()

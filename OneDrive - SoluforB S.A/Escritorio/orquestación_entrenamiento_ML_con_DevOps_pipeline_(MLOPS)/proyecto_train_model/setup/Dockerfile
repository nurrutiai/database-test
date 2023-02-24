
##############################################################################
# Definición de ambiente de ejecución de pasos de R (dockerfile)
##############################################################################

FROM rocker/tidyverse:4.0.4


# Instalar python
RUN apt-get update -qq && \
 apt-get install -y python3
 
# Crear link simbólico para python
RUN ln -f /usr/bin/python3 /usr/bin/python
RUN apt-get install -y build-essential

RUN R -e "install.packages(c('ddalpha', 'recipes'), dependencies = TRUE)"
RUN R -e "update.packages('Rcpp')"

# Instalar parquetes de R requeridos para scoring y preparacion
RUN R -e "install.packages(c('ellipsis','dplyr','gower','caret','azuremlsdk','optparse'),dependencies = TRUE)"

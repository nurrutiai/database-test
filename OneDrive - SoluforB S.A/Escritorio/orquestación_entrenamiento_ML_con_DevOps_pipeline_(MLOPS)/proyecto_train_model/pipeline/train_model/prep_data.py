#%%writefile $experiment_folder/prep_data.py

##############################################################################
# Paso de preparación de datos
##############################################################################

# Importar librerías
import os
import argparse
import pandas as pd
from azureml.core import Run
# Librerías
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
from dateutil.relativedelta import relativedelta

# Cargar parámetros
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='Datos crudos del proceso de ADF')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Carpeta para resultado')
args = parser.parse_args()
save_folder = args.prepped_data

# Cargar contexto de ejecución
run = Run.get_context()

# Cargar datos(pasados como un dataset de entrada)
print("Cargando datos...")
df = pd.read_csv(args.raw_dataset_id, delimiter='\t', encoding='latin1', quotechar='"')

# Log cantidad de filas
row_count = (len(df))
run.log('#filas_crudas', row_count)

##############################################################################
#
# Preparación de datos
#
##############################################################################

# Encabezados a lower case
df = df.rename(columns=str.lower)
df = df.drop(columns=['m_avg_monto_pagado','m_last_monto_pagado','m_c3'])

run.log('fecha_actual',(datetime.datetime.now()).strftime('%Y%m'))
run.log('fecha_de_corte',(datetime.datetime.now()-relativedelta(months=2)).strftime('%Y%m'))
fecha_de_corte = int((datetime.datetime.now()-relativedelta(months=2)).strftime('%Y%m'))

df = df[df['mes'] <= fecha_de_corte]


# Eliminar NaN
df['m_litros'] = df['m_litros'].fillna(0)
df['m_n_documentos'] = df['m_n_documentos'].fillna(0)
df['m_n_estaciones'] = df['m_n_estaciones'].fillna(0)
df['m_n_patentes'] = df['m_n_patentes'].fillna(0)
df['m_n_tarjetas'] = df['m_n_tarjetas'].fillna(0)
#df['m_c3'] = df['m_c3'].fillna(0)
df['m_antiguedad'] = df['m_antiguedad'].fillna(-1)
df['m_ratio_meses_sin_consumo'] = df['m_ratio_meses_sin_consumo'].fillna(0)

df['m_var_doc_mes_ant'] = df['m_var_doc_mes_ant'].fillna(0)
df['m_var_doc_2m'] = df['m_var_doc_2m'].fillna(0)
df['m_var_doc_3m'] = df['m_var_doc_3m'].fillna(0)
df['m_var_doc_6m'] = df['m_var_doc_6m'].fillna(0)
df['m_var_doc_9m'] = df['m_var_doc_9m'].fillna(0)
df['m_var_doc_12m'] = df['m_var_doc_12m'].fillna(0)

df['m_var_est_mes_ant'] = df['m_var_est_mes_ant'].fillna(0)
df['m_var_est_2m'] = df['m_var_est_2m'].fillna(0)
df['m_var_est_3m'] = df['m_var_est_3m'].fillna(0)
df['m_var_est_6m'] = df['m_var_est_6m'].fillna(0)
df['m_var_est_9m'] = df['m_var_est_9m'].fillna(0)
df['m_var_est_12m'] = df['m_var_est_12m'].fillna(0)

df['m_var_pat_mes_ant'] = df['m_var_pat_mes_ant'].fillna(0)
df['m_var_pat_2m'] = df['m_var_pat_2m'].fillna(0)
df['m_var_pat_3m'] = df['m_var_pat_3m'].fillna(0)
df['m_var_pat_6m'] = df['m_var_pat_6m'].fillna(0)
df['m_var_pat_9m'] = df['m_var_pat_9m'].fillna(0)
df['m_var_pat_12m'] = df['m_var_pat_12m'].fillna(0)

df['m_var_tar_mes_ant'] = df['m_var_tar_mes_ant'].fillna(0)
df['m_var_tar_2m'] = df['m_var_tar_2m'].fillna(0)
df['m_var_tar_3m'] = df['m_var_tar_3m'].fillna(0)
df['m_var_tar_6m'] = df['m_var_tar_6m'].fillna(0)
df['m_var_tar_9m'] = df['m_var_tar_9m'].fillna(0)
df['m_var_tar_12m'] = df['m_var_tar_12m'].fillna(0)

df['m_avg_deuda_abierta'] = df['m_avg_deuda_abierta'].fillna(-1)
df['m_avg_deuda_total'] = df['m_avg_deuda_total'].fillna(-1)
df['m_avg_limite_credito'] = df['m_avg_limite_credito'].fillna(-1)
df['m_avg_monto_disponible'] = df['m_avg_monto_disponible'].fillna(-1)
#df['m_avg_monto_pagado'] = df['m_avg_monto_pagado'].fillna(-1)
df['m_avg_ratio_monto_disp'] = df['m_avg_ratio_monto_disp'].fillna(-1)
df['m_num_bloqueos'] = df['m_num_bloqueos'].fillna(-1)

df['m_last_cod_bloqueo'] = df['m_last_cod_bloqueo'].fillna('-1')
df['m_last_deuda_abierta'] = df['m_last_deuda_abierta'].fillna(-1)
df['m_last_deuda_total'] = df['m_last_deuda_total'].fillna(-1)
df['m_last_limite_credito'] = df['m_last_limite_credito'].fillna(-1)
df['m_last_monto_disponible'] = df['m_last_monto_disponible'].fillna(-1)
#df['m_last_monto_pagado'] = df['m_last_monto_pagado'].fillna(-1)
df['m_last_ratio_monto_disp'] = df['m_last_ratio_monto_disp'].fillna(-1)
df['m_dias_ult_trx_mes_actual'] = df['m_dias_ult_trx_mes_actual'].fillna(99)
df['m_dias_ult_trx_mes_anterior'] = df['m_dias_ult_trx_mes_anterior'].fillna(99)
df['m_meses_ult_trx_mes_actual'] = df['m_meses_ult_trx_mes_actual'].fillna(99)
df['m_meses_ult_trx_mes_anterior'] = df['m_meses_ult_trx_mes_anterior'].fillna(99)
df['m_var_c3_mes_ant'] = df['m_var_c3_mes_ant'].fillna(0)
df['m_prc_contrib_c3'] = df['m_prc_contrib_c3'].fillna(-1)
df['m_prc_contrib_acum_c3'] = df['m_prc_contrib_acum_c3'].fillna(-1)
df['m_c3_12m'] = df['m_c3_12m'].fillna(-1)
df['m_prc_contrib_c3_12m'] = df['m_prc_contrib_c3_12m'].fillna(-1)
df['m_prc_contrib_acum_c3_12m'] = df['m_prc_contrib_acum_c3_12m'].fillna(-1)
df['m_prc_contrib_lts_12m'] = df['m_prc_contrib_lts_12m'].fillna(-1)
df['m_prc_contrib_acum_lts_12m'] = df['m_prc_contrib_acum_lts_12m'].fillna(-1)

# Label encoding
num, cat = [],[]
for n in df.columns:
    if type(df[n][0]) != str:
        num.append(n)
    else:
        cat.append(n)
dfcat=df[cat]
dfnum=df[num]
le = LabelEncoder()

for n in cat:
    print(n)
    df[n] = le.fit_transform(df[n])




##############################################################################
# Exportación de datos
##############################################################################

# Loguear filas procesadas
row_count = (len(df))
run.log('#filas_preparadas', row_count)

#Guardar datos preparados
print("Guardando Datos...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
df.to_csv(save_path, index=False, header=True)

# Completar la ejecución
run.complete()

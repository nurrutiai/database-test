##############################################################################
# Paso de exportación de datos (Export)
##############################################################################

# Importar Librerías
from azureml.core import Run, Model, Datastore
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Cargar parámetros
parser = argparse.ArgumentParser()
parser.add_argument('--scored_data', type=str, dest='scored_data', help='Carpeta con datos scoreados')
parser.add_argument('--output_dir', type=str, dest='output_dir', default='output_dir', help='Carpeta para resultado')
args = parser.parse_args()

prediction_dir = args.scored_data
output_dir = args.output_dir

# Cargar el contexto de ejecución
run = Run.get_context()

##############################################################################
# Exportación
##############################################################################

# Carfar los datos
print("Cargando Datos...")
file_path = os.path.join(prediction_dir,'predicciones.csv')
data = pd.read_csv(file_path)

# Log cantidad de filas
row_count = (len(data))
run.log('#filas_scoreadas', row_count)

# Nombre archivo exportación dinámico
exportid = datetime.now().strftime('%Y%m-%d%H-%M%S-') 

# Selección de columnas para exportación
#data = data.rename(columns=str.lower)
data = data[['CuentaCC','rut','m_c3','m_litros','m_meses_sin_consumo','pred','prob_y_0','prob_y_1','tipo_cliente','mes','churn_real']] 

# Loguear filas procesadas
row_count = (len(data))
run.log('#filas_exportadas', row_count)

# Guardar los datos en datalake
print("Guardando Datos...")
os.makedirs(output_dir, exist_ok=True)
model_file = os.path.join(output_dir, exportid+'data_clasificada.csv')
data.to_csv(model_file, index=False, header=True)

run.complete()

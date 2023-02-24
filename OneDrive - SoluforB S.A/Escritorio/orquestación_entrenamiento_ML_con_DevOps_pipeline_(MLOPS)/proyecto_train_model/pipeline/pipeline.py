#pipeline
##############################################################################
# Cluster de ejecución de Pipeline
##############################################################################

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
##############################################################################
# Configuración de ejecución de Pipeline Python (Runconfing)
##############################################################################

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration

from azureml.core import ScriptRunConfig
from azureml.core.runconfig import DockerConfiguration

from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Dataset, Datastore
from azureml.pipeline.steps import CommandStep
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import PipelineParameter
from azureml.data.datapath import DataPath, DataPathComputeBinding
from datetime import datetime
#Cargar cluster de ejecución
cluster_name = ev_cluster_name
pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)


# Crear ambiente de Python para ejecución de pasos Python
scoring_env = Environment.from_conda_specification(name = ev_python_env_name, file_path = './' + ev_experiment_folder_name + '/' + ev_python_env_name + '.yml')

# Registrar el ambiente
scoring_env.register(workspace=ws)
registered_env = Environment.get(ws,ev_python_env_name)

# Crear una configuración de ejecución para el pipeline
pipeline_run_config = RunConfiguration()

# Asignar el cluster a la configuración de ejecución del pipeline
pipeline_run_config.target = pipeline_cluster

# Asignar el ambiente de Python para la ejecución del pipeline
pipeline_run_config.environment = registered_env

print ("Configuración de ejecución creada.")
# Crear carpetas temporales para el traspaso de datos entre los pasos del pipeline (PipelineData)

#Traspaso PREP -> TRAINING
prepped_data_folder = PipelineData("prepped_data_folder", datastore=ws.get_default_datastore())

#Traspaso SCORING -> EXPORT
final_dir = PipelineData(name='trained_model', 
                          datastore=ws.get_default_datastore())

# Definición Datalake
datastore = Datastore.get(ws,ev_datalake_name) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< REVSISAR

# Parámetro de entrada del Pipeline con ruta y nombre dinámico para tablón de entrada
input_datapath = DataPath(datastore=datastore, path_on_datastore=ev_default_datastore_input_path)
input_path_parameter = PipelineParameter(name="input_data", default_value=input_datapath)
input_path = (input_path_parameter, DataPathComputeBinding(mode="mount"))


##############################################################################
# Definición de pasos del pipeline (pipeline steps)
##############################################################################

# Definición de Paso 1 (Preparación de datos)
prep_step = PythonScriptStep(name = "Preparación de Datos",
                                source_directory = experiment_folder,
                                script_name = "prep_data.py",
                                arguments = ['--input-data', input_path,
                                             '--prepped-data', prepped_data_folder],
                                outputs=[prepped_data_folder],
                                inputs=[input_path],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = False)

# Definición de Paso 2 (Carga de modelos predictivos)
train_step = PythonScriptStep(name = "Training Modelo",
                                source_directory = experiment_folder,
                                script_name = "train_model.py",
                                arguments = ['--trained-model', final_dir,
                                            '--prepped-data', prepped_data_folder],
                                inputs=[prepped_data_folder],
                                outputs=[final_dir],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = False)


print("Pasos Python del pipeline definidos.")

##############################################################################
# Creación del Pipeline y ejecución
##############################################################################

from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.widgets import RunDetails

# Contrucción del pipeline
pipeline_steps = [prep_step,train_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline construido.")

# Creación del pipeline y ejecución
experiment = Experiment(workspace=ws, name = ev_pipeline_exp_name)
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Ejecución del pipeline en progreso.")
RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)

##############################################################################
# Creación de Endpoint para integración de pipeline con Data Factory
##############################################################################

# Publicación del pipeline
published_pipeline = pipeline_run.publish_pipeline(
    name=ev_pipeline_name, description='Model training', version='1.0')

#Obtener información del pipeline publicado
published_pipeline

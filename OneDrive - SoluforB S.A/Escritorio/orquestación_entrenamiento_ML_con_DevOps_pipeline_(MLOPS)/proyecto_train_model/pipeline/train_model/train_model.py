#%%writefile $experiment_folder/train_model.py

##############################################################################
# Paso de entrenamiento de modelo (Training)
##############################################################################

# Importar Librerías
from azureml.core import Run, Model, Datastore
import argparse
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,  roc_auc_score, auc, roc_curve, recall_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar parámetros
parser = argparse.ArgumentParser()
parser.add_argument('--trained-model', type=str, dest='trained_model', help='Carpeta con modelo entrenado')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Carpeta con datos preparados')
args = parser.parse_args()

trained_model_dir = args.trained_model
prepped_data_dir = args.prepped_data

# Cargar el contexto de ejecución
run = Run.get_context()

# Cargar el workspace
ws = run.experiment.workspace 

##############################################################################
# Scoring
##############################################################################

# Cargar los datos
print("Cargando Datos...")
file_path = os.path.join(prepped_data_dir,'data.csv')
data = pd.read_csv(file_path)

# Log cantidad de filas
row_count = (len(data))
run.log('#filas_preparadas', row_count)

# Data split
df = data
X = df.drop(['soft_churn','soft_churn_active','hard_churn','cuentacc','rut','mes','mes_del_anio'], axis=1)
y= df['soft_churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train_index=pd.Series(X_train.index,index=X_train.index)
X_test_index=pd.Series(X_test.index,index=X_test.index)
sc=StandardScaler()
X_train = pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(sc.transform(X_test),columns=X_test.columns)
X_train.index=X_train_index
X_test.index=X_test_index

# Train
#log_reg = sm.Logit(y_train, X_train).fit()
clf1 = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
clf1.fit(X_train, y_train)
# Validate
#y_hat = log_reg.predict(X_test)
#prediction = list(map(round, y_hat))
y_pred=clf1.predict(X_test)

# Calcular metricas
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#fpr, tpr, _ = roc_curve(y_test, prediction)
#auc_score = auc(fpr, tpr)
#rec = tp / (tp+fn)
#prec = tp / (tp+fp)

fpr, tpr, _ = roc_curve(y_test, y_pred)
auc_score = auc(fpr, tpr)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
# Logear metricas
run.log('ROC AUC reentrenado', auc_score)
run.log('#True Positives reentrenado', tp)
run.log('#True Negatives reentrenado', tn)
run.log('#False Positives reentrenado', fp)
run.log('#False Negatives reentrenado', fn)
run.log('Recall reentrenado', recall)
run.log('Presicion reentrenado', precision)

sns.set(rc={'figure.figsize':(10,40)})
feature_importance = abs(clf1.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos, feature_importance[sorted_idx], align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=18)
featax.set_xlabel('Relative Feature Importance')
run.log_image('Plot Variables Importantes', plot=plt)

features = X.columns
importances = abs(clf1.coef_[0])
indices = np.argsort(importances)

# customized number 
num_features = 10 

plt.figure(figsize=(10,100))
plt.title('Feature Importances')

# only plot the customized number of features
plt.barh(range(num_features), importances[indices[-num_features:]], color='b', align='center')
plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
plt.xlabel('Relative Importance')
plt.show()

# Calcular performance del modelo vigente
nombre = 'modelo_clasificacion_fsc'
model = Model(ws, name = nombre)

path = model.download(target_dir=os.path.join(trained_model_dir,'model_fsc.pkl.previous'), exist_ok=False)
file = open(path,'rb')
log_reg_previo = pickle.load(file)
file.close()

y_hat_previo = log_reg_previo.predict(X_test)
prediction_previo = list(map(round, y_hat_previo))

tn, fp, fn, tp = confusion_matrix(y_test, prediction_previo).ravel()
fpr, tpr, _ = roc_curve(y_test, prediction_previo)
auc_score_previo = auc(fpr, tpr)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)


# Logear metricas
run.log('ROC AUC entrenado', auc_score_previo)
run.log('#True Positives entrenado', tp)
run.log('#True Negatives entrenado', tn)
run.log('#False Positives entrenado', fp)
run.log('#False Negatives entrenado', fn)
run.log('Recall entrenado', recall)
run.log('Presicion entrenado', precision)

##############################################################################
##############################################################################

# Guardar modelo
print("Guardando Modelo...")
os.makedirs(trained_model_dir, exist_ok=True)
model_file = os.path.join(trained_model_dir,'model_fsc.pkl')
pickle.dump(clf1, open(model_file, 'wb'))

if float(auc_score) > float(auc_score_previo):
    new_model = Model.register(model_path=model_file,
                            model_name=nombre,
                            tags={'training': "Script Local", 'tipo': "regresion", 'framework': 'statsmodels', 'format': 'pkl'},
                            properties={'auc': float(auc_score), 'recall': float(recall), 'presicion': float(precision)},
                            description="Modelo entrenado en Python",
                            workspace=ws)



run.complete()

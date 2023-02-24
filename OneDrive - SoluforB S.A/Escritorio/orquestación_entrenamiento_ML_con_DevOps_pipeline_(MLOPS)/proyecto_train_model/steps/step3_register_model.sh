#!/bin/bash

az config set extension.use_dynamic_install=yes_without_prompt

az account set --subscription $SUBSCRIPTION_ID
az configure --defaults workspace=$WORKSPACE group=$RESOURCE_GROUP
echo "Configure resource group and workspace ready"

# Register model OK
az ml model create --name $MODEL_NAME_0I  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_0I \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 0i registered"

# Register model OK
az ml model create --name $MODEL_NAME_1I  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_1I \
                     --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 1i registered"

# Register model OK
az ml model create --name $MODEL_NAME_2I  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_2I \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 2i registered"

# Register model OK
az ml model create --name $MODEL_NAME_3I  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_3I \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 3i registered"

# Register model OK
az ml model create --name $MODEL_NAME_4I  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_4I \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 4i registered"

# Register model OK
az ml model create --name $MODEL_NAME_5I  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_5I \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 5i registered"

# Register model OK
az ml model create --name $MODEL_NAME_1C  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_1C \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 1c registered"

# Register model OK
az ml model create --name $MODEL_NAME_2C  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_2C \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 2c registered"

# Register model OK
az ml model create --name $MODEL_NAME_3C  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_3C \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 3c registered"

# Register model OK
az ml model create --name $MODEL_NAME_4C  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_4C \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 4c registered"

# Register model OK
az ml model create --name $MODEL_NAME_5C  \
                    --type "custom_model"  \
                    --path $MODEL_PATH_5C \
                    --tags 'training:Script Local' 'tipo:classification' 'framework:R' 'formato:rds' \
                    --description "Modelo de reprobación entrenado en R utilizando Random Forest." \
                    #--set 'auc'=0.907 'recall'=0.907 'precision'=0.85 \
                    --version 1  \
                    --resource-group $RESOURCE_GROUP  \
                    --workspace-name $WORKSPACE

echo "Model 5c registered"

#!/bin/bash
az config set extension.use_dynamic_install=yes_without_prompt

az account set --subscription $SUBSCRIPTION_ID
az configure --defaults workspace=$WORKSPACE group=$RESOURCE_GROUP
echo "Configure resource group and workspace ready"

# Create container
az storage container create \
    --name $CONTAINER_NAME \
    --account-name $STORAGE_ACCOUNT \
    --account-key $STORAGE_KEY \
    --auth-mode key

echo "Container created"

# Create directory
az storage blob directory create --container-name $CONTAINER_NAME \
                                 --directory-path $DIRECTORY_PROYECTO \
                                 --account-name $STORAGE_ACCOUNT \
                                 --account-key $STORAGE_KEY \
                                 --auth-mode key     
echo "Directory project created"
sleep 30 

az storage blob directory create --container-name $CONTAINER_NAME \
                                 --directory-path $DIRECTORY_PROYECTO_INPUT \
                                 --account-name $STORAGE_ACCOUNT \
                                 --account-key $STORAGE_KEY \
                                 --auth-mode key    
echo "Directory input created"
sleep 30 

az storage blob directory create --container-name $CONTAINER_NAME \
                                 --directory-path $DIRECTORY_PROYECTO_OUTPUT \
                                 --account-name $STORAGE_ACCOUNT \
                                 --account-key $STORAGE_KEY \
                                 --auth-mode key       
echo "Directory output created"
sleep 30 


# Create datastore
az ml datastore create --file $DATASTORE_PATH --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE
echo "Datastore created"
sleep 30 


az storage blob directory upload --container $CONTAINER_NAME \
                                 --destination-path $DIRECTORY_PROYECTO_INPUT \
                                 --source $PATH_DATA \
                                 --account-key $STORAGE_KEY \
                                 --account-name $STORAGE_ACCOUNT
echo "Uploaded file1"
#!/bin/bash

az extension add -n ml -y
az config set extension.use_dynamic_install=yes_without_prompt

# Configure resource group and workspace in production OK
az configure --defaults workspace=$WORKSPACE group=$RESOURCE_GROUP location=$LOCATION
echo "Configure resource group and workspace ready"

# Create Compute
az ml compute create -n $CLUSTER_NAME --size $CLUSTER_SIZE --type amlcompute --min-instances 0 --max-instances 5
echo "Compute Cluster Ready"
sleep 30 

az account set --subscription $SUBSCRIPTION_ID
az configure --defaults workspace=$WORKSPACE group=$RESOURCE_GROUP

# Create evironment
az ml environment create --file $ENV_PATH  \
                            --name $ENV_NAME \
                            --resource-group $RESOURCE_GROUP  \
                            --workspace-name $WORKSPACE

echo "Environment created"

az ml environment create --name $ENV_NAME_R \
                            --file $ENV_PATH_R  \
                            --dockerfile-path $ENV_PATH_IMAGE \
                            --resource-group $RESOURCE_GROUP \
                            --workspace-name $WORKSPACE

echo "Environment R created"
#!/bin/bash

pip install mldesigner
pip install mltable
pip install pandas
pip install azure-ai-ml
#pip list

az extension add -n ml -y
az config set extension.use_dynamic_install=yes_without_prompt

# Configure resource group and workspace in production OK
az account set --subscription $SUBSCRIPTION_ID
az configure --defaults workspace=$WORKSPACE group=$RESOURCE_GROUP location=$LOCATION
echo "Configure resource group and workspace ready"
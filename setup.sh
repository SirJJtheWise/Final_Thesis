#!/bin/bash

# Define the name for the new environment
NEW_ENV_NAME="thesis"

# Check if the environment already exists
if conda env list | grep -q "$NEW_ENV_NAME"; then
    echo "Environment '$NEW_ENV_NAME' already exists."
    exit 1
fi

# Create the new environment from the YAML file
conda env create --name "$NEW_ENV_NAME" --file env.yml

# Activate the new environment
source activate "$NEW_ENV_NAME"

# Verify the environment creation
if [ $? -eq 0 ]; then
    echo "Environment '$NEW_ENV_NAME' created and activated successfully."
else
    echo "Failed to create the environment '$NEW_ENV_NAME'."
fi

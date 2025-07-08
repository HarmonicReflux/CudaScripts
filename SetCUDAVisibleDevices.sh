#!/bin/bash

# **********************************************************************************************************
# Script to set the CUDA_VISIBLE_DEVICES environment variable
# Usage: Source this script to set the visible CUDA devices for the current shell session.
#
# Author: HarmonicReflux
# Date: 08 July 2025
# **********************************************************************************************************

# Ensure that the script is sourced, not executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please source this script instead of executing it directly."
    echo "Usage: source ./SetCUDAVisibleDevices.sh"
    exit 1
fi

# Check if CUDA is installed by verifying nvcc
if ! command -v nvcc &>/dev/null; then
    echo "CUDA toolkit is not installed or not found in the PATH. Exiting."
    exit 1
fi

# Get the number of visible CUDA devices (available GPUs)
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ "$num_gpus" -eq 0 ]; then
    echo "No NVIDIA GPUs found! Exiting."
    exit 1
fi

# Log the detected GPUs
echo "Found $num_gpus NVIDIA GPUs."

# Set CUDA_VISIBLE_DEVICES to include both GPUs (0 and 1)
export CUDA_VISIBLE_DEVICES=0,1

# Check if the environment variable was successfully set
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Error: Failed to set CUDA_VISIBLE_DEVICES."
    exit 1
else
    echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
fi

# Display a message indicating that the script has completed successfully
echo "CUDA environment has been configured. You can now run CUDA applications on the specified GPUs."

# End of script

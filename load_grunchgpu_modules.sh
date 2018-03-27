#!/bin/bash
set -e

echo "Loading modules for GPU-enabled TensorFlow."

module load CUDA/8.0.44-foss-2016a 
module load cuDNN/5.1-foss-2016a-CUDA-8.0.44 
module load GCC/5.4.0-2.26

echo "Done."


#!/bin/bash
set -e

############################################
# CONFIGURATION
############################################
# Override these from CLI as needed:
#   ./run_embedding_slim_cuda.sh MODEL_PATH=... TASK_IDS="3 4" MAX_LENGTH=256

MODEL_PATH="./bge-base-en-v1.5"    # Local path or HF hub name
OUTPUT_DIR="./bge_emb_config"      # Output directory
TASK_IDS="3"                       # Space-separated list of task IDs to process
PRECISION="fp16"                   # fp16 or bf16
MAX_LENGTH=512                     # Max tokens
BATCH_SIZE=""                      # Leave empty for auto
BASE_DIR="."                       # Base directory for script
CLEAN_TEMP="True"                  # Clean temporary files

############################################
# APPLY CLI OVERRIDES
############################################
for arg in "$@"; do
    eval "$arg"
done

############################################
# EXPORT TO PYTHON ENV
############################################
export MODEL_PATH OUTPUT_DIR TASK_IDS PRECISION MAX_LENGTH BATCH_SIZE BASE_DIR CLEAN_TEMP

############################################
# RUN
############################################
echo "🚀 Running embedding generation with:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  TASK_IDS=$TASK_IDS"
echo "  PRECISION=$PRECISION"
echo "  MAX_LENGTH=$MAX_LENGTH"
echo "  BATCH_SIZE=${BATCH_SIZE:-auto}"
echo "  OUTPUT_DIR=$OUTPUT_DIR"

python3 embedding-slim-cuda-config.py


############################################
# bash extention/run_embedding_slim_cuda.sh MODEL_PATH="BAAI/bge-small-en-v1.5" TASK_IDS="3 4" PRECISION="bf16"
# bash extention/run_embedding_slim_cuda.sh MAX_LENGTH=256 BATCH_SIZE=1024
# bash extention/run_embedding_slim_cuda.sh
############################################

#!/bin/bash
#
# Save & cleanup script for PPlug Slim training artifacts
# Usage:
#   bash save_and_cleanup_pplug_slim.sh <task_id> [--delete]
# Example:
#   bash save_and_cleanup_plug_slim.sh 3
#   bash save_and_cleanup_pplug_slim.sh 3 --delete

TASK_ID=$1
DELETE_FLAG=$2

if [ -z "$TASK_ID" ]; then
  echo "❌ ERROR: Please provide task_id"
  echo "Usage: bash save_and_cleanup_pplug_slim.sh <task_id> [--delete]"
  exit 1
fi

# --------------------------
# Paths to essential artifacts
# --------------------------
OUTPUT_DIR="output_${TASK_ID}"
BGE_DIR="./bge_emb"
BGE_FILES="task_${TASK_ID}_train_bge.emb task_${TASK_ID}_dev_bge.emb"
FLAN_DIR="../FlanT5-small"   # adjust if you used base/large, or swap with HF path mirror

ARCHIVE_NAME="pplug_slim_task${TASK_ID}_artifacts.tar.gz"

echo "📦 Collecting artifacts for task_id=${TASK_ID} ..."

# Check required items exist before zipping
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "❌ ERROR: $OUTPUT_DIR not found."
  exit 1
fi

for f in $BGE_FILES; do
  if [ ! -f "${BGE_DIR}/${f}" ]; then
    echo "❌ ERROR: ${BGE_DIR}/${f} not found."
    exit 1
  fi
done

if [ ! -d "$FLAN_DIR" ]; then
  echo "⚠ WARNING: Base model dir ($FLAN_DIR) not found — ensure you can re-download from HF."
fi

# --------------------------
# Create archive
# --------------------------
tar -czvf "$ARCHIVE_NAME" \
  "$OUTPUT_DIR" \
  ${BGE_DIR}/task_${TASK_ID}_train_bge.emb \
  ${BGE_DIR}/task_${TASK_ID}_dev_bge.emb \
  "$FLAN_DIR" \
  output.log

echo "✅ Archive created: $ARCHIVE_NAME"
echo "💡 You can download it from your cloud with: scp or rsync"

# --------------------------
# Delete local artifacts (optional)
# --------------------------
if [ "$DELETE_FLAG" == "--delete" ]; then
  echo "🗑  Deleting local artifacts for task_id=${TASK_ID} ..."
  rm -rf "$OUTPUT_DIR"
  rm -f ${BGE_DIR}/task_${TASK_ID}_train_bge.emb
  rm -f ${BGE_DIR}/task_${TASK_ID}_dev_bge.emb
  rm -rf "$FLAN_DIR"
  echo "✅ Local artifacts deleted."
fi

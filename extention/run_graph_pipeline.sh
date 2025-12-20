
#!/bin/bash

# Exit immediately if any command fails
set -e

echo "=== Stage 1: Embedding Chunk 0 ==="
python compute_graph_emb_generic.py --stage embed --chunk-index 0 --num-chunks 2

echo "=== Stage 2: Embedding Chunk 1 ==="
python compute_graph_emb_generic.py --stage embed --chunk-index 1 --num-chunks 2

echo "Stage 3: merge all partial embeddings into final file"
python compute_graph_emb_generic.py --stage merge

echo "=== Stage 4: train the GNNs ==="
python compute_graph_emb_generic.py --stage train

echo "=== Stage 5: Infer the GNNs ==="
python compute_graph_emb_generic.py --stage infer

echo "✅ Pipeline completed successfully."

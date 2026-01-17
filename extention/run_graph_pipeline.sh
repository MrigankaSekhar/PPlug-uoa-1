
#!/bin/bash

# Exit immediately if any command fails
set -e


echo "=== Stage 0: Build Graph Once ==="
python compute_graph_emb_generic_npy.py --stage build

echo "=== Stage 1: Embedding Chunk 0 ==="
python compute_graph_emb_generic_npy.py --stage embed --chunk-index 0 --num-chunks 2 

echo "=== Stage 2: Embedding Chunk 1 ==="
python compute_graph_emb_generic_npy.py --stage embed --chunk-index 1 --num-chunks 2 

echo "Stage 3: merge all partial embeddings into final file"
python compute_graph_emb_generic_npy.py --stage merge 

echo "=== Stage 4: train the GNNs ==="
python compute_graph_emb_generic_npy.py --stage train 

echo "=== Stage 5: Infer the GNNs ==="
python compute_graph_emb_generic_npy.py --stage infer 

# echo "=== Stage 6: compute metrics ==="
python compute_graph_emb_generic_npy.py --stage metrics
# echo "=== Stage 5: evaluate the GNNs ==="
# python compute_graph_emb_generic_npy.py --stage evaluate --use-subset --rebuild-pairs

# echo "=== Stage 5: check his_id alignment ==="
# python ../scripts/check_id_alignment.py 

echo "✅ Pipeline completed successfully."

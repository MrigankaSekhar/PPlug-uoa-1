1. **Environment Setup**  
   Use conda and the `environment.yml` file to create the required environment for PPlug‑Slim and its GNN extensions.  
   ```bash
   conda env create -f environment.yml
   conda activate pplug-env
   ```
   🧩 *Tip:* You might get errors installing **torch‑geometric** dependencies via conda.  
   In that case, install them manually using pip:
   ```bash
   pip install torch-sparse==0.6.17 \
     --no-build-isolation \
     -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
   ```
   This ensures PyTorch Geometric libraries (scatter, sparse, cluster, etc.) align with your Torch build.

---

2. **Model and Embeddings Download**  
   Run the following script to download required pretrained models and embedders:
   ```bash
   bash setup_download.sh
   ```
   This sets up:
   - **Flan‑T5** models (`FlanT5-small`, `FlanT5-base`, large or XXL)
   - **bge-base-en-v1.5** embedder for computing semantic embeddings used in personalization.

---

3. **Data Preparation and Subsetting**  
   To create smaller, development-size subsets of the full LaMP dataset:
   ```bash
   python create_subset_data.py
   ```
   Run this before full experiments for debugging or lightweight local runs.

---

4. **Generate Persona‑Aware Training Data**  
   Build the question–answer training corpus for T5 with:
   ```bash
   python aggr_id-slim.py
   ```
   This script merges input/output JSONs and appends persona IDs (`his_id`) with each example so that the model can associate user history with questions.

---

5. **Compute Embeddings (BGE)**  
   Create dense historical embeddings using:
   ```bash
   python embedding-slim-mac-mmap.py --dataset-root ./LaMP_time_3/ --batch-size 4 --device cpu
   ```
   The output includes:
   - `task_<id>_train_bge.memmap.npy` → long‑term profile embeddings  
   - `task_<id>_dev_bge.memmap.npy` → dev/test profile embeddings  
   - Each record corresponds to one review, profile item, or history element.
   - `task_<id>_<split>_metric.json` → distribution of the embeddings
   - `task_<id>_<split>_offsets.json` → is a versy important mapping file which is used when building and training the graph

---

6. **Graph Pipeline (GNN Training)**  
   Run:
   ```bash
   bash run_graph_pipeline.sh
   ```
   This creates the **Neo4j‑style relational graph** between users, items, and reviews, trains a simple GNN encoder, and stores outputs under:
   ```
   graph_emb/
       ├── task_3_graph.npy
       ├── task_3_his_to_graph.json
   ```
   The graph embeddings allow relational reasoning across similar users or items.

---

7. **Graph Structure Validation**  
   After training, validate your graph:
   ```bash
   python check_graph_chunk.py
   ```
   This script checks edge connectivity and embeddings per chunk, ensuring all graph nodes were properly written.

---

8. **Run Full Training Pipeline (T5‑Slim‑GNN)**  
   ```bash
   bash run_all_t5-slim-GNN-mac.sh
   ```
   This executes the full personalized pipeline:
   - Loads the persona dataset  
   - Initializes `ModelForPer_slim_GNN`  
   - Trains over both textual and graph embeddings  
   - Logs metrics to `extention/output_<task_id>/`
   - The checkpoint files (which is the trained weights) are created in output3/checkpoint-<num>

---

9. **Files of Importance**
   ```text
   a. aggr_id-slim.py             → Build persona‑augmented question–answer pairs  
   b. embedding-slim-mac-mmap.py  → Compute historical embeddings in BGE space  
   c. run_graph_pipeline.sh       → Create/train GNN embeddings and relational graph  
   d. run_all_t5-slim-GNN-mac.sh  → End‑to‑end T5 + GNN training and evaluation pipeline  
   ```
   **In-depth Architecture Files**
   - `PersonalDataset_profile_GNN.py` → Defines personalized dataset loader emitting profile, session, and graph node IDs.  
     Handles both long‑term (`his_id`) and short‑term (`session_ids`) personalization with correct padding and mapping.  
   - `ModelForPer_slim_GNN.py` → Full architecture integrating text encoder (Flan‑T5), BGE embedder, session transformer, and graph embeddings.  
     Implements **gated cross‑attention**, allowing controlled fusion between task and personalization signals.  
   - `main_profile-slim-GNN.py` → Training/evaluation entry point that ties dataset, model, tokenizer, optimizer, and logging together into a single runnable pipeline.

---

10. **Check Personalization Effects and Diagnostics**  
   After training, use these scripts to inspect how personalization modifies model behavior:

   - `compare_personalization_effects.py`  
     Performs a **side‑by‑side comparison** between:
     1. Flan‑T5 baseline (no personalization)  
     2. Personalized model (profile embeddings only)  
     3. Personalized + Graph‑enhanced model (profile + GNN embeddings)  
     It prints generated outputs for each case and monitors **gate activation** values, helping visualize how much personalization influences the output text.  
     Typical metrics shown:
     - Text differences between Base and Personalized Output  
     - `gate_avg` ≈ 0.5 ⇒ balanced fusion; > 0.8 ⇒ strong personalization

   - `test_query_single_persona.py`  
     Runs quantitative evaluation comparing predictions with persona vs. without persona (empty embeddings) and against plain T5 baseline.  
     Outputs include:
     - Textual accuracy and numeric F‑score or RMSE  
     - Gate activation diagnostics  
     - Embedding cosine similarities (real vs generic vs empty personas)  
     - Confirms model learns distinct, consistent behavioral patterns per persona.

   Together, these scripts **demonstrate and measure personalization strength**, validating whether persona and graph signals genuinely impact model output semantics and accuracy.

---

✅ **Complete Workflow Summary**  
1️⃣ Setup environment → 2️⃣ Download models → 3️⃣ Prepare data → 4️⃣ Compute embeddings →  
5️⃣ Build graph → 6️⃣ Train model → 7️⃣ Validate graph → 8️⃣ Run pipeline → 9️⃣ Analyze personalization → 🔟 Compare personalization and graph effects.  
This end‑to‑end process showcases how PPlug‑Slim‑GNN evolves from base text modeling into contextual and personalized LLM reasoning.

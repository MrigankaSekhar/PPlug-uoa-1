```markdown
# 🚀 PPlug Slim Variant — Quick Start Guide

This guide walks you through running the **Slim** variant of PPlug for development or resource‑friendly experimentation.

---

## 1️⃣ Update & Create Conda Environment
Make sure you are in the project root directory.

```bash
conda env update -f environment.yml
```

---

## 2️⃣ Activate the Environment
```bash
conda activate cloudspace
```

---

## 3️⃣ Verify Python Version (Must Be 3.9)
```bash
python --version
```
You should see:
```
Python 3.9.x
```

---

## 4️⃣ Verify PyTorch Installation & GPU
```bash
python scripts/verify_torch.py
```
This script checks:
- PyTorch version
- CUDA availability
- GPU name

---

## 5️⃣ Download Required Models & Datasets
```bash
./setup_download.sh
```
This will ensure:
- **LaMP‑3 dataset** is downloaded
- **Flan‑T5 model** (small/base/large depending on path) is present
- **BGE embedding model** exists

---

## 6️⃣ Aggregate Dataset IDs
```bash
python aggr_id-slim.py
```
This merges LaMP question & output files, preparing them for slim runs.

---

## 7️⃣ Create Historical Embeddings (Slim CUDA Version)
```bash
python embedding-slim-cuda.py
```
This precomputes user history embeddings using the BGE model for faster training.

---

## 8️⃣ Run All Slim T5 Training Tasks
```bash
bash run_all_t5-slim.sh
```
This will:
- Train the Slim PersonalLLM model for all task IDs (LaMP variants)
- Log results to `output_slim_<task_id>.log`

---

## ℹ️ Notes
- **Model Choice**: In `run_all_t5-slim.sh`, you can quickly change:
  ```bash
  --model_path google/flan-t5-small
  ```
  to `flan-t5-base` or a downloaded local path.
- **Metrics**: For LaMP‑3, evaluation outputs `MAE` and `RMSE`.
- **Frozen LLM**: The Slim model freezes all Flan‑T5 weights, training only adapters + embedding alignment layers.

---


## **1. Overview**
- **`/code/` folder** → *Baseline PPlug (Slim)*  
  * Only **long‑term personalization** using **precomputed BGE embeddings** of user history (profile).
  * Freezes entire LLM (Flan‑T5) — only the small MLP alignment layers + `inst_token` are trained.
  * Profile embeddings are injected directly into the frozen LLM input embedding space.

- **`/extention/` folder** → *Extended PPlug (Slim‑GNN)*  
  * Keeps the baseline pipeline but **adds two more personalization sources**:
    1. **Short‑term session embeddings** (recent user events → Transformer)
    2. **Graph embeddings** (relational context from Neo4j / PyG graph)
  * **Uses Gated Cross‑Attention** instead of naive concatenation to fuse personalization with the LLM task embeddings.

---

## **2. Key Architectural Differences**
Based on `/extention/architecture/layers.md` & `/extention/readme.md`:

| Component | Baseline `/code` | Extended `/extention` | Why It Matters |
|-----------|------------------|-----------------------|----------------|
| **LLM Backbone** | Flan‑T5 frozen, adapter injection | Same (Flan‑T5 frozen) | No change in core text generation LLM — compute cost is same. |
| **Long‑term Personalization (`profile_embs`)** | ✅ Yes — precomputed history embeddings from BGE (`his_train_emb_table`) | ✅ Same | Both use precomputed stable embeddings for user profile. |
| **Short‑term Personalization (`session_embs`)** | ❌ No | ✅ Session‑Aware Transformer encodes last N clicks/interactions | Adds immediate context — helps in tasks where user’s current interest deviates from long‑term average (e.g., trending products). |
| **Relational Personalization (`graph_embs`)** | ❌ No | ✅ Precomputed GNN node embeddings from Neo4j graph | Captures collaborative filtering signals & relationships between users and items — useful when history is sparse. |
| **Fusion Method** | Concatenate profile_embs into task embedding | **Gated Cross‑Attention** between task_embs and `[profile_embs, session_embs, graph_embs]` | Instead of fixed addition/concat, learns **how much personalization to inject** dynamically per token. |
| **Flexibility to Missing Data** | If profile missing → zeros | If session/graph missing → can mask them in attention | New arch gracefully degrades when new personalization components aren’t available. |

---

## **3. Why the Extension is More Useful**
### **(A) More Personalization Signals**
Baseline is **one‑dimensional personalization**: profile history.  
Extension is **multi‑source personalization**:  
- Long‑term memory (stable profile)
- Short‑term behaviour (session Transformer)
- Social/relational knowledge (GNN from Neo4j)

This is important for **LaMP‑3** because:
- Ratings can depend on both *long‑term taste* (profile) and *momentary interest* (session).
- Graph relationships add collaborative context even if explicit history is short.

---

### **(B) Smarter Fusion**
Baseline = “dump the profile vector into the LLM input” — personalization is either fully on or off.  
Extension = “calculate per‑token, per‑example **how much personalization to mix in**” via a **gate**.  
This is research‑relevant because you can:
- Quantitatively show the gating behaves differently for different tasks.
- Avoid over‑personalization on irrelevant queries.

---

### **(C) Still Low Computational Cost**
- LLM still frozen, so the biggest computation (Flan‑T5 forward) is unchanged.
- All added components (session transformer, graph embedding lookup, attention layer) are **lightweight** compared to training large transformer blocks.
- Means you can **compare baseline vs extension on the same GPU** easily.

---

## **4. Expected Gains for LaMP‑3**
📈  Where it should help:
- **Cold start**: few/no profile items → graph embeddings can fill gaps.
- **Shift in interest**: session embeddings steer away from stale profile patterns.
- **Highly relational domains**: product ratings influenced by similar users/items in graph.

📉  Where gains might be minimal:
- If user behaviour is very consistent over time → profile only may be enough.
- If session & graph data are noisy or incomplete — fusion may add noise instead.

---

## **5. Academic Value**
From an academic perspective, this extension is a **novel multi‑signal personalization plug‑in** that:
- Keeps the large generative LLM fixed, ensuring efficiency.
- Integrates multiple embedding spaces into a **learnable fusion module**.
- Is **modular** — you can independently ablate session, graph, or gate to study their impact.
- Gives an explainable hook: you can inspect gate values to see when personalization helps or hurts.

This makes it **publishable‑level** if you run:
- **Ablation study** (Baseline / +Session / +Graph / +Both)
- **Gate behaviour analysis**
- **Performance boost on MAE / RMSE**

---

✅ **Conclusion**:  
The `/extention` architecture is a **strict superset** of `/code` — it adds richer context and smarter fusion without increasing LLM training cost.  
It is particularly **useful academically** because:
1. It allows a clean, fair side‑by‑side comparison with the baseline.
2. It’s modular — enabling detailed ablation experiments.
3. It makes the personalization process more interpretable and flexible.

---


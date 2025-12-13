***

### **Stage A: Context Encoders**

**Goal:** Capture user-specific and task-specific context to create rich semantic embeddings.

1.  **Long-term User Behaviour Encoder**
    *   Encodes historical user interactions (long-term patterns).
    *   Captures preferences, habits, and recurring behaviors.
    *   Outputs embeddings representing global user tendencies.

2.  **Session-aware Encoder**
    *   Focuses on recent sessions or short-term context.
    *   Captures temporal dynamics and session-specific signals.
    *   Complements long-term behavior with recency bias.

3.  **Transformer Encoder (New)**
    *   Adds contextual attention across user history and sessions.
    *   Handles sequential dependencies and complex relationships.
    *   Improves representation quality for personalization.

4.  **Graph Embedding Encoder (New)**
    *   Converts graph schema (e.g., user-item relationships, social graph) into embeddings.
    *   Captures structural and relational context beyond sequences.
    *   Useful for personalization in multi-entity environments.

5.  **User Input Encoder**
    *   Encodes the current task input (query, instruction, or prompt).
    *   Produces instruction token embeddings for fusion with user context.

**Output of Stage A:**

*   **Consolidated embeddings** (semantic context from A+B+C).
*   **Instruction token embeddings** (from task input).

***

### **Stage B: Aggregation and Fusion**

**Goal:** Combine user context and task input into a unified personalized representation.

1.  **Input-aware Personal Aggregator**
    *   Aligns user context embeddings with task input embeddings.
    *   Ensures personalization is relevant to the current task.
    *   Handles weighting between long-term, short-term, and input signals.

2.  **Gated Cross Attention Fusion (New)**
    *   Applies attention mechanism between user context and task input.
    *   Uses gating to control influence of different sources (e.g., history vs input).
    *   Outputs **personalized embeddings** ready for LLM injection.

**Output of Stage B:**

*   **Personalized embeddings** (rich, task-aware, user-specific).

***

### **Stage C: LLM Processing**

**Goal:** Inject personalized embeddings into the LLM for final output generation.

1.  **LLM Input Embedding Layer**
    *   Combines original input embeddings with personalized embeddings.
    *   Maintains tunable vs fixed components for efficient personalization.

2.  **Large Language Model**
    *   Generates personalized output using fused embeddings.
    *   Can adapt tone, content, or recommendations based on personalization signals.

***


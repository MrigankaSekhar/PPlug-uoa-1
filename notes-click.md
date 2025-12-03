# notes

three set of approaches as the original paper mentions
1. train LLM / Model with each users historical data , this is not a practical approach , as it becomes very expensive and can't scale as the historical data changes
2. RAG based personalisation , more practical, and used by most , user history and preferences are retrieved as demonstrations in the prompts for the model . its is plug and play and no model tuning is needed , however it breaks the continuity of the use history . One can argue that we can make this continuous but feed clickstream to the RAG DB , however it still would be expensive with regrads to Token size sent , it would still be very content relevant and not behavioural pattern . scalability and latency will still be a issue , single vector v/s text stream .
3. Plug-in embedding approach : suggests to have a users specific embedding for all historical context , and attach this embedding to the LLM so it can condition user habits and preferences

| Aspect | Retrieval + Clickstream | PPlug (Embedding-Based) |
| ---| ---| --- |
| Continuity | Provides point-in-time context only; fragments long-term history | Aggregates entire user history into a unified embedding |
| Style & Preference Modeling | Focuses on content relevance; misses global stylistic patterns | Captures tone, phrasing, and behavioral patterns holistically |
| Efficiency | Requires repeated retrieval and prompt concatenation → higher latency and token cost | Fixed-size embedding attached to input → minimal overhead |
| Scalability | Retrieval grows with user history size; managing large clickstreams is expensive | Embedding compresses history into a single vector → scalable |
| Robustness | Sensitive to retrieval quality; missing relevant texts hurts personalization | Embedding is comprehensive and less error-prone |

![](https://t90161334763.p.clickup-attachments.com/t90161334763/4c49a249-9a64-40ae-9cb3-988c96757cd0/image.png)

changes to the architecture
**Gated Cross-Attention Layer**
*   **Placement**: Between **Input-aware Personal Aggregator** and the final personalized embedding output.
*   **Purpose**: Fuse hierarchical linguistic embeddings (style) with behavioral embeddings dynamically, rather than simple concatenation.
*   **Sub notes:**
    *   **Cross-Attention**:
    *   It attends to two different sources of information:
        *   **User embeddings** (behavioral + linguistic + graph signals)
        *   **Task input embeddings** (from the LLM input encoder)
    *   This allows the model to learn how much each source should influence the final representation.
    *   **Gated**:
    *   A **gate mechanism** controls the flow of information from each source.
    *   Think of it as a learnable weight that decides:
        *   When to prioritize user style vs task semantics
        *   When to suppress irrelevant personalization signals
    *   This prevents over-personalization or irrelevant bias.
    *   **Example for this layer :**
    *   When the user types **“find me protein bars”** in an e-commerce search scenario, that query is converted into **task input embeddings** by the **LLM Input Encoder** in your architecture diagram.
    *   Here’s how it fits into the pipeline:
    *   **Task Input Embeddings** represent the **semantic meaning of the current query** (e.g., “find me protein bars” → intent: search for protein bars).
    *   These embeddings are then combined with **personalized embeddings** (from user history, session, and linguistic style) using the **Input-aware Personal Aggregator** or the proposed **Gated Cross-Attention Layer**.
    *   The fusion ensures that the LLM understands both:
        *   **What the user wants now** (protein bars).
        *   **How the user typically interacts** (preferences like vegan, low sugar, conversational tone).
    *   So in **Gated Cross-Attention Layer**, the **query embedding (X)** would be the representation of “find me protein bars,” and the **user embedding (U)** would encode long-term preferences and style. The gate then decides how much personalization to inject into the response or recommendation.
    *   
**Graph Neural Network (GNN) Layer**
*   **Placement**: Parallel to **User Behavior Encoder**, operating on the Neo4j graph.
*   **Purpose**: Generate graph-aware embeddings (user-item-review relationships) and feed them into the aggregator.
*   **Sub notes:**
    
    Here’s the same style of explanation for the **Graph Neural Network (GNN) Layer**:
    
    **Why is it called a Graph Neural Network Layer?**
    
    *   **Graph Structure**:
        *   Unlike flat embeddings, user-item-review relationships form a **graph** (nodes = users, items, reviews; edges = interactions like RATED, VIEWED).
    *   **Neural Network on Graphs**:
        *   A GNN propagates information across nodes and edges, learning **contextual embeddings** that capture multi-hop relationships (e.g., “users similar to you liked vegan protein bars”).
    *   **Why It Matters**:
        *   Traditional embeddings treat users/items independently.
        *   GNN learns **relational context** → better personalization and explainability.
    
    **How to Implement It**
    
    **Inputs**:
    
    *   Graph schema from LaMP-3:
        *   Nodes: `User`, `Item`, `Review`, `Category`
        *   Edges: `:RATED`, `:WROTE`, `:DESCRIBES`, `:BELONGS_TO`
    
    **Steps**:
    
    1. **Graph Construction**:
        *   Store in Neo4j or PyTorch Geometric format.
        *   Each node has initial features (static embeddings, metadata).
    2. **Message Passing**:
        *   For each node, aggregate features from neighbors: $ h\_v^{(k+1)} = \\sigma \\Big( W \\cdot \\text{AGG}({ h\_u^{(k)} : u \\in \\mathcal{N}(v) }) \\Big) $
        *   Where `AGG` = mean/sum/max, `σ` = activation (ReLU).
    3. **GraphSAGE or GAT**:
        *   **GraphSAGE**: Samples neighbors and aggregates.
        *   **GAT (Graph Attention Network)**: Learns attention weights for edges.
    4. **Output**:
        *   Node embeddings enriched with relational context.
        *   Feed these embeddings into the **Input-aware Personal Aggregator** or fuse with static/session embeddings.
    
    **Integration Point in Your Diagram**
    
    *   **Parallel to User Behavior Encoder**:
        *   GNN runs on the graph and outputs **graph-aware embeddings**.
    *   These embeddings join static + session + linguistic embeddings in the aggregator (or in the Gated Cross-Attention layer).
    
      
    
**Session-Aware Transformer Block**
*   **Placement**: After **Input Encoder**, before aggregation.
*   **Purpose**: Model short-term intent from clickstream or recent interactions using attention over session data.
*   **Sub Notes**:
    
    **Why is it called a Session-Aware Transformer Block?**
    
    *   **Session-Aware**:
        *   It focuses on **recent user interactions** (clickstream, last N searches, last few purchases) to model **short-term intent**.
        *   This is critical because user intent can shift rapidly (e.g., from “vegan recipes” to “protein bars”).
    *   **Transformer Block**:
        *   Uses **self-attention** to capture dependencies among recent actions and their temporal order.
        *   Unlike static embeddings, this block dynamically updates personalization signals based on the latest session context.
    
    **How to Implement It**
    
    **Inputs**:
    
    *   A sequence of recent interactions:
        *   Example: `[“viewed vegan protein powder”, “searched protein bars”, “added whey protein to cart”]`
    *   Each interaction is converted into an embedding (using SentenceTransformers or product metadata).
    
    **Steps**:
    
    1. **Positional Encoding**:
        *   Add time-aware positional embeddings to preserve order and recency.
    2. **Transformer Encoder**:
        *   Apply multi-head self-attention: $ \\text{Attention}(Q,K,V) = \\text{softmax}\\Big(\\frac{QK^T}{\\sqrt{d\_k}}\\Big)V $
        *   Captures relationships between recent actions (e.g., “search protein bars” is more relevant than “viewed vegan powder”).
    3. **Session Representation**:
        *   Aggregate the output (e.g., take the last token or mean-pool) to form a **session embedding**.
    4. **Fusion**:
        *   Combine session embedding with static and graph embeddings using:
            *   Weighted sum (α·static + β·session)
            *   Or **Gated Cross-Attention** for dynamic fusion.
    
    **Integration Point in Architecture**
    
    *   **Placement**:
        *   After **Input Encoder**, before the **Input-aware Personal Aggregator**.
    *   **Role**:
        *   Provides **real-time personalization signal** that complements long-term history and linguistic style.
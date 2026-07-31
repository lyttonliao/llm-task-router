# Vector Databases: How They Work, What You Built, How to Extend

## Part 1: How Vector Databases Work (The Fundamentals)

A vector database solves a different problem than traditional SQL: **"Find rows similar to this new thing"** instead of **"Find rows where column X equals Y."**

### The core pipeline:

1. **Embedding: text → dense vector**
   Your code does this with sentence-transformers:
   ```python
   # embeddings.py
   model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim model
   vector = model.encode("rewrite this React component")  # [0.123, -0.456, ...]
   ```
   The model converts language into a 384-dimensional space where semantically similar text clusters together. `"debug a crash"` and `"investigate an error"` land near each other.

2. **Storage: index the vectors for fast lookup**
   pgvector stores them in Postgres with an HNSW (Hierarchical Navigable Small World) index:
   ```sql
   CREATE INDEX routing_examples_embedding_hnsw
       ON routing_examples USING hnsw (embedding vector_cosine_ops);
   ```
   HNSW is a graph-based approximate nearest-neighbor (ANN) algorithm. Instead of scanning all 1M vectors linearly (exact but slow), it navigates a probabilistic graph to find close neighbors in log(n) hops. Tradeoff: it's approximate (might miss the true #1 closest), but fast.

3. **Query: find neighbors + use them for inference**
   When a new task arrives:
   ```python
   # tier2_classifier.py
   matches = vector_store.nearest_neighbors(embedding, "task_type", k=5)
   # Returns 5 closest neighbors + their similarity scores
   ```
   Then vote:
   ```python
   majority = _nn_majority(matches)  # 4/5 neighbors say "CODE_GEN"? Confident.
   ```

### Why this beats a similarity-threshold heuristic:

Your code discovered this through validation (see the AGREEMENT_THRESHOLD comment in tier2_classifier.py). Raw cosine similarity is noisy at small data scales—median similarity for same-label pairs (0.32) overlaps with cross-label pairs (0.29). But **relative agreement** is signal: voting among 5 neighbors gives 72% baseline accuracy, rising to 87% at >=4/5 agreement. That's why you check agreement fraction, not raw similarity.

---

## Part 2: Your Architecture (What You Built)

Your system is a **two-tier continuous-learning classifier**. Tier 2 uses the vector store as its foundation:

### The flow:

**Incoming task:** `"fix a data corruption bug in production"`

```
1. Embed it locally (sentence-transformers, ~50ms)
   embedding = [0.234, -0.567, ...]  # 384 dims

2. Tier-2 NN lookup (pgvector cosine search, ~1-5ms with HNSW)
   neighbors = nearest_neighbors(embedding, "task_type", k=5)
   # Returns: [(TRIAGE, sim=0.39), (TRIAGE, sim=0.38), (CODE_GEN, sim=0.29), ...]

3. Majority vote
   if 4/5+ agree on TRIAGE → confident, return NN result
   else → not confident

4. LLM fallback (on tier-1 unconfident)
   call haiku: "classify this: fix a data corruption bug..."
   haiku returns: TRIAGE

5. Write-back (only on LLM fallback)
   insert_example(description, embedding, task_type="TRIAGE", source="llm_fallback")
   # Next time a similar task comes in, it's in the store
```

### Key design choices you made:

| Decision | Why |
|----------|-----|
| **Lazy-load embeddings** | Avoid paying model-load cost on every import; singleton in process lifetime (embeddings.py:23-33) |
| **pgvector + Postgres, not Pinecone** | Avoid adding infrastructure; leverage existing database; HNSW index is standard in modern Postgres |
| **No similarity floor** | Validation showed it's too noisy; agreement fraction carries the real signal |
| **Write-back only on LLM fallback** | NN-confident resolutions already cover that region of embedding space; writing again adds near-dupes with no new info |
| **AGREEMENT_THRESHOLD = 0.8 (4/5)** | Sweet spot: 87% accuracy, 32% coverage. Escalates under uncertainty (cheaper LLM fallback is more reliable) |
| **Separate label_column validation** | SQL injection protection despite the f-string—validate against a fixed allow-list, never interpolate raw caller input (vector_store.py:29, 64, 88) |

### Storage schema:

```sql
CREATE TABLE routing_examples (
    id BIGSERIAL PRIMARY KEY,
    description TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,  -- the dense vector
    task_type TEXT,                   -- label for tier2_classifier.resolve_task_type()
    is_high_stakes BOOLEAN,           -- label for tier2_classifier.resolve_high_stakes()
    source TEXT NOT NULL,             -- "seed" (from llm-eval-harness) or "llm_fallback"
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX routing_examples_embedding_hnsw
    ON routing_examples USING hnsw (embedding vector_cosine_ops);
```

The `routing_decisions` table (audit log) is separate—it's for drift monitoring, not vector search, so no HNSW index.

---

## Part 3: How to Extend It

### 1. **Add a new label axis** (next tier-2 dimension)

Say you want to classify by `urgency` (LOW, MEDIUM, HIGH).

**Steps:**
1. Add the column to `routing_examples`:
   ```sql
   ALTER TABLE routing_examples ADD COLUMN urgency TEXT;
   ```

2. Create a new resolve function in tier2_classifier.py:
   ```python
   def resolve_urgency(description: str, embedding: list[float]) -> Tier2Resolution | None:
       matches = vector_store.nearest_neighbors(embedding, "urgency", k=NEIGHBOR_K)
       majority = _nn_majority(matches)
       if majority is not None:
           return Tier2Resolution(urgency=majority, source="nn")
       # LLM fallback...
   ```

3. Call it from router.py when classifying. Write-back works automatically—`insert_example()` already accepts arbitrary label columns (it validates them against VALID_LABEL_COLUMNS, so add "urgency" to that list).

**Why this scales:** The embedding is one-time; neighbors are one query per new label. Query cost is O(log n) with HNSW, not O(n).

### 2. **Experiment with different embedding models**

Your current model: **all-MiniLM-L6-v2** (384 dims, ~33MB, ~50ms encode).

If quality drifts, swap it:
```python
# embeddings.py
MODEL_NAME = "all-mpnet-base-v2"  # 768 dims, more expensive, better quality
# Then re-embed: run a one-off script over all routing_examples
```

But **don't do this without validation.** The AGREEMENT_THRESHOLD was tuned for all-MiniLM-L6-v2 on 98 seed examples. Switching models changes similarity distributions—re-run the leave-one-out check from audit_tier2.py to re-derive thresholds.

### 3. **Dynamic threshold tuning**

Right now AGREEMENT_THRESHOLD is static (0.8). As real traffic accumulates:

```python
# In tier2_classifier.py, after accumulating 100+ real examples:
def _estimate_agreement_threshold():
    examples = vector_store.all_labeled_examples("task_type")
    # Leave-one-out: for each example, score its k neighbors' majority vote
    # Measure accuracy at different thresholds (0.6, 0.7, 0.8, ...)
    # Pick the threshold that maximizes F1 or your chosen metric
```

This is exactly what audit_tier2.py does—you can wire it to auto-update the threshold.

### 4. **Handle drift**: Monitor when neighbors stop agreeing

Your routing_decisions table logs `task_type_source` (nn vs. llm_fallback). If the ratio flips—LLM fallback suddenly > NN confidence—the embedding quality or label distribution may have drifted. Set up alerting:

```sql
SELECT 
    DATE(created_at), 
    task_type_source, 
    COUNT(*)
FROM routing_decisions
WHERE created_at > now() - interval '7 days'
GROUP BY DATE(created_at), task_type_source
ORDER BY created_at DESC;
```

If llm_fallback rate spikes, investigate: did your descriptions change? Did a new task_type emerge?

### 5. **Hybrid search** (future: combine NN + keyword matching)

Pure embedding search misses exact-match signals. If a task literally says "SQL", a keyword match is instant confidence. Postgres + pgvector can do both in one query:

```sql
SELECT id, 1 - (embedding <=> %s) as vector_sim, 
       ts_rank(description_tsv, query) as keyword_score
FROM routing_examples
WHERE description_tsv @@ query  -- full-text search
   OR embedding <=> %s < 0.1     -- or vector similarity
ORDER BY vector_sim * keyword_score DESC  -- blend scores
```

This requires adding a tsvector column and a GIN index, but Postgres handles both natively.

---

## Part 4: Skills to Learn (What You Should Build)

### Tier 1: Production fundamentals (you should know these)

1. **HNSW algorithm internals**
   - How it builds a navigable graph of vectors
   - Why it's approximate (skips distant branches)
   - Tuning parameters: M (connections per node), ef (search breadth)
   - When to use exact search (small stores, high-stakes) vs. HNSW (large stores, latency-sensitive)
   
   **Concrete exercise:** Check your index stats:
   ```sql
   SELECT * FROM pg_stat_user_indexes WHERE indexname = 'routing_examples_embedding_hnsw';
   ```
   How many pages? How's the query planner using it? (EXPLAIN ANALYZE on nearest_neighbors calls)

2. **Embedding model selection trade-offs**
   - Dimensionality vs. inference speed vs. quality
   - Fine-tuning embeddings on domain-specific data (vs. off-the-shelf)
   - When to use mean-pooling (what you do) vs. CLS tokens vs. learned aggregation
   - Cross-encoder re-ranking (embedding finds candidates, cross-encoder ranks them)
   
   **Concrete exercise:** Run llm-eval-harness's embedding quality check on your seed set. Does all-MiniLM-L6-v2 separate task types cleanly? If not, why not?

3. **ANN index tuning (pgvector-specific)**
   - HNSW parameters: ef_construction (index-time cost/quality), ef_search (query-time accuracy/speed)
   - When to rebuild the index (after many inserts)
   - Comparing exact vs. HNSW recall
   
   **Concrete exercise:**
   ```sql
   -- Check current HNSW settings
   SELECT * FROM pg_class WHERE relname = 'routing_examples_embedding_hnsw';
   
   -- Run a recall check: exact NN vs. HNSW for 10 random queries
   ```

### Tier 2: Advanced techniques (career-strengthening)

4. **Multi-stage retrieval pipelines**
   - Dense retrieval (your NN lookup) as the first stage
   - Re-ranking with a small cross-encoder model
   - Sparse retrieval (BM25 full-text) for hybrid
   - When to use each stage (trade-off complexity for accuracy)

5. **Semantic search at scale (1M+ vectors)**
   - When does HNSW start to slow down? (typically 50M+ vectors, depending on dimensionality)
   - Sharding strategies: geographic (store by region), semantic (cluster vectors, one shard per cluster)
   - Read replicas for query-heavy workloads
   - Write-optimized stores (Milvus, Weaviate) vs. write-minimal (pgvector + Postgres)

6. **Continuous learning patterns** (your tier-2 architecture)
   - How to detect when write-back should slow down (store saturation, diminishing returns)
   - Active learning: which uncertain queries should get human labels vs. LLM fallback?
   - Concept drift: retraining schedules when embedding quality degrades on real traffic
   - Cost-benefit of storing everything vs. selective persistence (keep high-uncertainty examples)

7. **Vector database comparison** (if you need to defend pgvector later)
   - pgvector: SQL-native, transaction guarantees, but needs Postgres
   - Pinecone: fully managed, serverless, but vendor lock-in + API costs
   - Milvus: self-hosted, high throughput, more complex ops
   - Qdrant: modern, web-native (gRPC), good for Rust integrations
   - Weaviate: GraphQL API, semantic graphs (not just vectors)
   
   **Concrete exercise:** Run a benchmark—your current workload with pgvector vs. a trial Qdrant instance. Latency? Cost? Operational overhead?

---

## Part 5: Suggested Learning Path

### Month 1: Deepen HNSW + embedding models
- Read the HNSW paper (Malkov & Yashunin, 2018)—it's surprisingly readable.
- Run leave-one-out validation on your 98 seed examples with different similarity thresholds (you already have the data).
- Try a 768-dim model (all-mpnet-base-v2) on a copy of your store; measure if agreement improves.

### Month 2: Production hardening
- Set up monitoring on your routing_decisions drift signals.
- Add the hybrid search query (BM25 + vector) to vector_store.py; benchmark it against pure vector search.
- Document HNSW settings and why you chose them (for future ops).

### Month 3: Scale thinking
- Model your growth: at 10K queries/day, when does HNSW degrade? (Postgres can handle it; ops becomes the constraint)
- Write a sharding sketch—if you needed to split the store, how would you do it?
- Explore fine-tuning: collect 50-100 human-labeled examples from real traffic, fine-tune all-MiniLM-L6-v2 on your task domain, compare accuracy.

---

## Quick Reference: Your Vector DB Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embedding model | sentence-transformers (all-MiniLM-L6-v2) | 384 dims, fast (~50ms), standard for NLP |
| Vector storage | pgvector + Postgres | Native SQL, HNSW index, no ops overhead |
| Index algorithm | HNSW (Hierarchical Navigable Small World) | Log(n) lookup, ~1-5ms for 1K rows |
| Fallback logic | haiku LLM | 10x cheaper than sonnet, writes back to store |
| Validation method | Leave-one-out agreement check | Measured 87% accuracy at 4/5 neighbors agreeing |
| Write strategy | LLM fallback only, not NN-confident | Avoids near-duplicate rows; store grows semantically |

---

## Follow-up Question

Given that AGREEMENT_THRESHOLD was tuned on 98 seed examples via leave-one-out, what would you worry about when applying it to real traffic that arrives weeks later? (Hint: think about distribution shift.)

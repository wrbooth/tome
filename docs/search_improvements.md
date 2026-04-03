# Codex Search System - Advanced Architectural Patterns for Semantic Relationships

## Executive Summary

This document outlines advanced architectural patterns and indexing strategies to improve semantic relationships and role-based queries in the Codex search system. Based on analysis of current performance issues with three specific query types, we've identified key areas for enhancement.

## Current Performance Analysis

### Poor-Performing Query Patterns

Our analysis identified three query types that consistently rank poorly (rank 15-17):

1. **"Who were the main historians of Guernsey County?"** (Expected: Page 9, Found at rank 17)
2. **"In what years did the early settlers from the Isle of Guernsey arrive in Guernsey County?"** (Expected: Page 29, Found at rank 15)  
3. **"Who was the first sitting president to pass through Cambridge?"** (Expected: Page 58, Found at rank 15)

### Common Failure Patterns

1. **Semantic Gap**: Queries use different terminology than content (e.g., "historians" vs "principal characters")
2. **Entity Relationship Complexity**: Complex relationships between entities not captured
3. **Role-Based Queries**: Specific roles ("first sitting president") not recognized
4. **Temporal Specificity**: Exact year requests not prioritized
5. **Geographic Confusion**: Cambridge, Ohio vs Cambridge, Massachusetts

## 1. Multi-Modal Embedding Strategies

### Dense Passage Retrieval (DPR) with Query-Specific Encoders

**Current Limitation**: Single embedding per passage
**Proposed Solution**: Multiple specialized embeddings

```python
class MultiModalEmbeddings:
    def __init__(self):
        self.query_encoder = SentenceTransformer('facebook/dpr-question_encoder-single-nq-base')
        self.passage_encoder = SentenceTransformer('facebook/dpr-ctx_encoder-single-nq-base')
        self.role_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def encode_query(self, query: str, query_type: str) -> Dict[str, List[float]]:
        return {
            'general': self.query_encoder.encode(query),
            'role_specific': self.role_encoder.encode(f"{query_type}: {query}"),
            'temporal': self.role_encoder.encode(f"when: {query}") if 'when' in query else None
        }
```

### Hybrid Embedding Architecture

```sql
-- Enhanced schema for multiple embedding types
ALTER TABLE passages ADD COLUMN embedding_general vector(768);
ALTER TABLE passages ADD COLUMN embedding_temporal vector(768);
ALTER TABLE passages ADD COLUMN embedding_role vector(768);
ALTER TABLE passages ADD COLUMN embedding_entity vector(768);

-- Index each embedding type separately
CREATE INDEX idx_passages_general_vec ON passages USING ivfflat (embedding_general vector_cosine_ops);
CREATE INDEX idx_passages_temporal_vec ON passages USING ivfflat (embedding_temporal vector_cosine_ops);
CREATE INDEX idx_passages_role_vec ON passages USING ivfflat (embedding_role vector_cosine_ops);
```

## 2. Knowledge Graph Integration

### Entity-Relationship Graph

```sql
-- New tables for knowledge graph
CREATE TABLE entity_relationships (
    id uuid PRIMARY KEY,
    source_entity text,
    target_entity text,
    relationship_type text, -- 'FOUNDED_BY', 'VISITED_BY', 'ARRIVED_IN', 'WAS_PRESIDENT'
    confidence float,
    source_passage_id uuid REFERENCES passages(id),
    created_at timestamp DEFAULT now()
);

CREATE TABLE entity_roles (
    entity_id text,
    role_type text, -- 'PRESIDENT', 'HISTORIAN', 'SETTLER', 'FOUNDER'
    context text,
    temporal_start int,
    temporal_end int,
    source_passage_id uuid REFERENCES passages(id)
);
```

### Graph-Based Retrieval

```python
def graph_enhanced_search(query: str, entities: Dict) -> List[Tuple[str, float]]:
    """Use knowledge graph to enhance search results."""
    # Find entities in query
    query_entities = extract_entities(query)
    
    # Find related entities through graph
    related_entities = []
    for entity in query_entities:
        neighbors = get_graph_neighbors(entity)
        related_entities.extend(neighbors)
    
    # Boost passages containing related entities
    return boost_by_entity_relationships(candidates, related_entities)
```

## 3. Advanced Query Understanding

### Semantic Role Labeling

```python
def extract_semantic_roles(query: str) -> Dict[str, Any]:
    """Extract semantic roles from queries."""
    return {
        'agent': extract_agent(query),      # "Who" questions
        'patient': extract_patient(query),  # "What" questions  
        'temporal': extract_temporal(query), # "When" questions
        'location': extract_location(query), # "Where" questions
        'manner': extract_manner(query),    # "How" questions
        'role': extract_role(query)         # Role-based queries
    }

def extract_role(query: str) -> Optional[str]:
    """Extract role information from queries."""
    role_patterns = {
        'first_sitting_president': r'first\s+sitting\s+president',
        'main_historians': r'main\s+historians?',
        'early_settlers': r'early\s+settlers?',
        'founders': r'founders?',
        'commanders': r'commanders?'
    }
    
    for role, pattern in role_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            return role
    return None
```

### Query Intent Classification with Fine-tuning

```python
class QueryIntentClassifier:
    def __init__(self):
        self.intents = [
            'entity_lookup',      # "Who is X?"
            'temporal_query',     # "When did X happen?"
            'causal_query',       # "Why did X happen?"
            'role_query',         # "Who was the first X?"
            'comparison_query',   # "Which X was better?"
            'factual_query'       # "What is X?"
        ]
    
    def classify(self, query: str) -> Dict[str, float]:
        # Use fine-tuned model for intent classification
        return self.model.predict(query)
```

## 4. Hierarchical Indexing

### Multi-Level Passage Indexing

```python
class HierarchicalIndex:
    def __init__(self):
        self.document_level = DocumentIndex()
        self.section_level = SectionIndex()
        self.passage_level = PassageIndex()
        self.entity_level = EntityIndex()
    
    def search(self, query: str) -> List[Dict]:
        # Search at multiple levels
        doc_results = self.document_level.search(query)
        section_results = self.section_level.search(query)
        passage_results = self.passage_level.search(query)
        entity_results = self.entity_level.search(query)
        
        # Combine with hierarchical boosting
        return self.fuse_hierarchical_results(
            doc_results, section_results, passage_results, entity_results
        )
```

### Semantic Chunking with Overlap

```python
def semantic_chunking(text: str, max_tokens: int = 500) -> List[str]:
    """Chunk text based on semantic boundaries, not just token count."""
    # Use sentence transformers to find semantic boundaries
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        # Check semantic similarity with current chunk
        if current_chunk and semantic_similarity(sentence, current_chunk[-1]) < 0.7:
            # Start new chunk if semantic similarity is low
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = len(sentence.split())
        else:
            current_chunk.append(sentence)
            current_tokens += len(sentence.split())
            
            if current_tokens > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
    
    return chunks
```

## 5. Cross-Encoder Reranking

### Multi-Stage Retrieval Pipeline

```python
class MultiStageRetriever:
    def __init__(self):
        self.first_stage = HybridRetriever()  # BM25 + Vector
        self.second_stage = CrossEncoderReranker()
        self.third_stage = LLMReranker()
    
    def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        # Stage 1: Get initial candidates
        candidates = self.first_stage.retrieve(query, k=100)
        
        # Stage 2: Cross-encoder reranking
        reranked = self.second_stage.rerank(query, candidates, k=20)
        
        # Stage 3: LLM-based final ranking
        final_results = self.third_stage.rerank(query, reranked, k=k)
        
        return final_results
```

### Query-Specific Reranking

```python
class QuerySpecificReranker:
    def __init__(self):
        self.temporal_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.entity_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.role_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, candidates: List[Dict], query_type: str) -> List[Dict]:
        if query_type == 'when':
            return self.temporal_reranker.rerank(query, candidates)
        elif query_type == 'who':
            return self.entity_reranker.rerank(query, candidates)
        elif 'role' in query_type:
            return self.role_reranker.rerank(query, candidates)
        else:
            return candidates
```

## 6. Temporal-Aware Indexing

### Temporal Embeddings

```python
def create_temporal_embedding(text: str, years: List[int]) -> List[float]:
    """Create embeddings that encode temporal information."""
    base_embedding = get_base_embedding(text)
    
    # Encode temporal context
    temporal_context = encode_temporal_context(years)
    
    # Combine base and temporal embeddings
    return combine_embeddings(base_embedding, temporal_context)
```

### Temporal Indexing Schema

```sql
-- Enhanced temporal indexing
CREATE TABLE temporal_events (
    id uuid PRIMARY KEY,
    passage_id uuid REFERENCES passages(id),
    event_type text, -- 'ARRIVAL', 'FOUNDING', 'VISIT', 'BATTLE'
    start_year int,
    end_year int,
    confidence float,
    entities text[]
);

CREATE INDEX idx_temporal_events_year ON temporal_events (start_year, end_year);
CREATE INDEX idx_temporal_events_type ON temporal_events (event_type);
```

## 7. Contextual Query Expansion

### Dynamic Query Expansion

```python
class ContextualQueryExpander:
    def __init__(self):
        self.synonym_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.knowledge_base = load_knowledge_base()
    
    def expand_query(self, query: str, query_type: str) -> List[str]:
        expansions = [query]
        
        # Add semantic synonyms
        synonyms = self.get_semantic_synonyms(query)
        expansions.extend(synonyms)
        
        # Add role-specific expansions
        if 'role' in query_type:
            role_expansions = self.get_role_expansions(query)
            expansions.extend(role_expansions)
        
        # Add temporal expansions
        if 'when' in query_type:
            temporal_expansions = self.get_temporal_expansions(query)
            expansions.extend(temporal_expansions)
        
        # Add entity expansions
        entities = extract_entities(query)
        for entity in entities:
            entity_expansions = self.get_entity_expansions(entity)
            expansions.extend(entity_expansions)
        
        return list(set(expansions))
```

## 8. Implementation Roadmap

### Phase 1: Quick Wins (High Impact, Low Effort)
1. **Enhanced Query Expansion**
   - Add semantic synonyms for key terms
   - Implement role-specific expansions
   - Add temporal context expansions

2. **Improved Entity Recognition**
   - Enhance regex patterns for role extraction
   - Add synonym mappings for common terms
   - Implement fuzzy matching for entity names

### Phase 2: Core Improvements (High Impact, Medium Effort)
1. **Cross-Encoder Reranking**
   - Implement query-specific rerankers
   - Add temporal-aware reranking
   - Integrate role-based reranking

2. **Knowledge Graph Foundation**
   - Create entity relationship tables
   - Implement basic graph traversal
   - Add relationship-based boosting

### Phase 3: Advanced Features (Medium Impact, High Effort)
1. **Multi-Modal Embeddings**
   - Implement query-specific encoders
   - Add temporal embeddings
   - Create role-aware embeddings

2. **Hierarchical Indexing**
   - Implement multi-level search
   - Add semantic chunking
   - Create document-level indexing

## 9. Expected Performance Improvements

### Target Metrics
- **Role-based queries**: Improve from rank 15-17 to rank 1-5
- **Temporal queries**: Improve from rank 15 to rank 1-3
- **Entity relationship queries**: Improve from rank 17 to rank 1-5

### Success Criteria
- 90% of role-based queries return correct page in top 5 results
- 95% of temporal queries return correct page in top 3 results
- 85% of entity relationship queries return correct page in top 5 results

## 10. Technical Considerations

### Performance Impact
- **Memory**: Multi-modal embeddings will increase memory usage by ~4x
- **Latency**: Cross-encoder reranking adds ~100-200ms per query
- **Storage**: Knowledge graph tables will add ~20-30% storage overhead

### Scalability
- **Horizontal Scaling**: Vector search can be distributed across multiple nodes
- **Caching**: Implement Redis caching for frequently accessed embeddings
- **Batch Processing**: Use async processing for knowledge graph updates

### Monitoring
- **Query Performance**: Track rank improvements for different query types
- **System Metrics**: Monitor memory usage, latency, and storage growth
- **User Feedback**: Implement query result feedback mechanism

## Conclusion

The proposed architectural improvements address the core limitations of the current search system by adding sophisticated semantic understanding, temporal awareness, and entity relationship modeling. The phased implementation approach allows for incremental improvements while maintaining system stability.

The key insight is that current systems excel at term matching but struggle with semantic relationships. These patterns bridge that gap through advanced NLP techniques, knowledge graphs, and multi-stage retrieval pipelines.

---

*Document Version: 1.0*  
*Last Updated: 2025-01-21*  
*Author: AI Assistant*  
*Status: Draft for Review*

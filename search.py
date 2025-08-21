#!/usr/bin/env python3
"""
Codex Search Script

Implements hybrid search with RRF fusion, query type detection, and specialized handling.
"""

import os
import sys
import click
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
import re
from typing import List, Dict, Any, Tuple, Optional

load_dotenv()

# RRF parameters
RRF_K = 60

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "codex"),
        user=os.getenv("DB_USER", "codex"),
        password=os.getenv("DB_PASSWORD", "codex")
    )

def rrf(rank: int) -> float:
    """Reciprocal rank fusion score."""
    return 1.0 / (RRF_K + rank)

def detect_query_type(query: str) -> str:
    """Detect query type: 'who', 'when', 'where', 'factoid', 'company', or 'general'."""
    query_lower = query.lower()
    
    # Person queries
    if re.search(r'\bwho\s+(is|was)\b', query_lower):
        return 'who'
    
    # Company/organization queries
    elif (re.search(r'\bwhat\s+was\s+the\s+name\s+of\b', query_lower) or
          re.search(r'\bcompany\b', query_lower) or
          re.search(r'\bcorporation\b', query_lower) or
          re.search(r'\borganization\b', query_lower) or
          re.search(r'\bsteel\s+mill\b', query_lower) or
          re.search(r'\biron\s+and\s+steel\b', query_lower)):
        return 'company'
    
    # Temporal queries - expanded patterns
    elif (re.search(r'\bwhen\b', query_lower) or 
          re.search(r'\bdid.*\b(in|during|on)\b.*\d{4}', query_lower) or
          re.search(r'\bwhat\s+year', query_lower) or
          re.search(r'\bwhat\s+years', query_lower)):
        return 'when'
    
    # Location queries
    elif re.search(r'\bwhere\b', query_lower):
        return 'where'
    
    # Factual queries about specific entities/events
    elif (re.search(r'\bwhat\b.*\b(taverns?|inns?|places?|buildings?)\b', query_lower) or
          re.search(r'\bdid\b.*\b(family|person|group)\b', query_lower) or
          re.search(r'\bwhat\s+had\b', query_lower) or
          re.search(r'\bwhat\s+did\b', query_lower)):
        return 'factoid'
    
    else:
        return 'general'

def extract_person_from_query(query: str) -> Optional[str]:
    """Extract person name from 'who is/was X' queries."""
    match = re.search(r'\bwho\s+(is|was)\s+([^?]+)', query, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return None

def extract_entities_from_query(query: str) -> Dict[str, List[str]]:
    """Extract entities from query for expansion."""
    entities = {
        "persons": [],
        "places": [],
        "events": [],
        "dates": [],
        "families": [],
        "companies": [],
        "industries": []
    }
    
    # Extract company names (e.g., "steel mill", "iron and steel company")
    company_patterns = [
        r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+Company\b',
        r'\b([A-Z][a-z]+)\s+Company\b',
        r'\bsteel\s+mill\b',
        r'\biron\s+and\s+steel\b'
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                entities["companies"].append(" ".join(match))
            else:
                entities["companies"].append(match)
    
    # Extract industry terms
    industry_patterns = [
        r'\bsteel\b',
        r'\biron\b',
        r'\bmill\b',
        r'\bfactory\b',
        r'\bmanufacturing\b'
    ]
    
    for pattern in industry_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        entities["industries"].extend(matches)
    
    # Extract family names (e.g., "Naftal family", "McDonald family")
    family_patterns = [
        r'\b([A-Z][a-z]+)\s+family\b',
        r'\bfamily\s+([A-Z][a-z]+)\b'
    ]
    
    for pattern in family_patterns:
        matches = re.findall(pattern, query)
        for match in matches:
            entities["families"].append(match)
            # Also add as person for individual name matching
            entities["persons"].append(match)
    
    # Extract person names (simple heuristic)
    person_patterns = [
        r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b',  # First Last
        r'\b([A-Z][a-z]+)\b'  # Single capitalized word
    ]
    
    for pattern in person_patterns:
        matches = re.findall(pattern, query)
        for match in matches:
            if isinstance(match, tuple):
                name = " ".join(match)
                # Avoid adding family names twice
                if name not in entities["families"]:
                    entities["persons"].append(name)
            else:
                # Avoid adding family names twice
                if match not in entities["families"]:
                    entities["persons"].append(match)
    
    # Extract dates/years
    year_pattern = r'\b(17|18|19|20)\d{2}\b'
    entities["dates"] = re.findall(year_pattern, query)
    
    # Also extract century references and convert to specific years
    century_patterns = [
        (r'\b1700s\b', ['1700', '1701', '1702', '1703', '1704', '1705', '1706', '1707', '1708', '1709']),
        (r'\b1800s\b', ['1800', '1801', '1802', '1803', '1804', '1805', '1806', '1807', '1808', '1809']),
        (r'\b1900s\b', ['1900', '1901', '1902', '1903', '1904', '1905', '1906', '1907', '1908', '1909']),
    ]
    
    for pattern, years in century_patterns:
        if re.search(pattern, query):
            entities["dates"].extend(years)
    
    # Extract places (simple heuristic)
    place_patterns = [
        r'\b([A-Z][a-z]+)\s+(County|State|Town|City|Village)\b',
        r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'  # Potential place names
    ]
    
    for pattern in place_patterns:
        matches = re.findall(pattern, query)
        for match in matches:
            if isinstance(match, tuple):
                entities["places"].append(" ".join(match))
            else:
                entities["places"].append(match)
    
    return entities

def generate_query_expansions(query: str, query_type: str) -> List[str]:
    """Generate query expansions based on type and entities."""
    expansions = [query]
    
    # Extract entities
    entities = extract_entities_from_query(query)
    
    # Add family name expansions
    for family in entities["families"]:
        expansions.extend([
            family,  # Just the family name
            f"{family} family",
            f"family {family}"
        ])
    
    # Add person name expansions
    for person in entities["persons"]:
        expansions.append(person)
    
    # Add date expansions for temporal queries
    if query_type == "when":
        for date in entities["dates"]:
            expansions.append(date)
        # Add common temporal terms
        expansions.extend([
            "date", "year", "time", "period", "era"
        ])
    
    # Add place expansions
    for place in entities["places"]:
        expansions.append(place)
    
    # Add company expansions
    for company in entities["companies"]:
        expansions.append(company)
    
    # Add industry expansions
    for industry in entities["industries"]:
        expansions.append(industry)
        # Add industry synonyms
        if industry.lower() == "steel":
            expansions.extend(["iron", "metal", "steelworks"])
        elif industry.lower() == "iron":
            expansions.extend(["steel", "metal", "ironworks"])
        elif industry.lower() == "mill":
            expansions.extend(["factory", "plant", "works"])
    
    # Add company-related terms for company queries
    if query_type == "company" or entities["companies"] or entities["industries"]:
        company_terms = [
            "company", "corporation", "incorporated", "inc",
            "steel mill", "iron works", "factory", "plant",
            "manufacturing", "industry", "business"
        ]
        expansions.extend(company_terms)
    
    # Add arrival/settlement synonyms for family queries
    if entities["families"] or query_type == "factoid":
        arrival_synonyms = [
            "arrive", "arrived", "arrival",
            "come", "came", "coming",
            "settle", "settled", "settlement",
            "migrate", "migrated", "migration",
            "move", "moved", "moving"
        ]
        expansions.extend(arrival_synonyms)
    
    # Add temporal context for factoid queries
    if query_type == "factoid" and entities["dates"]:
        for date in entities["dates"]:
            expansions.extend([
                f"in {date}",
                f"during {date}",
                f"by {date}",
                f"around {date}"
            ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expansions = []
    for exp in expansions:
        if exp.lower() not in seen:
            seen.add(exp.lower())
            unique_expansions.append(exp)
    
    return unique_expansions

def get_meili_candidates(query: str, k: int = 200, document_id: Optional[str] = None) -> List[Tuple[str, int]]:
    """Get BM25 candidates from Meilisearch."""
    try:
        from meilisearch import Client
        
        client = Client(
            os.getenv("MEILI_URL", "http://localhost:7700")
        )
        
        index = client.index("passages")
        
        # Build search parameters
        search_params = {
            "q": query,
            "limit": k,
            "attributesToRetrieve": ["id"]
        }
        
        if document_id:
            search_params["filter"] = f"document_id = {document_id}"
        
        response = index.search(query, search_params)
        
        # Extract passage IDs and ranks
        candidates = []
        for i, hit in enumerate(response["hits"]):
            candidates.append((hit["id"], i + 1))  # rank starts at 1
        
        return candidates
        
    except ImportError:
        print("Warning: Meilisearch client not available")
        return []
    except Exception as e:
        print(f"Warning: Failed to get Meilisearch candidates: {e}")
        return []

def get_vector_candidates(query: str, k: int = 200, document_id: Optional[str] = None) -> List[Tuple[str, int]]:
    """Get vector similarity candidates from pgvector."""
    try:
        # Get query embedding
        query_embedding = get_query_embedding(query)
        if not query_embedding:
            return []
        
        conn = get_db_connection()
        
        # Build query
        sql = """
            SELECT id, 1 - (embedding <=> %s::vector) as similarity
            FROM passages 
            WHERE embedding IS NOT NULL
        """
        params = [query_embedding]
        
        if document_id:
            sql += " AND document_id = %s"
            params.append(document_id)
        
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, k])
        
        with conn.cursor() as cur:
            cur.execute(sql, params)
            results = cur.fetchall()
        
        conn.close()
        
        # Return passage IDs with ranks
        return [(row[0], i + 1) for i, row in enumerate(results)]
        
    except Exception as e:
        print(f"Warning: Failed to get vector candidates: {e}")
        return []

def get_query_embedding(query: str) -> Optional[List[float]]:
    """Get embedding for query text."""
    try:
        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=query,
                encoding_format="float"
            )
            return response.data[0].embedding
        
        # Fallback to local model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/e5-base-v2")
        embedding = model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
        
    except Exception as e:
        print(f"Warning: Failed to get query embedding: {e}")
        return None

def apply_query_boosting(candidates: List[Tuple[str, float]], query: str, conn) -> List[Tuple[str, float]]:
    """Apply query-specific boosting to candidates."""
    
    query_type = detect_query_type(query)
    entities = extract_entities_from_query(query)
    
    if query_type == 'who':
        person = extract_person_from_query(query)
        if person:
            # Boost passages with PERSON entity matching the query
            boosted_candidates = []
            
            for passage_id, score in candidates:
                # Check if passage has matching person entity
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 1 FROM passage_entities 
                        WHERE passage_id = %s 
                        AND ent_type = 'PERSON' 
                        AND norm_entity = %s
                    """, (passage_id, person.lower()))
                    
                    if cur.fetchone():
                        # Boost score for matching person
                        boosted_score = score * 1.5
                        boosted_candidates.append((passage_id, boosted_score))
                    else:
                        boosted_candidates.append((passage_id, score))
            
            return boosted_candidates
    
    elif query_type == 'when':
        # Boost passages with dates/years for temporal queries
        boosted_candidates = []
        
        for passage_id, score in candidates:
            boost_multiplier = 1.0
            
            # Check if passage has years mentioned
            with conn.cursor() as cur:
                # Check for specific years mentioned in the query
                if entities.get("dates"):
                    for date in entities["dates"]:
                        cur.execute("""
                            SELECT COUNT(*) FROM passage_years 
                            WHERE passage_id = %s AND year = %s
                        """, (passage_id, int(date)))
                        
                        if cur.fetchone()[0] > 0:
                            # Strong boost for exact year match
                            boost_multiplier *= 2.0
                            break
                
                # Also check for any years in the passage (weaker boost)
                if boost_multiplier == 1.0:  # Only if no exact match found
                    cur.execute("""
                        SELECT COUNT(*) FROM passage_years 
                        WHERE passage_id = %s
                    """, (passage_id,))
                    
                    year_count = cur.fetchone()[0]
                    if year_count > 0:
                        # Moderate boost for passages with any dates
                        boost_multiplier *= 1.3
            
            boosted_score = score * boost_multiplier
            boosted_candidates.append((passage_id, boosted_score))
        
        return boosted_candidates
    
    elif query_type == 'factoid':
        # Boost passages with family names and temporal information
        boosted_candidates = []
        
        for passage_id, score in candidates:
            boost_multiplier = 1.0
            
            # Check if passage has family-related entities
            with conn.cursor() as cur:
                # Check for family names in entities
                if entities.get("families"):
                    for family in entities["families"]:
                        cur.execute("""
                            SELECT 1 FROM passage_entities 
                            WHERE passage_id = %s 
                            AND ent_type = 'PERSON' 
                            AND (norm_entity = %s OR norm_entity LIKE %s)
                        """, (passage_id, family.lower(), f"%{family.lower()}%"))
                        
                        if cur.fetchone():
                            boost_multiplier *= 1.3
                            break
                
                # Check for temporal information
                if entities.get("dates"):
                    for date in entities["dates"]:
                        cur.execute("""
                            SELECT COUNT(*) FROM passage_years 
                            WHERE passage_id = %s AND year = %s
                        """, (passage_id, int(date)))
                        
                        if cur.fetchone()[0] > 0:
                            boost_multiplier *= 1.2
                            break
            
            boosted_score = score * boost_multiplier
            boosted_candidates.append((passage_id, boosted_score))
        
        return boosted_candidates
    
    return candidates

def get_passage_details(conn, passage_ids: List[str]) -> List[Dict[str, Any]]:
    """Get detailed passage information."""
    if not passage_ids:
        return []
    
    placeholders = ','.join(['%s'] * len(passage_ids))
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT p.id, p.text, p.page, d.title, p.headings_path
            FROM passages p
            JOIN documents d ON p.document_id = d.id
            WHERE p.id IN ({placeholders})
            ORDER BY p.id
        """, passage_ids)
        
        return cur.fetchall()

def format_results(results: List[Dict[str, Any]], scores: Dict[str, float]) -> str:
    """Format search results for display."""
    output = []
    
    for i, result in enumerate(results, 1):
        passage_id = result['id']
        score = scores.get(passage_id, 0.0)
        title = result['title']
        page = result['page']
        
        # Truncate text for snippet
        text = result['text']
        snippet = text[:200] + "..." if len(text) > 200 else text
        
        output.append(f"{i:2d}  {score:.3f}  {title}  p. {page}  \"{snippet}\"")
    
    return '\n'.join(output)

def generate_answer(query: str, query_type: str, results: List[Dict[str, Any]]) -> str:
    """Generate a simple answer summary from search results."""
    if not results:
        return "I couldn't find any relevant information to answer your question."
    
    # For now, just return the top result as a snippet
    # This will be replaced by LLM processing later
    top_result = results[0]
    return f"Top result: {top_result['text'][:300]}... (Source: p. {top_result['page']})"

@click.command()
@click.option('--q', 'query', required=True, help='Search query')
@click.option('--k', default=20, type=int, help='Number of results to return')
@click.option('--doc', 'document_id', help='Filter by document ID')
def main(query: str, k: int, document_id: Optional[str]):
    """Search passages using hybrid retrieval with RRF fusion."""
    
    print(f"Searching for: {query}")
    
    # Analyze query
    query_type = detect_query_type(query)
    print(f"Query type: {query_type}")
    
    # Generate query expansions
    expansions = generate_query_expansions(query, query_type)
    if len(expansions) > 1:
        print(f"Query expansions: {expansions[1:]}")  # Skip the original query
    
    # Get candidates from both sources using original query AND expansions
    print("Getting BM25 candidates...")
    meili_candidates = get_meili_candidates(query, k=200, document_id=document_id)
    
    # Add candidates from expansions
    for expansion in expansions[1:]:  # Skip original query
        expansion_candidates = get_meili_candidates(expansion, k=50, document_id=document_id)
        meili_candidates.extend(expansion_candidates)
    
    # Remove duplicates while preserving order
    seen_ids = set()
    unique_meili_candidates = []
    for passage_id, rank in meili_candidates:
        if passage_id not in seen_ids:
            seen_ids.add(passage_id)
            unique_meili_candidates.append((passage_id, rank))
    
    meili_candidates = unique_meili_candidates[:200]  # Keep top 200
    print(f"Found {len(meili_candidates)} BM25 candidates")
    
    print("Getting vector candidates...")
    vector_candidates = get_vector_candidates(query, k=200, document_id=document_id)
    print(f"Found {len(vector_candidates)} vector candidates")
    
    if not meili_candidates and not vector_candidates:
        print("No candidates found")
        return
    
    # Apply RRF fusion
    print("Applying RRF fusion...")
    scores = defaultdict(float)
    
    for passage_id, rank in meili_candidates:
        scores[passage_id] += rrf(rank)
    
    for passage_id, rank in vector_candidates:
        scores[passage_id] += rrf(rank)
    
    # Sort by score and take top k
    top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    if not top_candidates:
        print("No results after fusion")
        return
    
    # Apply query-specific boosting
    conn = get_db_connection()
    boosted_candidates = apply_query_boosting(top_candidates, query, conn)
    
    # Get passage details
    passage_ids = [pid for pid, _ in boosted_candidates]
    passage_details = get_passage_details(conn, passage_ids)
    
    # Create score mapping
    score_map = {pid: score for pid, score in boosted_candidates}
    
    # Generate answer
    answer = generate_answer(query, query_type, passage_details)
    print(f"\nAnswer:")
    print("=" * 80)
    print(answer)
    
    # Format and display results
    print(f"\nTop {len(passage_details)} results:")
    print("=" * 80)
    print(format_results(passage_details, score_map))
    
    conn.close()

if __name__ == "__main__":
    main()









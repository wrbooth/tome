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
import json
from typing import List, Dict, Any, Tuple, Optional
from meilisearch import Client
from sentence_transformers import SentenceTransformer
import openai

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
    """
    Use OpenAI to classify query type based on semantic understanding.
    Returns: 'who', 'when', 'where', 'factoid', 'company', 'person', or 'general'
    """
    try:
        # Ensure .env is loaded
        load_dotenv()
        
        # Check if OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment")
            return 'general'
        
        system_prompt = """You are a query classifier for a historical document search system. 
        Classify the query type based on what the user is asking for:

        - 'who': Questions asking about identity or role (e.g., "Who was the first president to...")
        - 'when': Questions about timing, dates, or years (e.g., "When did...", "What year...")
        - 'where': Questions about locations or places (e.g., "Where was...", "What was the old name for...")
        - 'person': Questions about specific individuals, their actions, or personal history (e.g., "Was [person] ever in combat?", "Did the [family] arrive...")
        - 'company': Questions about businesses, organizations, corporations, or institutions (e.g., "What company built...", "What was the name of the company...")
        - 'factoid': Questions about specific facts, events, or details (e.g., "What had the men done...", "What was the name of...")
        - 'general': General information requests that don't fit other categories

        Return only the classification label, nothing else."""

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this query: {query}"}
            ],
            max_tokens=10,
            temperature=0
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate the classification
        valid_types = ['who', 'when', 'where', 'factoid', 'company', 'person', 'general']
        if classification in valid_types:
            return classification
        else:
            print(f"Warning: Invalid classification '{classification}', using 'general'")
            return 'general'
            
    except Exception as e:
        print(f"Error in OpenAI classification: {e}, using 'general'")
        return 'general'

def extract_person_from_query(query: str) -> Optional[str]:
    """Use OpenAI to extract person name from queries."""
    try:
        load_dotenv()
        if not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY not found, returning None")
            return None
        
        system_prompt = """You are a person name extractor for a historical document search system.
        Extract the name of the person being asked about in the query.
        
        Examples:
        - "Who was the first president to..." -> "first president"
        - "Was [person] ever in combat?" -> "[person]"
        - "Did the [family] arrive..." -> "[family]"
        - "What did [person] do?" -> "[person]"
        
        Return only the person name, nothing else. If no specific person is mentioned, return "none"."""
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract person name from: {query}"}
            ],
            max_tokens=20,
            temperature=0
        )
        
        person = response.choices[0].message.content.strip()
        if person.lower() == "none":
            return None
        return person
        
    except Exception as e:
        print(f"Error in OpenAI person extraction: {e}, returning None")
        return None

def extract_entities_from_query(query: str) -> Dict[str, List[str]]:
    """Use OpenAI to extract entities from query for expansion."""
    entities = {
        "persons": [],
        "places": [],
        "events": [],
        "dates": [],
        "families": [],
        "companies": [],
        "industries": [],
        "settlement_terms": []
    }
    
    try:
        load_dotenv()
        if not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY not found, returning empty entities")
            return entities
        
        system_prompt = """You are an entity extractor for a historical document search system.
        Extract entities from the query and categorize them into the following types:
        
        - persons: Individual people mentioned (e.g., "[person name]", "[historical figure]")
        - places: Locations, cities, counties, states, countries (e.g., "[county name]", "[city name]", "[state name]")
        - events: Historical events or incidents (e.g., "[event name]", "[historical incident]")
        - dates: Years, decades, centuries mentioned (e.g., "[year]", "[decade]s", "[century] century")
        - families: Family names or groups (e.g., "[family name] family", "[surname] family")
        - companies: Businesses, organizations, corporations (e.g., "[company name]", "[organization name]")
        - industries: Industry types or sectors (e.g., "steel", "coal", "manufacturing")
        - settlement_terms: Terms related to settlement, immigration, arrival (e.g., "settlers", "arrived", "founded")
        
        IMPORTANT: 
        1. For temporal queries asking "what years", "when", or "what time", include relevant historical periods or timeframes that might be relevant for the search. For example, if asking about early settlers in the 1800s, include "1800s" or "19th century" in dates. If the query asks about "early settlers" or "first settlers", this typically refers to the early 1800s (1800-1850) or 19th century.
        2. For "who" questions about roles or positions (e.g., "first president", "first sitting president"), include relevant historical figures who might fit that description. For example, if asking about "first sitting president", consider including early presidents who might fit that description.
        
        Return a JSON object with these categories as keys and arrays of extracted entities as values.
        Example:
        {
            "persons": ["[person name]"],
            "places": ["[location name]"],
            "events": [],
            "dates": ["[year]", "[year]"],
            "families": [],
            "companies": [],
            "industries": [],
            "settlement_terms": ["settlers", "arrived"]
        }
        
        Only include categories that have entities. If a category is empty, omit it from the JSON."""
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract entities from: {query}"}
            ],
            max_tokens=300,
            temperature=0
        )
        
        try:
            extracted_entities = json.loads(response.choices[0].message.content.strip())
            
            # Update our entities dict with extracted values
            for category, values in extracted_entities.items():
                if category in entities and isinstance(values, list):
                    entities[category].extend(values)
            
            return entities
            
        except json.JSONDecodeError as e:
            print(f"Error parsing OpenAI entity extraction JSON: {e}, returning empty entities")
            return entities
        
    except Exception as e:
        print(f"Error in OpenAI entity extraction: {e}, returning empty entities")
        return entities



def generate_query_expansions_openai(query: str, query_type: str) -> List[str]:
    """Use OpenAI to generate contextually relevant query expansions."""
    try:
        # Ensure .env is loaded
        load_dotenv()
        
        # Check if OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            print("Warning: OPENAI_API_KEY not found, returning original query only")
            return [query]
        
        system_prompt = f"""You are a query expansion expert for a historical document search system. 
        Given a query and its type, generate 5-10 additional search terms that would help find relevant passages.
        
        Query type: {query_type}
        Original query: "{query}"
        
        Generate search terms that:
        1. Include synonyms and related concepts
        2. Add specific names, dates, or places mentioned or implied
        3. Include broader and narrower terms
        4. Add historical context terms
        5. Include alternative phrasings
        6. ALWAYS include key entity combinations (e.g., "[location] settlers" for queries about settlers from a specific location)
        7. For temporal queries about settlers/immigration, include terms like "settlers", "immigration", "arrival", "founding"
        8. For "who" questions about roles or positions, focus on the role/position terms and avoid geographic confusion (e.g., for "president visiting [city]", focus on "president", "visit", "[city]" not "[city] University")
        9. Be specific to the context - if asking about a specific city, avoid terms that would match other cities with the same name
        10. For questions about "first" or pioneering individuals in roles/positions, include terms about early examples, role-specific visits, and historical precedents
        
        Return only the search terms, one per line, without numbering or explanations.
        Start with the original query, then add expansions."""

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate search expansions for: {query}"}
            ],
            max_tokens=200,
            temperature=0
        )
        
        expansions_text = response.choices[0].message.content.strip()
        
        # Parse the response into a list
        expansions = [line.strip() for line in expansions_text.split('\n') if line.strip()]
        
        # Ensure the original query is included
        if query not in expansions:
            expansions.insert(0, query)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expansions = []
        for exp in expansions:
            if exp.lower() not in seen:
                seen.add(exp.lower())
                unique_expansions.append(exp)
        
        return unique_expansions[:15]  # Limit to 15 expansions to avoid overwhelming
        
    except Exception as e:
        print(f"Error in OpenAI query expansion: {e}, returning original query only")
        return [query]



def generate_query_expansions(query: str, query_type: str) -> List[str]:
    """Main query expansion function - uses OpenAI with fallback."""
    return generate_query_expansions_openai(query, query_type)

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
    """Get embedding for query text using OpenAI."""
    try:
        # Ensure .env is loaded
        load_dotenv()
        
        # Check if OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment")
            return None
        
        client = openai.OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            encoding_format="float"
        )
        return response.data[0].embedding
        
    except Exception as e:
        print(f"Warning: Failed to get query embedding: {e}")
        return None

def apply_query_boosting(candidates: List[Tuple[str, float]], query: str, conn) -> List[Tuple[str, float]]:
    """Apply query-specific boosting to candidates."""
    
    query_type = detect_query_type(query)
    entities = extract_entities_from_query(query)
    
    if query_type == 'who':
        person = extract_person_from_query(query)
        boosted_candidates = []
        
        for passage_id, score in candidates:
            boost_multiplier = 1.0
            
            # Check if passage has matching person entity
            with conn.cursor() as cur:
                # If we have a specific person from the query
                if person:
                    cur.execute("""
                        SELECT 1 FROM passage_entities 
                        WHERE passage_id = %s 
                        AND ent_type = 'PERSON' 
                        AND norm_entity = %s
                    """, (passage_id, person.lower()))
                    
                    if cur.fetchone():
                        # Boost score for matching person
                        boost_multiplier *= 1.5
                
                                # For role-based "who" questions, boost passages with PERSON entities
                # This is more generic and scalable than hardcoding specific names
                if query_type == 'who':
                    cur.execute("""
                        SELECT COUNT(*) FROM passage_entities 
                        WHERE passage_id = %s 
                        AND ent_type = 'PERSON'
                    """, (passage_id,))
                    
                    person_count = cur.fetchone()[0]
                    if person_count > 0:
                        # Boost for passages with person entities (more people = more relevant for "who" questions)
                        boost_multiplier *= (1.0 + (person_count * 0.2))  # 20% boost per person entity
            
            boosted_score = score * boost_multiplier
            boosted_candidates.append((passage_id, boosted_score))
        
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
                        # Handle century references like "1800s" or "19th century"
                        if date.endswith('s') and date[:-1].isdigit():
                            # Convert "1800s" to range 1800-1899
                            century_start = int(date[:-1])
                            century_end = century_start + 99
                            cur.execute("""
                                SELECT COUNT(*) FROM passage_years 
                                WHERE passage_id = %s AND year BETWEEN %s AND %s
                            """, (passage_id, century_start, century_end))
                        elif "century" in date.lower():
                            # Handle "19th century" -> 1800-1899
                            if "19th" in date.lower():
                                cur.execute("""
                                    SELECT COUNT(*) FROM passage_years 
                                    WHERE passage_id = %s AND year BETWEEN 1800 AND 1899
                                """, (passage_id,))
                            elif "18th" in date.lower():
                                cur.execute("""
                                    SELECT COUNT(*) FROM passage_years 
                                    WHERE passage_id = %s AND year BETWEEN 1700 AND 1799
                                """, (passage_id,))
                            elif "20th" in date.lower():
                                cur.execute("""
                                    SELECT COUNT(*) FROM passage_years 
                                    WHERE passage_id = %s AND year BETWEEN 1900 AND 1999
                                """, (passage_id,))
                            else:
                                continue
                        elif date.isdigit():
                            # Regular year
                            cur.execute("""
                                SELECT COUNT(*) FROM passage_years 
                                WHERE passage_id = %s AND year = %s
                            """, (passage_id, int(date)))
                        else:
                            continue
                        
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
                
                # Check for settlement-related terms in the query
                if entities.get("settlement_terms"):
                    settlement_boost = False
                    for term in entities["settlement_terms"]:
                        # Check if passage contains settlement-related entities
                        cur.execute("""
                            SELECT 1 FROM passage_entities 
                            WHERE passage_id = %s 
                            AND (ent_type = 'ORG' OR ent_type = 'GPE')
                            AND (norm_entity LIKE %s OR norm_entity LIKE %s OR norm_entity LIKE %s)
                        """, (passage_id, f"%settler%", f"%arriv%", f"%settlers%"))
                        
                        if cur.fetchone():
                            # Boost for settlement-related content
                            boost_multiplier *= 1.4
                            settlement_boost = True
                            break
                    
                    # Additional boost for passages with both settlement terms AND years
                    if settlement_boost and year_count > 0:
                        boost_multiplier *= 1.2
                
                # Special boost for settlement-related organization entities
                cur.execute("""
                    SELECT 1 FROM passage_entities 
                    WHERE passage_id = %s 
                    AND ent_type = 'ORG' 
                    AND norm_entity LIKE %s
                """, (passage_id, f"%settler%"))
                
                if cur.fetchone():
                    # Strong boost for settlement organization entities
                    boost_multiplier *= 1.8
            
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
    
    elif query_type == 'person':
        # Boost passages with specific person names
        boosted_candidates = []
        
        for passage_id, score in candidates:
            boost_multiplier = 1.0
            
            # Check if passage has the specific person mentioned
            with conn.cursor() as cur:
                # Initialize variables
                person_found = False
                
                # Check for person names in entities
                if entities.get("persons"):
                    for person in entities["persons"]:
                        # More specific matching for person names
                        cur.execute("""
                            SELECT norm_entity FROM passage_entities 
                            WHERE passage_id = %s 
                            AND ent_type = 'PERSON' 
                            AND norm_entity LIKE %s
                        """, (passage_id, f"%{person.lower()}%"))
                        
                        if cur.fetchone():
                            # Boost for person entity match
                            boost_multiplier *= 2.0
                            person_found = True
                            break
            
            # For person queries, we've already applied the boost above
            
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
        """, passage_ids)
        
        results = cur.fetchall()
        
        # Preserve the order of passage_ids
        id_to_result = {result['id']: result for result in results}
        ordered_results = []
        for passage_id in passage_ids:
            if passage_id in id_to_result:
                ordered_results.append(id_to_result[passage_id])
        
        return ordered_results

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
    
    # Re-sort by boosted scores
    boosted_candidates = sorted(boosted_candidates, key=lambda x: x[1], reverse=True)
    
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









#!/usr/bin/env python3
"""
LLM-based answer generation for search results.
"""

import json
from typing import List, Dict, Any, Optional

from config import get_openai_client, ANSWER_MODEL

client = get_openai_client()

def format_chunks_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """
    Format search chunks for LLM consumption.
    
    Args:
        chunks: List of chunk dictionaries from search results
        
    Returns:
        Formatted string with chunks and metadata
    """
    formatted_chunks = []
    
    for i, chunk in enumerate(chunks, 1):
        # Extract relevant information
        text = chunk.get('text', '')
        page = chunk.get('page', 'Unknown')
        title = chunk.get('title', 'Unknown Document')
        
        # The text already contains document title and headings in the prefix
        # Format each chunk with additional context
        chunk_text = f"CHUNK {i} (Document: {title}, Page {page}):\n{text}\n"
        formatted_chunks.append(chunk_text)
    
    return "\n".join(formatted_chunks)

def generate_answer_with_llm(query: str, chunks: List[Dict[str, Any]], model: str = ANSWER_MODEL) -> Dict[str, Any]:
    """
    Generate an answer using LLM based on provided chunks.
    
    Args:
        query: User's question
        chunks: List of search result chunks
        model: LLM model to use
        
    Returns:
        Dictionary with answer and metadata
    """
    if not chunks:
        return {
            "answer": "I cannot provide an answer as no relevant information was found in the search results.",
            "sources": [],
            "confidence": "none"
        }
    
    # Format chunks for LLM
    chunks_text = format_chunks_for_llm(chunks)
    
    # Create the system prompt with strict instructions
    system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided text chunks. 

CRITICAL RULES:
1. ONLY use information from the provided chunks. Do not use any external knowledge.
2. If the answer cannot be found in the chunks, say "I cannot answer this question based on the provided information." But do talk about the information you do have.
3. Do not make assumptions or inferences beyond what is explicitly stated in the chunks.
4. Always provide source references at the end of your answer in this format:
   Sources: [Document Title, Page X; Document Title, Page Y]
5. Be concise but thorough in your answer.
6. If multiple chunks contain relevant information, synthesize them clearly.
7. For yes/no questions, be extremely precise about timing and conditions. If a question asks "Did X happen in YEAR Y?" and X happened in YEAR Z (different from Y), the answer is "No."
8. When information comes from multiple documents, clearly indicate which document each piece of information comes from.


The user will provide a question and relevant text chunks. Answer based ONLY on those chunks."""

    # Create the user prompt
    user_prompt = f"""Question: {query}

Relevant information from the document:

{chunks_text}

Please answer the question based ONLY on the information provided above. If the answer is not in the chunks, say so clearly."""

    try:
        # Call the LLM
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            #temperature=0.1,  # Low temperature for more consistent, factual responses
            max_completion_tokens=5000
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract source pages from the answer
        sources = extract_source_pages(answer, chunks)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": "high" if chunks else "none"
        }
        
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "confidence": "error"
        }

def extract_source_pages(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract source information mentioned in the answer.
    
    Args:
        answer: LLM-generated answer
        chunks: Original chunks used for generation
        
    Returns:
        List of source dictionaries with title and page
    """
    # Look for "Sources:" mentions in the answer
    import re
    
    # Extract sources from "Sources: [Document Title, Page X; Document Title, Page Y]" format
    sources_match = re.search(r'Sources:\s*\[(.*?)\]', answer, re.IGNORECASE)
    if sources_match:
        sources_text = sources_match.group(1)
        # Parse "Document Title, Page X" format
        sources = []
        for source in sources_text.split(';'):
            source = source.strip()
            if ',' in source:
                parts = source.split(',')
                if len(parts) >= 2:
                    title = parts[0].strip()
                    page_match = re.search(r'Page\s+(\d+)', parts[1], re.IGNORECASE)
                    if page_match:
                        sources.append({
                            'title': title,
                            'page': int(page_match.group(1))
                        })
        return sources
    
    # Fallback: return sources from chunks that were used
    return [{'title': chunk.get('title', 'Unknown'), 'page': chunk.get('page', 0)} for chunk in chunks]

def format_answer_with_sources(answer_data: Dict[str, Any]) -> str:
    """
    Format the final answer with proper source attribution.
    
    Args:
        answer_data: Dictionary with answer and sources
        
    Returns:
        Formatted answer string
    """
    answer = answer_data.get("answer", "")
    sources = answer_data.get("sources", [])
    
    # If sources are already mentioned in the answer, return as-is
    if "Sources:" in answer or "sources:" in answer:
        return answer
    
    # Otherwise, add source information
    if sources:
        if isinstance(sources[0], dict):
            # New format with document titles
            source_text = "; ".join([f"{s['title']}, Page {s['page']}" for s in sources])
        else:
            # Legacy format with just page numbers
            source_text = ", ".join([f"Page {p}" for p in sorted(sources)])
        return f"{answer}\n\nSources: {source_text}"
    else:
        return answer

def answer_query(query: str, search_results: List[Dict[str, Any]], model: str = ANSWER_MODEL) -> str:
    """
    Main function to answer a query using search results and LLM.
    
    Args:
        query: User's question
        search_results: List of search result dictionaries
        model: LLM model to use
        
    Returns:
        Formatted answer with sources
    """
    # Extract chunks from search results
    chunks = []
    for result in search_results:
        chunk = {
            'text': result.get('text', ''),
            'page': result.get('page', 0),
            'title': result.get('title', 'Unknown')
        }
        chunks.append(chunk)
    
    # Generate answer using LLM
    answer_data = generate_answer_with_llm(query, chunks, model)
    
    # Format and return the answer
    return format_answer_with_sources(answer_data)

if __name__ == "__main__":
    # Test the answer generator
    test_query = "What sort of things did Morgan's Raiders steal?"
    test_chunks = [
        {
            'text': 'Appendix E-1 — Newspaper account of Morgan\'s Raid | every horse they met with that was of any value, and when they stole a horse they generally turned loose some poor tired-out animal.',
            'page': 63,
            'title': 'A Brief History of Guernsey County'
        }
    ]
    
    result = answer_query(test_query, test_chunks)
    print("Test Answer:")
    print(result)

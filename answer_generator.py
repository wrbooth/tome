#!/usr/bin/env python3
"""
LLM-based answer generation for search results.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        
        # Format each chunk
        chunk_text = f"CHUNK {i} (Page {page}):\n{text}\n"
        formatted_chunks.append(chunk_text)
    
    return "\n".join(formatted_chunks)

def generate_answer_with_llm(query: str, chunks: List[Dict[str, Any]], model: str = "gpt-4") -> Dict[str, Any]:
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
2. If the answer cannot be found in the chunks, say "I cannot answer this question based on the provided information."
3. Do not make assumptions or inferences beyond what is explicitly stated in the chunks.
4. Always provide source references at the end of your answer in this format:
   Sources: [Page X, Page Y, Page Z]
5. Be concise but thorough in your answer.
6. If multiple chunks contain relevant information, synthesize them clearly.

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
            temperature=0.1,  # Low temperature for more consistent, factual responses
            max_tokens=1000
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

def extract_source_pages(answer: str, chunks: List[Dict[str, Any]]) -> List[int]:
    """
    Extract page numbers mentioned in the answer.
    
    Args:
        answer: LLM-generated answer
        chunks: Original chunks used for generation
        
    Returns:
        List of page numbers referenced
    """
    # Look for "Sources:" or "Page" mentions in the answer
    import re
    
    # Extract page numbers from "Sources: [Page X, Page Y, Page Z]" format
    sources_match = re.search(r'Sources:\s*\[(.*?)\]', answer, re.IGNORECASE)
    if sources_match:
        pages_text = sources_match.group(1)
        page_numbers = re.findall(r'Page\s+(\d+)', pages_text, re.IGNORECASE)
        return [int(p) for p in page_numbers]
    
    # Fallback: return pages from chunks that were used
    return list(set(chunk.get('page', 0) for chunk in chunks))

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
        source_text = ", ".join([f"Page {p}" for p in sorted(sources)])
        return f"{answer}\n\nSources: {source_text}"
    else:
        return answer

def answer_query(query: str, search_results: List[Dict[str, Any]], model: str = "gpt-4") -> str:
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

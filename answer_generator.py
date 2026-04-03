#!/usr/bin/env python3
"""
LLM-based answer generation for search results.
"""

import logging
from typing import Generator, List, Dict, Any, Optional

from config import get_openai_client, ANSWER_MODEL

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazily create the OpenAI client on first use."""
    global _client
    if _client is None:
        _client = get_openai_client()
    return _client

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

def _build_messages(query: str, chunks_text: str) -> List[Dict[str, str]]:
    """Build the system + user messages for answer generation."""
    system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided text chunks.

CRITICAL RULES:
1. ONLY use information from the provided chunks. Do not use any external knowledge.
2. If the answer cannot be found in the chunks, say "I cannot answer this question based on the provided information." But do talk about the information you do have.
3. Do not make assumptions or inferences beyond what is explicitly stated in the chunks.
4. For yes/no questions, be extremely precise about timing and conditions. If a question asks "Did X happen in YEAR Y?" and X happened in YEAR Z (different from Y), the answer is "No."
5. When information comes from multiple documents, clearly indicate which document each piece of information comes from.

FORMATTING RULES:
- You MUST use proper Markdown. Your output is rendered as Markdown.
- Start with a brief 1-2 sentence summary paragraph.
- Then use a bulleted list (using "- " at the start of each line) for individual points, items, people, or events. Each bullet should be its own line.
- Use **bold** for key names, places, and dates within bullets.
- Use "## Heading" for major sections if the answer covers distinct topics.
- NEVER write a single long paragraph. Always break information into bullets or short paragraphs separated by blank lines.
- Cite sources inline: [Document Title, Page X] immediately after the relevant fact.

Here is an example of a well-formatted answer:

The county had several important early settlers who shaped its development.

- **John Smith** arrived in 1798 and established the first trading post [County History, Page 12]
- **Mary Jones** founded the first school in 1802 [County History, Page 15]
- **Robert Brown** served as the first county commissioner from 1810 to 1815 [County Records, Page 23]

The user will provide a question and relevant text chunks. Answer based ONLY on those chunks."""

    user_prompt = f"""Question: {query}

Relevant information from the document:

{chunks_text}

Please answer the question based ONLY on the information provided above. If the answer is not in the chunks, say so clearly."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def stream_answer_with_llm(query: str, chunks: List[Dict[str, Any]], model: str = ANSWER_MODEL) -> Generator[str, None, None]:
    """
    Stream an answer token-by-token using the OpenAI streaming API.

    Yields:
        Individual token strings as they arrive from the LLM.
    """
    if not chunks:
        yield "I cannot provide an answer as no relevant information was found in the search results."
        return

    chunks_text = format_chunks_for_llm(chunks)
    messages = _build_messages(query, chunks_text)

    try:
        stream = _get_client().chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=5000,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"Error generating answer: {str(e)}"


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
    messages = _build_messages(query, chunks_text)

    try:
        # Call the LLM
        response = _get_client().chat.completions.create(
            model=model,
            messages=messages,
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
    
    from config import configure_logging
    configure_logging()
    result = answer_query(test_query, test_chunks)
    logger.info("Test Answer:")
    logger.info(result)

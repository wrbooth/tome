"""
Entity and year extraction for the Tome ingestion pipeline.

Uses spaCy NER for entity extraction and regex for year detection.
"""

import logging
import re

logger = logging.getLogger(__name__)

_nlp = None
_nlp_loaded = False


def _get_nlp():
    """Lazily load the spaCy model on first use."""
    global _nlp, _nlp_loaded
    if not _nlp_loaded:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            _nlp = None
        _nlp_loaded = True
    return _nlp


def extract_entities_and_years(text: str) -> tuple[list[dict[str, str]], list[int]]:
    """Extract entities and years from text."""
    entities = []
    years = []

    # Extract years using regex
    year_pattern = r"\b(1[4-9]\d{2}|20\d{2})\b"
    years = [int(year) for year in re.findall(year_pattern, text) if year.strip()]

    # Also extract decade references like "1940s", "1950s", etc.
    decade_pattern = r"\b(1[4-9]\d{2}s|20\d{2}s)\b"
    decade_matches = re.findall(decade_pattern, text)
    for decade in decade_matches:
        # Convert "1940s" to 1940, "1950s" to 1950, etc.
        base_year = int(decade[:-1])  # Remove 's' and convert to int
        years.append(base_year)

    # Extract entities using spaCy
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        entities.extend(
            {
                "entity": ent.text,
                "ent_type": ent.label_,
                "norm_entity": ent.text.lower(),
            }
            for ent in doc.ents
        )

    return entities, years

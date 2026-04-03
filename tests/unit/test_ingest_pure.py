"""Tests for pure functions in ingest.py."""

from unittest.mock import patch

from codex.ingestion.ingest import (
    count_tokens,
    detect_heading_patterns,
    extract_entities_and_years,
    filter_appendices,
    is_quality_heading,
    merge_heading_detection,
    merge_heading_results,
    rank_headings_by_relevance,
)


def _rank_fallback(query, headings, limit=10):
    """Exercise the simple string-matching fallback in rank_headings_by_relevance
    by making rapidfuzz un-importable."""
    import sys

    # Temporarily hide rapidfuzz
    saved = {}
    for mod_name in list(sys.modules):
        if mod_name.startswith("rapidfuzz"):
            saved[mod_name] = sys.modules.pop(mod_name)
    try:
        # Also block fresh imports
        import builtins

        real_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if name == "rapidfuzz" or name.startswith("rapidfuzz."):
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = patched_import
        try:
            return rank_headings_by_relevance(query, headings, limit)
        finally:
            builtins.__import__ = real_import
    finally:
        sys.modules.update(saved)


# ── count_tokens ──────────────────────────────────────────────────────────


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        result = count_tokens("hello")
        assert result == 1

    def test_known_sentence(self):
        result = count_tokens("The quick brown fox jumps over the lazy dog.")
        assert result == 10

    def test_whitespace_only(self):
        result = count_tokens("   ")
        assert result <= 1

    def test_unicode(self):
        result = count_tokens("Caf\u00e9 na\u00efve r\u00e9sum\u00e9")
        assert result > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens(
            "hello world this is a longer sentence with many more words"
        )
        assert long > short


# ── detect_heading_patterns ───────────────────────────────────────────────


class TestDetectHeadingPatterns:
    def test_chapter_uppercase(self):
        text = "CHAPTER 1: The Beginning of Time"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 1

    def test_chapter_titlecase(self):
        text = "Chapter 2: The Middle Years"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 1

    def test_part_pattern(self):
        text = "PART 3: Final Analysis"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 1

    def test_book_pattern(self):
        text = "Book 1: The Origins"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 1

    def test_appendix_uppercase(self):
        text = "APPENDIX A: Supporting Data Tables"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 1

    def test_appendix_with_dash_number(self):
        text = "APPENDIX E-4: Morgan's Raid Claims"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1

    def test_all_caps_long_title_rejected_by_quality_check(self):
        """ALL CAPS with only spaces is rejected by is_quality_heading noise pattern."""
        text = "INTRODUCTION AND BACKGROUND"
        headings = detect_heading_patterns(text)
        # The regex matches but is_quality_heading filters it (^[A-Z\s]+$ noise pattern)
        assert len(headings) == 0

    def test_section_level2(self):
        text = "Section 1.1: Background Information"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 2

    def test_numbered_section_level2(self):
        text = "1.2 Methodology Overview"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 2

    def test_numbered_subsection_level3(self):
        text = "1.1.1 Detailed Subsection"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["level"] == 3

    def test_only_level_1_filters_lower_levels(self):
        text = "CHAPTER 1: Main Heading\n1.2 Subsection Title"
        all_headings = detect_heading_patterns(text, only_level_1=False)
        level1_headings = detect_heading_patterns(text, only_level_1=True)
        assert len(level1_headings) <= len(all_headings)
        for h in level1_headings:
            assert h["level"] == 1

    def test_no_headings_in_plain_text(self):
        text = "this is just a regular sentence with no heading patterns at all."
        headings = detect_heading_patterns(text)
        assert headings == []

    def test_multiple_headings_in_text(self):
        text = "CHAPTER 1: First Part\nSome body text here.\nCHAPTER 2: Second Part"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 2

    def test_empty_string(self):
        headings = detect_heading_patterns("")
        assert headings == []

    def test_heading_has_detection_method(self):
        text = "CHAPTER 1: Introduction Section"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["detection_method"] == "regex"

    def test_heading_has_line_number(self):
        text = "Some text\nCHAPTER 1: Introduction Part"
        headings = detect_heading_patterns(text)
        assert len(headings) >= 1
        assert headings[0]["line_number"] == 1  # second line (0-indexed)


# ── is_quality_heading ────────────────────────────────────────────────────


class TestIsQualityHeading:
    def test_too_short(self):
        assert is_quality_heading("Hi", 12, False) is False

    def test_too_long(self):
        assert is_quality_heading("A" * 201, 12, False) is False

    def test_high_symbol_ratio(self):
        assert is_quality_heading("###!!$$$abc", 12, False) is False

    def test_high_digit_ratio(self):
        assert is_quality_heading("12345abc", 12, False) is False

    def test_only_numbers(self):
        assert is_quality_heading("12345", 12, False) is False

    def test_single_letter(self):
        assert is_quality_heading("A", 12, False) is False

    def test_very_short_all_caps(self):
        assert is_quality_heading("AB", 12, False) is False

    def test_table_pattern_name_location(self):
        assert is_quality_heading("Name, Location", 12, False) is False

    def test_table_pattern_page_number(self):
        assert is_quality_heading("Page 5", 12, False) is False

    def test_table_pattern_decimal(self):
        assert is_quality_heading("3.00", 12, False) is False

    def test_consecutive_symbols(self):
        assert is_quality_heading("hello---world", 12, False) is False

    def test_single_word(self):
        assert is_quality_heading("Introduction", 12, False) is False

    def test_valid_heading(self):
        assert is_quality_heading("The Introduction Section", 12, False) is True

    def test_valid_heading_mixed_case(self):
        assert is_quality_heading("About The Civil War", 12, False) is True

    def test_appendix_heading_allowed(self):
        """Appendix headings pass quality check via generic APPENDIX pattern."""
        assert is_quality_heading("APPENDIX A: Supporting Data", 12, False) is True
        assert is_quality_heading("APPENDIX E-4 Some Title", 12, False) is True

    def test_all_caps_with_spaces_only(self):
        # Pattern: ^[A-Z\s]+$ — ALL CAPS with only spaces
        assert is_quality_heading("ALL CAPS ONLY", 12, False) is False

    def test_coordinate_like_text(self):
        assert is_quality_heading("A5B coordinate", 12, False) is False

    def test_no_letters_at_all(self):
        """Line 252: text with no alphabetic characters."""
        assert is_quality_heading("12 34 56", 12, False) is False

    def test_short_bold_large_font_still_needs_length(self):
        """Line 256: text < 5 chars but bold + large font --
        still fails (< 5 check on line 210)."""
        assert is_quality_heading("Go", 14, True) is False

    def test_length_exactly_5_passes_length_check(self):
        """Text exactly 5 chars passes the length check but may fail other checks."""
        # "Ab Cd" is 5 chars, has letters, 2 words — should pass
        assert is_quality_heading("Ab Cd", 12, False) is True

    def test_high_punctuation_ratio_second_check(self):
        """Line 265: punctuation > 30% (different from symbol check on line 215)."""
        # 5 symbols out of 13 non-space chars = ~38%
        assert is_quality_heading("H!e@l#l$o W%orld", 12, False) is False

    def test_number_then_letter_coordinate(self):
        """Line 269 first pattern: digit+symbol+uppercase like '5A'."""
        assert is_quality_heading("Item 5A reference", 12, False) is False

    def test_valid_two_word_heading(self):
        """Ensure a clean 2-word heading passes all checks."""
        assert is_quality_heading("Historical Overview", 12, False) is True

    def test_pattern_number_word(self):
        """Table pattern: '1 horse' — starts with digit then lowercase."""
        assert is_quality_heading("1 horse", 12, False) is False


# ── merge_heading_detection ───────────────────────────────────────────────


class TestMergeHeadingDetection:
    def test_pages_without_headings_get_regex(self):
        pages = [{"page": 1, "text": "CHAPTER 1: Test Heading\nBody text."}]
        result = merge_heading_detection(pages)
        assert len(result) == 1
        # Should have added headings via regex
        assert "headings" in result[0]

    def test_pages_with_headings_merge_non_conflicting(self):
        pages = [
            {
                "page": 1,
                "text": "CHAPTER 1: Existing Heading\nSome body text here.",
                "headings": [
                    {
                        "text": "Existing Heading",
                        "line_number": 0,
                        "level": 1,
                        "detection_method": "font",
                    }
                ],
            }
        ]
        result = merge_heading_detection(pages)
        # Should have at least the original heading
        assert len(result[0]["headings"]) >= 1

    def test_pdf_mode_filters_to_level_1_and_2(self):
        pages = [
            {
                "page": 1,
                "text": "CHAPTER 1: Level One\nSome text.\n1.1.1 Level Three Item",
                "headings": [
                    {
                        "text": "Level One",
                        "line_number": 0,
                        "level": 1,
                        "detection_method": "font",
                    },
                    {
                        "text": "Level Three Item",
                        "line_number": 2,
                        "level": 3,
                        "detection_method": "font",
                    },
                ],
            }
        ]
        result = merge_heading_detection(pages, is_pdf=True)
        levels = [h["level"] for h in result[0]["headings"]]
        assert all(level in [1, 2] for level in levels)

    def test_non_pdf_keeps_all_levels(self):
        pages = [
            {
                "page": 1,
                "text": "CHAPTER 1: Level One\nSome text.\n1.1.1 Level Three Item",
                "headings": [
                    {
                        "text": "Level One",
                        "line_number": 0,
                        "level": 1,
                        "detection_method": "font",
                    },
                    {
                        "text": "Level Three Item",
                        "line_number": 2,
                        "level": 3,
                        "detection_method": "font",
                    },
                ],
            }
        ]
        result = merge_heading_detection(pages, is_pdf=False)
        levels = [h["level"] for h in result[0]["headings"]]
        assert 3 in levels

    def test_headings_sorted_by_line_number(self):
        pages = [
            {
                "page": 1,
                "text": (
                    "Second heading text.\n"
                    "First heading text.\n"
                    "CHAPTER 1: Third Heading"
                ),
                "headings": [
                    {
                        "text": "Second",
                        "line_number": 5,
                        "level": 1,
                        "detection_method": "font",
                    },
                    {
                        "text": "First",
                        "line_number": 1,
                        "level": 1,
                        "detection_method": "font",
                    },
                ],
            }
        ]
        result = merge_heading_detection(pages, is_pdf=True)
        line_nums = [h["line_number"] for h in result[0]["headings"]]
        assert line_nums == sorted(line_nums)

    def test_returns_pages_list(self):
        pages = [{"page": 1, "text": "Just text."}]
        result = merge_heading_detection(pages)
        assert isinstance(result, list)
        assert len(result) == 1


# ── extract_entities_and_years ────────────────────────────────────────────


class TestExtractEntitiesAndYears:
    def test_extracts_four_digit_year(self):
        _, years = extract_entities_and_years("The war ended in 1865.")
        assert 1865 in years

    def test_extracts_multiple_years(self):
        _, years = extract_entities_and_years(
            "From 1776 to 1783, the revolution raged."
        )
        assert 1776 in years
        assert 1783 in years

    def test_extracts_decade_pattern(self):
        _, years = extract_entities_and_years("During the 1940s, many things changed.")
        assert 1940 in years

    def test_out_of_range_year_rejected(self):
        _, years = extract_entities_and_years("In the year 1399 nothing happened.")
        assert 1399 not in years

    def test_future_year_rejected(self):
        _, years = extract_entities_and_years("In 2100 the future arrives.")
        assert 2100 not in years

    def test_no_years_returns_empty(self):
        _, years = extract_entities_and_years("No dates mentioned here at all.")
        assert years == [], f"Expected empty list but got {years}"

    def test_entities_have_correct_keys(self):
        entities, _ = extract_entities_and_years(
            "George Washington visited Philadelphia in 1789."
        )
        for entity in entities:
            assert "entity" in entity
            assert "ent_type" in entity
            assert "norm_entity" in entity

    def test_norm_entity_is_lowercase(self):
        entities, _ = extract_entities_and_years(
            "George Washington was the first president."
        )
        for entity in entities:
            assert entity["norm_entity"] == entity["entity"].lower()

    def test_nlp_none_returns_empty_entities(self):
        with patch("codex.ingestion.entities._get_nlp", return_value=None):
            entities, years = extract_entities_and_years("George Washington in 1776.")
            assert entities == []
            assert 1776 in years  # years still extracted via regex

    def test_boundary_year_1400(self):
        _, years = extract_entities_and_years("In 1400 the era began.")
        assert 1400 in years

    def test_boundary_year_2099(self):
        _, years = extract_entities_and_years("By 2099 we will know.")
        assert 2099 in years


# ── filter_appendices ─────────────────────────────────────────────────────


class TestFilterAppendices:
    def test_matches_appendix_uppercase(self):
        headings = [{"title": "APPENDIX A: Data Tables"}]
        result = filter_appendices(headings)
        assert len(result) == 1

    def test_matches_appendix_titlecase(self):
        headings = [{"title": "Appendix B: Notes"}]
        result = filter_appendices(headings)
        assert len(result) == 1

    def test_matches_appendix_with_dash(self):
        headings = [{"title": "APPENDIX E-4: Morgan's Raid Claims"}]
        result = filter_appendices(headings)
        assert len(result) == 1

    def test_skips_non_appendix(self):
        headings = [
            {"title": "Chapter 1: Introduction"},
            {"title": "APPENDIX A: Data"},
        ]
        result = filter_appendices(headings)
        assert len(result) == 1
        assert "APPENDIX" in result[0]["title"]

    def test_empty_list(self):
        assert filter_appendices([]) == []

    def test_handles_text_key(self):
        headings = [{"text": "APPENDIX C: References"}]
        result = filter_appendices(headings)
        assert len(result) == 1

    def test_no_title_or_text_skipped(self):
        headings = [{"level": 1}]
        result = filter_appendices(headings)
        assert len(result) == 0


# ── merge_heading_results ─────────────────────────────────────────────────


class TestMergeHeadingResults:
    def test_deduplicates_same_page_same_title(self):
        headings = [
            {"title": "Introduction", "page": 0, "confidence": 0.5, "line_number": 0},
            {"title": "introduction", "page": 0, "confidence": 0.9, "line_number": 0},
        ]
        result = merge_heading_results(headings)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_keeps_different_pages(self):
        headings = [
            {"title": "Same Title", "page": 0, "confidence": 0.5, "line_number": 0},
            {"title": "Same Title", "page": 5, "confidence": 0.5, "line_number": 0},
        ]
        result = merge_heading_results(headings)
        assert len(result) == 2

    def test_sorts_by_page_then_line_number(self):
        headings = [
            {"title": "Second", "page": 5, "confidence": 0.5, "line_number": 0},
            {"title": "First", "page": 1, "confidence": 0.5, "line_number": 0},
            {"title": "Third", "page": 5, "confidence": 0.5, "line_number": 10},
        ]
        result = merge_heading_results(headings)
        pages = [h["page"] for h in result]
        assert pages == sorted(pages)

    def test_skips_empty_title(self):
        headings = [
            {"title": "", "page": 0, "confidence": 0.5},
            {"title": "Valid Heading", "page": 0, "confidence": 0.5, "line_number": 0},
        ]
        result = merge_heading_results(headings)
        assert len(result) == 1

    def test_handles_text_key(self):
        headings = [
            {"text": "A Heading", "page": 0, "confidence": 0.5, "line_number": 0},
        ]
        result = merge_heading_results(headings)
        assert len(result) == 1

    def test_empty_list(self):
        assert merge_heading_results([]) == []

    def test_missing_confidence_key_uses_default(self):
        """When first heading has no confidence key and second does,
        the second heading (with higher confidence) wins."""
        headings = [
            {"title": "No Confidence", "page": 0, "line_number": 0},
            {"title": "no confidence", "page": 0, "confidence": 0.1, "line_number": 0},
        ]
        result = merge_heading_results(headings)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.1

    def test_both_have_confidence(self):
        """When both headings have confidence, higher wins."""
        headings = [
            {"title": "Heading", "page": 0, "confidence": 0.3, "line_number": 0},
            {"title": "heading", "page": 0, "confidence": 0.9, "line_number": 0},
        ]
        result = merge_heading_results(headings)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9


# ── rank_headings_by_relevance ────────────────────────────────────────────


class TestRankHeadingsByRelevance:
    def test_exact_match_high_score(self):
        headings = [
            {"title": "Civil War", "page": 0},
            {"title": "Appendix A", "page": 5},
        ]
        result = rank_headings_by_relevance("Civil War", headings)
        assert len(result) >= 1
        assert result[0]["relevance_score"] > 0.5

    def test_respects_limit(self):
        headings = [{"title": f"Heading {i}", "page": i} for i in range(20)]
        result = rank_headings_by_relevance("Heading", headings, limit=5)
        assert len(result) <= 5

    def test_empty_headings(self):
        result = rank_headings_by_relevance("test query", [])
        assert result == []

    def test_adds_relevance_score_field(self):
        headings = [{"title": "Civil War History", "page": 0}]
        result = rank_headings_by_relevance("Civil War", headings)
        assert len(result) >= 1
        assert "relevance_score" in result[0]

    def test_no_match_returns_empty_or_low_score(self):
        headings = [{"title": "Completely Unrelated Topic", "page": 0}]
        result = rank_headings_by_relevance("quantum physics", headings)
        # Either empty or very low score
        if result:
            assert result[0]["relevance_score"] < 0.5

    def test_handles_text_key(self):
        headings = [{"text": "Morgan's Raid", "page": 3}]
        result = rank_headings_by_relevance("Morgan", headings)
        assert len(result) >= 1

    def test_fallback_exact_match(self):
        """Fallback path line 1026-1027: exact match gives score 1.0."""
        headings = [{"title": "Civil War", "page": 0}]
        with patch.dict(
            "sys.modules",
            {"rapidfuzz": None, "rapidfuzz.process": None, "rapidfuzz.fuzz": None},
        ):
            # Need to force ImportError inside the function

            result = _rank_fallback("Civil War", headings)
            assert result[0]["relevance_score"] == 1.0

    def test_fallback_query_substring_of_title(self):
        """Fallback path line 1029-1030: query is substring of title gives 0.8."""
        headings = [{"title": "The Civil War History", "page": 0}]
        result = _rank_fallback("Civil War", headings)
        assert result[0]["relevance_score"] == 0.8

    def test_fallback_title_substring_of_query(self):
        """Fallback path line 1032-1033: title is substring of query gives 0.7."""
        headings = [{"title": "War", "page": 0}]
        result = _rank_fallback("Civil War History", headings)
        assert result[0]["relevance_score"] == 0.7

    def test_fallback_word_overlap(self):
        """Fallback path lines 1036-1040: word overlap scoring."""
        headings = [{"title": "War and Peace", "page": 0}]
        result = _rank_fallback("Civil War History", headings)
        assert len(result) >= 1
        assert 0 < result[0]["relevance_score"] < 0.7

    def test_fallback_no_overlap(self):
        """Fallback path: no match at all returns empty."""
        headings = [{"title": "Cooking Recipes", "page": 0}]
        result = _rank_fallback("quantum physics", headings)
        assert result == []

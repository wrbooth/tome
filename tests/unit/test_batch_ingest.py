"""Tests for batch_ingest.py."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tome.ingestion.batch_ingest import (
    extract_metadata_from_filename,
    get_document_files,
    ingest_single_document,
    load_metadata_file,
    main,
    process_documents_parallel,
    process_documents_sequential,
)

# ── extract_metadata_from_filename ────────────────────────────────────────


class TestExtractMetadataFromFilename:
    def test_simple_filename(self):
        result = extract_metadata_from_filename("/path/to/My Document.pdf")
        assert result["title"] == "My Document"
        assert result["authors"] is None
        assert result["pub_year"] is None
        assert result["language"] == "en"

    def test_nested_path(self):
        result = extract_metadata_from_filename("/a/b/c/d/Report.txt")
        assert result["title"] == "Report"

    def test_filename_with_spaces(self):
        result = extract_metadata_from_filename("/path/1998 A Brief History.pdf")
        assert result["title"] == "1998 A Brief History"


# ── load_metadata_file ────────────────────────────────────────────────────


class TestLoadMetadataFile:
    def test_json_file(self, tmp_path):
        metadata = [
            {"filename": "doc1.pdf", "title": "Doc 1", "authors": "Author A"},
            {"filename": "doc2.pdf", "title": "Doc 2"},
        ]
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(metadata))
        result = load_metadata_file(str(path))
        assert "doc1.pdf" in result
        assert result["doc1.pdf"]["title"] == "Doc 1"

    def test_csv_file(self, tmp_path):
        csv_content = (
            "filename,title,authors\ndoc1.pdf,Doc 1,Author A\ndoc2.pdf,Doc 2,Author B\n"
        )
        path = tmp_path / "meta.csv"
        path.write_text(csv_content)
        result = load_metadata_file(str(path))
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    def test_invalid_extension_raises(self, tmp_path):
        path = tmp_path / "meta.xml"
        path.write_text("<data/>")
        with pytest.raises(ValueError, match="CSV or JSON"):
            load_metadata_file(str(path))


# ── get_document_files ────────────────────────────────────────────────────


class TestGetDocumentFiles:
    def test_single_pdf_file(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf")
        result = get_document_files(str(pdf))
        assert len(result) == 1
        assert result[0].endswith("test.pdf")

    def test_single_txt_file(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("text content")
        result = get_document_files(str(txt))
        assert len(result) == 1

    def test_unsupported_extension_skipped(self, tmp_path):
        doc = tmp_path / "test.docx"
        doc.write_text("word doc")
        result = get_document_files(str(doc))
        assert len(result) == 0

    def test_directory_flat(self, tmp_path):
        (tmp_path / "a.pdf").write_text("pdf")
        (tmp_path / "b.txt").write_text("txt")
        (tmp_path / "c.docx").write_text("word")
        result = get_document_files(str(tmp_path), recursive=False)
        assert len(result) == 2

    def test_directory_recursive(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (tmp_path / "a.pdf").write_text("pdf")
        (subdir / "b.pdf").write_text("pdf")
        result = get_document_files(str(tmp_path), recursive=True)
        assert len(result) == 2

    def test_results_sorted(self, tmp_path):
        (tmp_path / "b.pdf").write_text("pdf")
        (tmp_path / "a.pdf").write_text("pdf")
        result = get_document_files(str(tmp_path))
        assert result == sorted(result)


# ── ingest_single_document ────────────────────────────────────────────────


class TestIngestSingleDocument:
    def test_success(self):
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"
        ) as mock_ingest:
            result = ingest_single_document("/path/doc.pdf", {"title": "Doc"})
            assert result["status"] == "success"
            assert result["file_path"] == "/path/doc.pdf"
            mock_ingest.assert_called_once_with(
                "/path/doc.pdf", title="Doc", authors=None, pub_year=None, debug=False
            )

    def test_exception_returns_error(self):
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", side_effect=Exception("Process failed")
        ):
            result = ingest_single_document("/path/doc.pdf", {"title": "Doc"})
            assert result["status"] == "error"
            assert "Process failed" in result["error"]

    def test_authors_list_converted_to_string(self):
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"
        ) as mock_ingest:
            ingest_single_document(
                "/path/doc.pdf", {"title": "Doc", "authors": ["A", "B"]}
            )
            mock_ingest.assert_called_once_with(
                "/path/doc.pdf", title="Doc", authors="A,B", pub_year=None, debug=False
            )

    def test_pub_year_converted_to_int(self):
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"
        ) as mock_ingest:
            ingest_single_document("/path/doc.pdf", {"title": "Doc", "pub_year": 2000})
            mock_ingest.assert_called_once_with(
                "/path/doc.pdf", title="Doc", authors=None, pub_year=2000, debug=False
            )

    def test_debug_flag(self):
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"
        ) as mock_ingest:
            ingest_single_document("/path/doc.pdf", {"title": "Doc"}, debug=True)
            mock_ingest.assert_called_once_with(
                "/path/doc.pdf", title="Doc", authors=None, pub_year=None, debug=True
            )


# ── process_documents_sequential ──────────────────────────────────────────


class TestProcessDocumentsSequential:
    def test_processes_all_files(self):
        files = ["/path/a.pdf", "/path/b.pdf"]
        metadata_dict = {}
        with patch("tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"):
            results = process_documents_sequential(files, metadata_dict, debug=False)
            assert len(results) == 2
            assert all(r["status"] == "success" for r in results)

    def test_uses_metadata_when_available(self):
        files = ["/path/doc.pdf"]
        metadata_dict = {"doc.pdf": {"title": "Custom Title", "authors": "Author X"}}
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"
        ) as mock_ingest:
            process_documents_sequential(files, metadata_dict, debug=False)
            mock_ingest.assert_called_once_with(
                "/path/doc.pdf",
                title="Custom Title",
                authors="Author X",
                pub_year=None,
                debug=False,
            )

    def test_falls_back_to_filename_metadata(self):
        files = ["/path/My Report.pdf"]
        metadata_dict = {}
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"
        ) as mock_ingest:
            process_documents_sequential(files, metadata_dict, debug=False)
            mock_ingest.assert_called_once_with(
                "/path/My Report.pdf",
                title="My Report",
                authors=None,
                pub_year=None,
                debug=False,
            )

    def test_handles_errors(self):
        files = ["/path/a.pdf"]
        with patch("tome.ingestion.batch_ingest.ingest_document", side_effect=Exception("failed")):
            results = process_documents_sequential(files, {}, debug=False)
            assert results[0]["status"] == "error"


# ── process_documents_parallel ────────────────────────────────────────────


class TestProcessDocumentsParallel:
    def test_processes_all_files(self):
        files = ["/path/a.pdf", "/path/b.pdf"]
        with patch("tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"):
            results = process_documents_parallel(files, {}, max_workers=2, debug=False)
            assert len(results) == 2

    def test_handles_future_exception(self):
        files = ["/path/a.pdf"]
        with patch(
            "tome.ingestion.batch_ingest.ingest_document", side_effect=Exception("Process crash")
        ):
            results = process_documents_parallel(files, {}, max_workers=1, debug=False)
            assert len(results) == 1
            assert results[0]["status"] == "error"


# ── CLI main ──────────────────────────────────────────────────────────────


class TestMainCli:
    def test_no_files_found(self, tmp_path, caplog):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        runner = CliRunner()
        with caplog.at_level("WARNING", logger="tome.ingestion.batch_ingest"):
            runner.invoke(main, [str(empty_dir)])
        assert "No document files found" in caplog.text

    def test_sequential_processing(self, tmp_path, caplog):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf")
        runner = CliRunner()
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_ingest"),
            patch("tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"),
        ):
            runner.invoke(main, [str(tmp_path)])
        assert "Successful: 1" in caplog.text

    def test_parallel_processing(self, tmp_path, caplog):
        (tmp_path / "a.pdf").write_text("pdf")
        (tmp_path / "b.pdf").write_text("pdf")
        runner = CliRunner()
        mock_results = [
            {"file_path": str(tmp_path / "a.pdf"), "status": "success", "output": "OK"},
            {"file_path": str(tmp_path / "b.pdf"), "status": "success", "output": "OK"},
        ]
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_ingest"),
            patch("tome.ingestion.batch_ingest.process_documents_parallel", return_value=mock_results),
        ):
            runner.invoke(main, [str(tmp_path), "--parallel", "2"])
        assert "Successful: 2" in caplog.text

    def test_with_metadata_file(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_text("fake pdf")
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps([{"filename": "doc.pdf", "title": "Custom Doc"}]))
        runner = CliRunner()
        with patch("tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"):
            result = runner.invoke(main, [str(tmp_path), "--metadata", str(meta)])
            assert result.exit_code == 0

    def test_output_file(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf")
        output = tmp_path / "results.json"
        runner = CliRunner()
        with patch("tome.ingestion.batch_ingest.ingest_document", return_value="doc-123"):
            runner.invoke(main, [str(tmp_path), "--output", str(output)])
            assert output.exists()
            data = json.loads(output.read_text())
            assert len(data) == 1

    def test_failure_exits_1(self, tmp_path, caplog):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf")
        runner = CliRunner()
        with (
            caplog.at_level("INFO", logger="tome.ingestion.batch_ingest"),
            patch("tome.ingestion.batch_ingest.ingest_document", side_effect=Exception("error")),
        ):
            result = runner.invoke(main, [str(tmp_path)])
        assert "Failed: 1" in caplog.text
        assert result.exit_code == 1

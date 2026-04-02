"""Tests for batch_ingest.py."""

import pytest
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from click.testing import CliRunner

from batch_ingest import (
    extract_metadata_from_filename, load_metadata_file, get_document_files,
    ingest_single_document, process_documents_parallel,
    process_documents_sequential, main,
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
        csv_content = "filename,title,authors\ndoc1.pdf,Doc 1,Author A\ndoc2.pdf,Doc 2,Author B\n"
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
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Ingestion complete!"
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            result = ingest_single_document("/path/doc.pdf", {"title": "Doc"})
            assert result["status"] == "success"
            assert result["file_path"] == "/path/doc.pdf"

    def test_error_return_code(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: file not found"
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            result = ingest_single_document("/path/doc.pdf", {"title": "Doc"})
            assert result["status"] == "error"

    def test_exception_returns_error(self):
        with patch("batch_ingest.subprocess.run", side_effect=Exception("Process failed")):
            result = ingest_single_document("/path/doc.pdf", {"title": "Doc"})
            assert result["status"] == "error"
            assert "Process failed" in result["error"]

    def test_authors_list_converted_to_string(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("batch_ingest.subprocess.run", return_value=mock_result) as mock_run:
            ingest_single_document("/path/doc.pdf", {"title": "Doc", "authors": ["A", "B"]})
            cmd = mock_run.call_args[0][0]
            assert "A,B" in cmd

    def test_pub_year_converted_to_string(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("batch_ingest.subprocess.run", return_value=mock_result) as mock_run:
            ingest_single_document("/path/doc.pdf", {"title": "Doc", "pub_year": 2000})
            cmd = mock_run.call_args[0][0]
            assert "2000" in cmd

    def test_debug_flag(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("batch_ingest.subprocess.run", return_value=mock_result) as mock_run:
            ingest_single_document("/path/doc.pdf", {"title": "Doc"}, debug=True)
            cmd = mock_run.call_args[0][0]
            assert "--debug" in cmd


# ── process_documents_sequential ──────────────────────────────────────────

class TestProcessDocumentsSequential:
    def test_processes_all_files(self):
        files = ["/path/a.pdf", "/path/b.pdf"]
        metadata_dict = {}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            results = process_documents_sequential(files, metadata_dict, debug=False)
            assert len(results) == 2
            assert all(r["status"] == "success" for r in results)

    def test_uses_metadata_when_available(self):
        files = ["/path/doc.pdf"]
        metadata_dict = {"doc.pdf": {"title": "Custom Title", "authors": "Author X"}}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("batch_ingest.subprocess.run", return_value=mock_result) as mock_run:
            process_documents_sequential(files, metadata_dict, debug=False)
            cmd = mock_run.call_args[0][0]
            assert "Custom Title" in cmd

    def test_falls_back_to_filename_metadata(self):
        files = ["/path/My Report.pdf"]
        metadata_dict = {}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("batch_ingest.subprocess.run", return_value=mock_result) as mock_run:
            process_documents_sequential(files, metadata_dict, debug=False)
            cmd = mock_run.call_args[0][0]
            assert "My Report" in cmd

    def test_handles_errors(self):
        files = ["/path/a.pdf"]
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "failed"
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            results = process_documents_sequential(files, {}, debug=False)
            assert results[0]["status"] == "error"


# ── process_documents_parallel ────────────────────────────────────────────

class TestProcessDocumentsParallel:
    def test_processes_all_files(self):
        files = ["/path/a.pdf", "/path/b.pdf"]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            results = process_documents_parallel(files, {}, max_workers=2, debug=False)
            assert len(results) == 2

    def test_handles_future_exception(self):
        files = ["/path/a.pdf"]
        with patch("batch_ingest.subprocess.run", side_effect=Exception("Process crash")):
            results = process_documents_parallel(files, {}, max_workers=1, debug=False)
            assert len(results) == 1
            assert results[0]["status"] == "error"


# ── CLI main ──────────────────────────────────────────────────────────────

class TestMainCli:
    def test_no_files_found(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(main, [str(empty_dir)])
        assert "No document files found" in result.output

    def test_sequential_processing(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        runner = CliRunner()
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            result = runner.invoke(main, [str(tmp_path)])
            assert "Successful: 1" in result.output

    def test_parallel_processing(self, tmp_path):
        (tmp_path / "a.pdf").write_text("pdf")
        (tmp_path / "b.pdf").write_text("pdf")
        runner = CliRunner()
        mock_results = [
            {"file_path": str(tmp_path / "a.pdf"), "status": "success", "output": "OK"},
            {"file_path": str(tmp_path / "b.pdf"), "status": "success", "output": "OK"},
        ]
        with patch("batch_ingest.process_documents_parallel", return_value=mock_results):
            result = runner.invoke(main, [str(tmp_path), "--parallel", "2"])
            assert "Successful: 2" in result.output

    def test_with_metadata_file(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_text("fake pdf")
        meta = tmp_path / "meta.json"
        meta.write_text(json.dumps([{"filename": "doc.pdf", "title": "Custom Doc"}]))
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        runner = CliRunner()
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            result = runner.invoke(main, [str(tmp_path), "--metadata", str(meta)])
            assert result.exit_code == 0

    def test_output_file(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf")
        output = tmp_path / "results.json"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        runner = CliRunner()
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            result = runner.invoke(main, [str(tmp_path), "--output", str(output)])
            assert output.exists()
            data = json.loads(output.read_text())
            assert len(data) == 1

    def test_failure_exits_1(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("fake pdf")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        runner = CliRunner()
        with patch("batch_ingest.subprocess.run", return_value=mock_result):
            result = runner.invoke(main, [str(tmp_path)])
            assert "Failed: 1" in result.output
            assert result.exit_code == 1

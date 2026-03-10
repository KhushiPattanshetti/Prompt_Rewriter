"""
tests/test_validator.py
Unit tests for the deterministic validation logic.
tests/test_api.py integrated here for single-file execution.

Run with:  python -m pytest tests/ -v
       or: python -m pytest tests/test_all.py -v
"""

import json
import os
import sys
import tempfile
import pytest

# ── Ensure backend root is on path ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.validator import validate_request, ValidationResult
from app.storage import store_instruction, get_all_instructions


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — validator.py
# ═══════════════════════════════════════════════════════════════════════════

class TestValidatorPresenceChecks:
    def test_missing_instruction_returns_not_ok(self):
        r = validate_request(None, "note.txt", "Patient has fever and cough diagnosis needed")
        assert not r.ok
        assert "instruction" in r.error.lower()

    def test_empty_instruction_returns_not_ok(self):
        r = validate_request("   ", "note.txt", "Patient has fever and cough diagnosis needed")
        assert not r.ok
        assert "instruction" in r.error.lower()

    def test_missing_filename_returns_not_ok(self):
        r = validate_request("Extract ICD-10 codes", None, "Patient presents with pneumonia diagnosis")
        assert not r.ok
        assert "document" in r.error.lower()

    def test_missing_file_content_returns_not_ok(self):
        r = validate_request("Extract ICD-10 codes", "note.txt", None)
        assert not r.ok
        assert "document" in r.error.lower()

    def test_empty_file_content_returns_not_ok(self):
        r = validate_request("Extract ICD-10 codes", "note.txt", "")
        assert not r.ok


class TestValidatorFileFormat:
    VALID_NOTE = "Patient presents with acute pneumonia, requiring ICD-10 diagnosis coding."

    def test_txt_extension_allowed(self):
        r = validate_request("Extract ICD-10 codes from note", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_json_extension_allowed(self):
        r = validate_request("Extract ICD-10 codes from note", "note.json", self.VALID_NOTE)
        assert r.ok

    def test_csv_extension_allowed(self):
        r = validate_request("Extract ICD-10 codes from note", "note.csv", self.VALID_NOTE)
        assert r.ok

    def test_pdf_extension_rejected(self):
        r = validate_request("Extract ICD-10 codes from note", "note.pdf", self.VALID_NOTE)
        assert not r.ok
        assert "pdf" in r.error.lower() or "format" in r.error.lower()

    def test_docx_extension_rejected(self):
        r = validate_request("Extract ICD-10 codes from note", "note.docx", self.VALID_NOTE)
        assert not r.ok

    def test_jpg_extension_rejected(self):
        r = validate_request("Extract ICD-10 codes from note", "note.jpg", self.VALID_NOTE)
        assert not r.ok

    def test_png_extension_rejected(self):
        r = validate_request("Extract ICD-10 codes from note", "note.png", self.VALID_NOTE)
        assert not r.ok


class TestValidatorNoteLength:
    def test_note_exactly_14_chars_rejected(self):
        r = validate_request("Extract ICD-10 codes", "note.txt", "Short note lol")  # 14 chars
        assert not r.ok
        assert "short" in r.error.lower() or "minimum" in r.error.lower()

    def test_note_exactly_15_chars_accepted(self):
        # 15 chars exactly — boundary value
        r = validate_request("Extract ICD-10 codes from note", "note.txt", "A" * 15)
        # May still fail keyword/verb checks — that's fine; just not length
        assert r.error == "" or "short" not in r.error.lower()

    def test_whitespace_only_note_rejected(self):
        r = validate_request("Extract ICD-10 codes", "note.txt", "     \n\t  ")
        assert not r.ok


class TestValidatorKeywordMatch:
    VALID_NOTE = "Patient admitted with respiratory infection requiring clinical evaluation."

    def test_icd_keyword_triggers_match(self):
        r = validate_request("Extract ICD-10 codes from the note", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_diagnosis_keyword_triggers_match(self):
        r = validate_request("Generate diagnosis codes for patient", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_medical_code_keyword_triggers_match(self):
        r = validate_request("Find medical code for patient", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_no_icd_keyword_returns_not_ok(self):
        r = validate_request("Please summarise this document", "note.txt", self.VALID_NOTE)
        assert not r.ok
        assert "icd" in r.error.lower() or "instruction" in r.error.lower()


class TestValidatorActionVerbs:
    VALID_NOTE = "Patient presents with type 2 diabetes and requires diagnosis codes."

    def test_extract_verb_accepted(self):
        r = validate_request("Extract ICD-10 codes from note", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_generate_verb_accepted(self):
        r = validate_request("Generate ICD-10 diagnosis codes", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_map_verb_accepted(self):
        r = validate_request("Map diagnoses to ICD-10 codes", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_identify_verb_accepted(self):
        r = validate_request("Identify ICD-10 codes in this document", "note.txt", self.VALID_NOTE)
        assert r.ok

    def test_all_valid_verbs_accepted(self):
        verbs = ["generate", "extract", "list", "find", "identify",
                 "assign", "map", "convert", "return", "provide"]
        for verb in verbs:
            r = validate_request(f"{verb} ICD-10 codes from note", "note.txt", self.VALID_NOTE)
            assert r.ok, f"Verb '{verb}' should be accepted but was rejected: {r.error}"

    def test_no_action_verb_returns_not_ok(self):
        r = validate_request("ICD-10 codes needed for this patient", "note.txt", self.VALID_NOTE)
        assert not r.ok


class TestValidatorInvalidIntents:
    VALID_NOTE = "Patient admitted with fever, cough, and elevated temperature."

    def test_explain_icd_rejected(self):
        r = validate_request("Explain ICD-10 coding system to me", "note.txt", self.VALID_NOTE)
        assert not r.ok

    def test_what_is_icd_rejected(self):
        r = validate_request("What is ICD-10?", "note.txt", self.VALID_NOTE)
        assert not r.ok

    def test_write_story_rejected(self):
        r = validate_request("Write a story about this patient's journey", "note.txt", self.VALID_NOTE)
        assert not r.ok

    def test_translate_rejected(self):
        r = validate_request("Translate this diagnosis to Spanish", "note.txt", self.VALID_NOTE)
        assert not r.ok

    def test_suggest_treatment_rejected(self):
        r = validate_request("Suggest treatment for this patient", "note.txt", self.VALID_NOTE)
        assert not r.ok

    def test_recommend_medication_rejected(self):
        r = validate_request("Recommend medication for the patient", "note.txt", self.VALID_NOTE)
        assert not r.ok


class TestValidatorValidExamples:
    """End-to-end valid request examples from the spec."""

    def test_discharge_summary_example(self):
        r = validate_request(
            "Extract ICD-10 codes from the following discharge summary.",
            "discharge.txt",
            "72 year old female presented with worsening shortness of breath and fatigue. "
            "Imaging confirmed pneumonia and the patient was admitted for treatment.",
        )
        assert r.ok

    def test_patient_records_example(self):
        r = validate_request(
            "Generate diagnosis codes from patient records",
            "records.csv",
            "Patient: John Doe, 58M. Chief complaint: chest pain. "
            "Assessment: Unstable angina, hypertension. Plan: admit for monitoring.",
        )
        assert r.ok

    def test_mapping_example(self):
        r = validate_request(
            "Map diagnoses to ICD-10 codes",
            "notes.json",
            "Patient presents with type 2 diabetes mellitus without complications "
            "and essential hypertension. No acute distress.",
        )
        assert r.ok


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS — storage.py
# ═══════════════════════════════════════════════════════════════════════════

class TestStorage:
    def test_store_instruction_returns_record(self, tmp_path, monkeypatch):
        # Patch paths to use tmp_path
        import app.storage as storage_mod
        dataset_path = str(tmp_path / "instruction_dataset.json")
        instructions_dir = str(tmp_path / "user_instructions")
        monkeypatch.setattr(storage_mod, "_DATASET_PATH", dataset_path)
        monkeypatch.setattr(storage_mod, "_USER_INSTRUCTIONS_DIR", instructions_dir)

        record = storage_mod.store_instruction("Extract ICD-10 codes from clinical note")
        assert "instruction_id" in record
        assert record["instruction_id"].startswith("ins_")
        assert "timestamp" in record
        assert record["user_instruction"] == "Extract ICD-10 codes from clinical note"

    def test_store_instruction_persists_to_json(self, tmp_path, monkeypatch):
        import app.storage as storage_mod
        dataset_path = str(tmp_path / "instruction_dataset.json")
        instructions_dir = str(tmp_path / "user_instructions")
        monkeypatch.setattr(storage_mod, "_DATASET_PATH", dataset_path)
        monkeypatch.setattr(storage_mod, "_USER_INSTRUCTIONS_DIR", instructions_dir)

        storage_mod.store_instruction("Generate diagnosis codes")
        storage_mod.store_instruction("Map ICD-10 codes")

        records = storage_mod.get_all_instructions()
        assert len(records) == 2

    def test_store_instruction_writes_flat_file(self, tmp_path, monkeypatch):
        import app.storage as storage_mod
        dataset_path = str(tmp_path / "instruction_dataset.json")
        instructions_dir = str(tmp_path / "user_instructions")
        monkeypatch.setattr(storage_mod, "_DATASET_PATH", dataset_path)
        monkeypatch.setattr(storage_mod, "_USER_INSTRUCTIONS_DIR", instructions_dir)

        record = storage_mod.store_instruction("Find ICD codes")
        flat_file = os.path.join(instructions_dir, f"{record['instruction_id']}.json")
        assert os.path.exists(flat_file)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — Flask API
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with isolated storage."""
    import app.storage as storage_mod
    dataset_path = str(tmp_path / "instruction_dataset.json")
    instructions_dir = str(tmp_path / "user_instructions")
    monkeypatch.setattr(storage_mod, "_DATASET_PATH", dataset_path)
    monkeypatch.setattr(storage_mod, "_USER_INSTRUCTIONS_DIR", instructions_dir)

    from main import create_app
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


VALID_NOTE_TEXT = (
    "72 year old female presented with worsening shortness of breath and fatigue. "
    "Imaging confirmed pneumonia and the patient was admitted for treatment."
)


class TestAPIHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"


class TestAPIValidateJSON:
    def test_valid_request_returns_ok(self, client):
        resp = client.post(
            "/api/validate",
            json={
                "user_instruction": "Extract ICD-10 codes from discharge summary",
                "filename": "note.txt",
                "clinical_note": VALID_NOTE_TEXT,
            },
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "OK"
        assert "instruction_id" in data

    def test_valid_request_stores_instruction(self, client):
        client.post(
            "/api/validate",
            json={
                "user_instruction": "Generate ICD-10 codes for patient record",
                "filename": "note.txt",
                "clinical_note": VALID_NOTE_TEXT,
            },
            content_type="application/json",
        )
        resp = client.get("/api/instructions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 1

    def test_missing_instruction_returns_422(self, client):
        resp = client.post(
            "/api/validate",
            json={"filename": "note.txt", "clinical_note": VALID_NOTE_TEXT},
            content_type="application/json",
        )
        assert resp.status_code == 422
        assert resp.get_json()["status"] == "NOT_OK"

    def test_missing_note_returns_422(self, client):
        resp = client.post(
            "/api/validate",
            json={"user_instruction": "Extract ICD-10 codes", "filename": "note.txt"},
            content_type="application/json",
        )
        assert resp.status_code == 422
        assert resp.get_json()["status"] == "NOT_OK"

    def test_unsupported_format_returns_422(self, client):
        resp = client.post(
            "/api/validate",
            json={
                "user_instruction": "Extract ICD-10 codes",
                "filename": "note.pdf",
                "clinical_note": VALID_NOTE_TEXT,
            },
            content_type="application/json",
        )
        assert resp.status_code == 422
        data = resp.get_json()
        assert data["status"] == "NOT_OK"
        assert "format" in data["error"].lower()

    def test_out_of_scope_instruction_returns_422(self, client):
        resp = client.post(
            "/api/validate",
            json={
                "user_instruction": "Write a blog post about this patient",
                "filename": "note.txt",
                "clinical_note": VALID_NOTE_TEXT,
            },
            content_type="application/json",
        )
        assert resp.status_code == 422

    def test_educational_query_returns_422(self, client):
        resp = client.post(
            "/api/validate",
            json={
                "user_instruction": "Explain ICD-10 to me",
                "filename": "note.txt",
                "clinical_note": VALID_NOTE_TEXT,
            },
            content_type="application/json",
        )
        assert resp.status_code == 422


class TestAPIValidateMultipart:
    def test_multipart_upload_valid_txt(self, client):
        data = {
            "user_instruction": "Extract ICD-10 codes from discharge summary",
            "clinical_document": (
                io.BytesIO(VALID_NOTE_TEXT.encode()),
                "discharge.txt",
            ),
        }
        resp = client.post(
            "/api/validate",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "OK"

    def test_multipart_upload_invalid_format(self, client):
        data = {
            "user_instruction": "Extract ICD-10 codes",
            "clinical_document": (
                io.BytesIO(VALID_NOTE_TEXT.encode()),
                "note.pdf",
            ),
        }
        resp = client.post(
            "/api/validate",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 422


class TestAPIInstructions:
    def test_instructions_endpoint_returns_list(self, client):
        resp = client.get("/api/instructions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "instructions" in data
        assert "count" in data

    def test_multiple_valid_requests_accumulate(self, client):
        for i in range(3):
            client.post(
                "/api/validate",
                json={
                    "user_instruction": f"Extract ICD-10 codes from note {i}",
                    "filename": "note.txt",
                    "clinical_note": VALID_NOTE_TEXT,
                },
                content_type="application/json",
            )
        resp = client.get("/api/instructions")
        assert resp.get_json()["count"] == 3


import io  # ensure io available at module level for multipart tests

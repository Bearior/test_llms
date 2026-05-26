"""Audit record must be PII-free."""
import pytest

from thaidoc import audit


def test_audit_record_is_pii_free():
    rec = audit.build_audit_record(
        predicted_class="หนังสือเดินทาง (Passport)",
        confidence=0.97, routing_decision="AUTO_ACCEPT")
    audit.assert_no_pii(rec)  # must not raise
    assert set(rec.keys()) == audit.ALLOWED_FIELDS


def test_injected_pii_is_rejected():
    rec = audit.build_audit_record("x", 0.5, "HUMAN_REVIEW")
    rec["customer_name"] = "สมชาย"          # leak
    with pytest.raises(AssertionError):
        audit.assert_no_pii(rec)


def test_raw_text_field_rejected():
    rec = audit.build_audit_record("x", 0.5, "HUMAN_REVIEW")
    rec["ocr_text"] = "..."                  # leak
    with pytest.raises(AssertionError):
        audit.assert_no_pii(rec)

"""End-to-end pipeline behavior on synthetic data.

Asserts the two narrative-critical behaviors:
  * a clean sample with a readable subtype marker is classified and accepted;
  * a blinded adversarial subtype sample is routed to HUMAN_REVIEW (graceful
    failure), NOT confidently misclassified.
"""
from thaidoc import confidence
from thaidoc.synth import image_path


def _first(rows, **kw):
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in kw.items()):
            return r
    return None


def test_clean_visa_subtype_is_accepted(pipeline_and_rows):
    pipe, rows = pipeline_and_rows
    r = _first(rows, physical_type='Non-Immigrant VISA "B"',
               is_adversarial=False, split="test")
    if r is None:  # fall back to any clean visa-B sample
        r = _first(rows, physical_type='Non-Immigrant VISA "B"', is_adversarial=False)
    assert r is not None
    c = pipe.classify(image_path(r["filename"]), record=r)
    assert c.routing != confidence.ROUTE_HUMAN_REVIEW
    assert c.physical_type == 'Non-Immigrant VISA "B"'


def test_adversarial_visa_subtype_routes_to_human_review(pipeline_and_rows):
    # The crux case: VISA subtypes differ ONLY by the quoted code. When that
    # code is unreadable (adversarial blinding), they are indistinguishable and
    # MUST route to human review rather than be confidently misclassified.
    from thaidoc import labels
    pipe, rows = pipeline_and_rows
    adv = [r for r in rows
           if r["is_adversarial"] and r["family"] == labels.FAMILY_VISA]
    assert adv, "expected at least one adversarial visa subtype sample"
    for r in adv:
        c = pipe.classify(image_path(r["filename"]), record=r)
        assert c.routing == confidence.ROUTE_HUMAN_REVIEW


def test_audit_record_emitted_pii_free(pipeline_and_rows):
    from thaidoc import audit
    pipe, rows = pipeline_and_rows
    c = pipe.classify(image_path(rows[0]["filename"]), record=rows[0])
    audit.assert_no_pii(c.audit_record)

"""Cloud-LLM pipeline: routing, snapping, prompt/schema, audit (mock provider)."""
import pytest

from thaidoc.labels import UNKNOWN_TYPE
from thaidoc.synth import image_path
from thaidoc_llm import pipeline, prompt
from thaidoc_llm.providers.factory import get_provider


def _pipe(rows):
    return pipeline.LLMPipeline(get_provider("mock", manifest_rows=rows))


def test_routing_thresholds():
    assert pipeline.route(0.99) == pipeline.ROUTE_AUTO_ACCEPT
    assert pipeline.route(0.72) == pipeline.ROUTE_SPOT_CHECK
    assert pipeline.route(0.40) == pipeline.ROUTE_HUMAN_REVIEW


def test_schema_enum_covers_all_canonical_plus_unknown():
    labels_in_enum = prompt.output_schema()["properties"]["physical_type"]["enum"]
    # 94 canonical + UNKNOWN sentinel.
    assert len(labels_in_enum) == 95
    assert UNKNOWN_TYPE in labels_in_enum


def test_system_prompt_lists_families_and_is_cacheable_size():
    sp = prompt.system_prompt()
    assert "VISA_STAMP" in sp and "PASSPORT_BOOKLET" in sp
    # Reasonably large (label catalogue) so prompt caching is worthwhile.
    assert len(sp) > 1500


def test_snap_maps_offlist_to_canonical_or_unknown():
    assert pipeline._snap('Non-Immigrant VISA "B"') == 'Non-Immigrant VISA "B"'
    assert pipeline._snap("total gibberish xyz") == UNKNOWN_TYPE


def test_clean_sample_accepted(manifest_rows):
    pipe = _pipe(manifest_rows)
    r = next(r for r in manifest_rows
             if r["physical_type"] == 'Non-Immigrant VISA "B"' and not r["is_adversarial"])
    c = pipe.classify(image_path(r["filename"]), record=r)
    assert c.routing != pipeline.ROUTE_HUMAN_REVIEW
    assert c.physical_type == 'Non-Immigrant VISA "B"'


def test_adversarial_visa_routes_to_human_review(manifest_rows):
    from thaidoc import labels
    pipe = _pipe(manifest_rows)
    adv = [r for r in manifest_rows
           if r["is_adversarial"] and r["family"] == labels.FAMILY_VISA]
    assert adv
    for r in adv:
        c = pipe.classify(image_path(r["filename"]), record=r)
        assert c.routing == pipeline.ROUTE_HUMAN_REVIEW
        assert c.physical_type == UNKNOWN_TYPE


def test_audit_record_pii_free(manifest_rows):
    from thaidoc import audit
    pipe = _pipe(manifest_rows)
    c = pipe.classify(image_path(manifest_rows[0]["filename"]), record=manifest_rows[0])
    audit.assert_no_pii(c.audit_record)


def test_gemini_default_model_and_clean_unavailability():
    import importlib.util
    from thaidoc_llm import config
    from thaidoc_llm.providers.base import ProviderUnavailable
    # Default real provider is the small, free Gemini model.
    assert config.DEFAULT_PROVIDER in ("gemini", "anthropic", "mock")
    assert config.default_model_for("gemini") == config.GEMINI_MODEL
    # If the SDK isn't installed, constructing it must raise cleanly (not crash).
    if importlib.util.find_spec("google") is None:
        with pytest.raises(ProviderUnavailable):
            get_provider("gemini")


def test_ollama_default_model_and_clean_unavailability():
    """Local on-prem provider: defaults to a vision model and degrades cleanly
    to ProviderUnavailable when no Ollama server is reachable."""
    from thaidoc_llm import config
    from thaidoc_llm.providers.base import ProviderUnavailable
    from thaidoc_llm.providers.factory import _PROVIDERS
    assert config.default_model_for("ollama") == config.OLLAMA_MODEL
    assert "ollama" in _PROVIDERS
    # No local Ollama server running in CI -> must raise cleanly, never crash.
    with pytest.raises(ProviderUnavailable):
        get_provider("ollama")


def test_transformers_provider_wiring_and_json_extractor():
    """In-process local provider is registered and its JSON extractor is
    tolerant of fenced / noisy model output. We don't load the heavy model."""
    from thaidoc_llm import config
    from thaidoc_llm.providers.base import ProviderUnavailable
    from thaidoc_llm.providers.factory import _PROVIDERS
    from thaidoc_llm.providers.transformers_provider import _extract_json
    assert "transformers" in _PROVIDERS
    assert config.default_model_for("transformers") == config.TRANSFORMERS_MODEL
    # Clean JSON, ```json fenced, and JSON embedded in prose all parse.
    assert _extract_json('{"physical_type": "X", "confidence": 0.9}')["physical_type"] == "X"
    assert _extract_json('```json\n{"physical_type": "Y", "confidence": 0.5}\n```')["confidence"] == 0.5
    assert _extract_json('Sure: {"physical_type": "Z", "confidence": 0.1} done')["physical_type"] == "Z"
    # Non-JSON output must fail cleanly (caller -> human review / mock).
    with pytest.raises(ProviderUnavailable):
        _extract_json("I cannot determine the document type.")


def test_transformers_full_catalog_and_pick():
    """Evidence-first single catalog numbers every label and maps ids back."""
    import thaidoc_llm.providers.transformers_provider as tp
    from thaidoc.labels import UNKNOWN_TYPE, build_catalog
    id_map, sys_prompt = tp._full_catalog()
    assert id_map[0] == UNKNOWN_TYPE
    # Every canonical label is present exactly once in the numbered catalogue.
    listed = {v for k, v in id_map.items() if k}
    assert {dt.physical_type for dt in build_catalog()} == listed
    # No family gate, but families remain as scan-only section headers.
    assert "##" in sys_prompt and "KEYWORDS" in sys_prompt.upper()
    # _pick maps an id via the mapping; bad/absent ids -> None.
    m = {0: UNKNOWN_TYPE, 1: "A", 2: "B"}
    assert tp.TransformersProvider._pick({"id": 2}, m) == "B"
    assert tp.TransformersProvider._pick({"id": 1}, m) == "A"
    assert tp.TransformersProvider._pick({"confidence": 0.9}, m) is None
    assert tp.TransformersProvider._pick({"id": "x"}, m) is None


def test_ood_unknown_routes_to_review(manifest_rows):
    # Provider with no index + no record -> UNKNOWN low-confidence -> human review.
    pipe = pipeline.LLMPipeline(get_provider("mock", manifest_rows=[]))
    c = pipe.classify(image_path(manifest_rows[0]["filename"]), record=None)
    assert c.routing == pipeline.ROUTE_HUMAN_REVIEW
    assert c.physical_type == UNKNOWN_TYPE

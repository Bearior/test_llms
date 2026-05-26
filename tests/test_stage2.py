"""Stage-2 refinement & the abstention / unknown route."""
from thaidoc import labels, stage2
from thaidoc.readers.base import ReaderResult


def test_readable_marker_resolves_subtype_with_positive_margin():
    from thaidoc import confidence
    from thaidoc.config import MED_THRESHOLD
    res = stage2.refine(labels.FAMILY_VISA,
                        ReaderResult(text='Non-Immigrant VISA "B" CODE B'))
    assert res.physical_type == 'Non-Immigrant VISA "B"'
    assert res.margin > 0.1  # distinguished enough to clear the routing gate
    # And that margin must translate into an acceptable confidence.
    conf = confidence.combine(0.9, res.score, res.margin)
    assert conf >= MED_THRESHOLD


def test_blinded_marker_collapses_margin():
    # Adversarial: marker stripped -> two visa subtypes look near-identical.
    res = stage2.refine(labels.FAMILY_VISA,
                        ReaderResult(text='Non-Immigrant VISA ""'))
    assert res.margin < 0.1  # ambiguous -> low confidence downstream


def test_empty_read_yields_no_signal():
    res = stage2.refine(labels.FAMILY_VISA, ReaderResult(text=""))
    assert res.score < 0.5


def test_unknown_family_returns_none():
    res = stage2.refine("NOT_A_FAMILY", ReaderResult(text="anything"))
    assert res.physical_type is None


def test_type_hypotheses_path_snaps_to_canonical_label():
    # VLM-as-classifier path: a type hypothesis is snapped to the nearest
    # allow-listed canonical label within the family.
    res = stage2.refine(
        labels.FAMILY_VISA,
        ReaderResult(type_hypotheses=[('Non-Immigrant VISA "F"', 1.0)]))
    assert res.physical_type == 'Non-Immigrant VISA "F"'
    assert res.score > 0.9


def test_empty_read_abstains_explicitly():
    # Below the score floor Stage-2 returns no type (explicit abstention).
    res = stage2.refine(labels.FAMILY_VISA, ReaderResult(text=""))
    assert res.physical_type is None

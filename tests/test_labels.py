"""Label catalog & de-duplication."""
from thaidoc import labels


def test_business_vs_canonical_counts():
    # 116 business rows collapse to 94 canonical physical types (17 duplicates).
    assert labels.business_label_count() == 116
    assert labels.canonical_count() == 94


def test_alias_map_covers_all_canonical():
    amap = labels.alias_map()
    cat = {dt.physical_type for dt in labels.build_catalog()}
    assert set(amap.values()) == cat


def test_families_assigned():
    cat = labels.build_catalog()
    fams = {dt.family for dt in cat}
    assert labels.FAMILY_VISA in fams
    assert labels.FAMILY_PASSPORT in fams
    assert labels.FAMILY_ID_CARD in fams
    # No type should be left without a family string.
    assert all(dt.family for dt in cat)


def test_visa_subtype_markers_extracted():
    visas = labels.types_by_family(labels.FAMILY_VISA)
    markers = {v.subtype_marker for v in visas if v.subtype_marker}
    # The near-identical-subtype codes the project hinges on.
    assert {"B", "F"}.issubset(markers)

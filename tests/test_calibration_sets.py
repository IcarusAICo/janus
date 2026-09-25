

def test_temperature_for_uses_the_kind_fallback_only_without_a_family():
    from janus.calibration_sets import temperature_for
    cal = {"temperature": 1.1, "by_kind": {"noul": 0.8}, "by_family": {"judge": {"temperature": 1.5, "by_cardinality": {}}}}
    assert temperature_for(cal, None, 2, "noul") == 0.8
    assert temperature_for(cal, None, 4, "choice") == 1.1
    assert temperature_for(cal, "judge", 2, "noul") == 1.5
    assert temperature_for({"temperature": 1.1}, None, 2, "noul") == 1.1

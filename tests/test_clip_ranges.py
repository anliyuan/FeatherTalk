import json

from dataset_mouth_roi import load_clip_ranges


def test_load_clip_ranges_defaults_to_one_continuous_clip(tmp_path) -> None:
    assert load_clip_ranges(str(tmp_path), 12) == [(0, 12)]


def test_load_clip_ranges_validates_complete_coverage(tmp_path) -> None:
    path = tmp_path / "clip_ranges.json"
    path.write_text(
        json.dumps({"ranges": [{"start": 0, "end": 4}, {"start": 4, "end": 9}]}),
        encoding="utf-8",
    )
    assert load_clip_ranges(str(tmp_path), 9) == [(0, 4), (4, 9)]

    path.write_text(
        json.dumps({"ranges": [{"start": 0, "end": 4}, {"start": 5, "end": 9}]}),
        encoding="utf-8",
    )
    try:
        load_clip_ranges(str(tmp_path), 9)
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("a discontinuous clip range was accepted")

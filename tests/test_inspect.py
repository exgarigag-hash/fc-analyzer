import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspect_api import field_summary, mask, sections, shot_type_table  # noqa: E402


def test_mask_hides_ids_keeps_structure():
    d = {"ouid": "abc", "matchInfo": [{"nickname": "x", "shoot": {"goalTotal": 2}}]}
    m = mask(d)
    assert m["ouid"] == "***" and m["matchInfo"][0]["nickname"] == "***"
    assert m["matchInfo"][0]["shoot"]["goalTotal"] == 2


def test_field_summary_counts_and_types():
    s = field_summary([{"a": 1, "b": "x"}, {"a": 2.5}])
    assert s["a"][0] == 2 and s["a"][1] == {"int", "float"} and s["b"][2] == "x"


def test_sections_and_shot_types():
    side = {"ouid": "o", "matchDetail": {"foul": 1}, "shoot": {}, "pass": {}, "defence": {},
            "shootDetail": [{"type": 9, "result": 3, "x": .89, "y": .5, "inPenalty": True},
                            {"type": 1, "result": 2, "x": .7, "y": .4}],
            "player": [{"spId": 1, "status": {"goal": 0}}]}
    sec = sections([{"matchId": "m", "matchInfo": [side, side]}])
    assert len(sec["shootDetail[]"]) == 4 and len(sec["player[].status"]) == 2
    rows = {r["type"]: r for r in shot_type_table(sec["shootDetail[]"])}
    assert rows[9]["goal%"] == 100 and rows[1]["goal%"] == 0 and rows[9]["box%"] == 100

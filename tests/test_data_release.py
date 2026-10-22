from types import SimpleNamespace

import yaml

from byosm.hub.data_release import front


def test_front_matter_is_valid_yaml_with_a_config_per_extra_test_set():
    base = SimpleNamespace(languages=["en", "it"], title="Turn", name="turn")
    f = SimpleNamespace(license="cc-by-nc-4.0", base=base)
    found = {"train": [], "validation": [], "test": [], "hard-cases": []}
    meta = yaml.safe_load(front(f, found, 12_000))
    assert meta["language"] == ["en", "it"]
    assert meta["size_categories"] == ["10K<n<100K"]
    assert [c["config_name"] for c in meta["configs"]] == ["default", "hard-cases"]
    assert meta["configs"][1]["data_files"] == [{"split": "test", "path": "test-sets/hard-cases.jsonl"}]

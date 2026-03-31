import json

from utils.identity_dataset_v1 import (
    SHOT_TEMPLATE,
    assign_person_splits,
    build_manifest_rows,
    generate_person_ids,
    validate_manifest_rows,
)


def test_shot_template_has_20_unique_shots():
    assert len(SHOT_TEMPLATE) == 20
    shot_ids = [row.shot_id for row in SHOT_TEMPLATE]
    assert len(set(shot_ids)) == 20


def test_generate_manifest_default_counts():
    person_ids = generate_person_ids(num_people=75, prefix="person")
    split_by_person = assign_person_splits(person_ids, train_count=60, val_count=8, test_count=7)
    rows = build_manifest_rows(person_ids, split_by_person)

    assert len(rows) == 1500

    # Each person has exactly 20 shots.
    per_person = {}
    for row in rows:
        per_person[row["person_id"]] = per_person.get(row["person_id"], 0) + 1
    assert set(per_person.values()) == {20}


def test_split_counts_match_spec():
    person_ids = generate_person_ids(num_people=75, prefix="person")
    split_by_person = assign_person_splits(person_ids, train_count=60, val_count=8, test_count=7)
    assert sum(1 for v in split_by_person.values() if v == "train") == 60
    assert sum(1 for v in split_by_person.values() if v == "val") == 8
    assert sum(1 for v in split_by_person.values() if v == "test") == 7


def test_manifest_validation_passes_for_generated_template():
    person_ids = generate_person_ids(num_people=75, prefix="person")
    split_by_person = assign_person_splits(person_ids, train_count=60, val_count=8, test_count=7)
    rows = build_manifest_rows(person_ids, split_by_person)
    errors = validate_manifest_rows(rows)
    assert errors == []


def test_manifest_validation_catches_missing_shot():
    person_ids = generate_person_ids(num_people=1, prefix="person")
    split_by_person = assign_person_splits(person_ids, train_count=1, val_count=0, test_count=0)
    rows = build_manifest_rows(person_ids, split_by_person)
    # Drop one row to break 20-shot requirement.
    rows = rows[:-1]
    errors = validate_manifest_rows(rows)
    assert any("expected 20 rows" in err for err in errors)


def test_manifest_validation_catches_invalid_quality_flags():
    person_ids = generate_person_ids(num_people=1, prefix="person")
    split_by_person = assign_person_splits(person_ids, train_count=1, val_count=0, test_count=0)
    rows = build_manifest_rows(person_ids, split_by_person)
    rows[0]["quality_flags"] = json.dumps({"bad": "shape"})
    errors = validate_manifest_rows(rows)
    assert any("quality_flags must decode to a JSON array" in err for err in errors)


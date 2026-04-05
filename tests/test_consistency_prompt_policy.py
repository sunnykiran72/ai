from utils.consistency_prompt_policy import (
    build_consistency_prompt_plan,
    infer_source_worn_types,
)


def test_infer_source_worn_types_from_text():
    text = "full-body user wearing a long dress with a light jacket"
    kinds = infer_source_worn_types(text)
    assert "dress" in kinds
    assert "outer" in kinds


def test_single_top_over_dress_policy():
    plan = build_consistency_prompt_plan(
        target_types=["top"],
        source_worn_types=["dress"],
        garment_descriptions=["cropped black knit top"],
        board_mode="single",
    )
    assert plan.policy_id == "single_top_over_dress_v1"
    assert "Replace the entire upper-garment structure" in plan.prompt
    assert "lower dress panel" in plan.prompt
    assert "Do not keep the original upper-dress neckline" in plan.prompt


def test_single_bottom_over_dress_policy():
    plan = build_consistency_prompt_plan(
        target_types=["bottom"],
        source_worn_types=["dress"],
        garment_descriptions=["straight-leg denim pants"],
        board_mode="single",
    )
    assert plan.policy_id == "single_bottom_over_dress_v1"
    assert "lower clothing region" in plan.prompt
    assert "upper bodice" in plan.prompt


def test_single_outer_policy():
    plan = build_consistency_prompt_plan(
        target_types=["outer"],
        source_worn_types=["top", "bottom"],
        garment_descriptions=["beige blazer"],
        board_mode="single",
    )
    assert plan.policy_id == "single_outer_v1"
    assert "outerwear from Image 2" in plan.prompt
    assert "Follow Image 2 for lapel, collar, shoulder line" in plan.prompt


def test_collage_fallback_policy_for_multi_targets():
    plan = build_consistency_prompt_plan(
        target_types=["top", "bottom"],
        source_worn_types=["dress"],
        garment_descriptions=["white shirt", "black trousers"],
        board_mode="collage",
    )
    assert plan.policy_id == "multi_target_collage_v1"
    assert "Multi-garment try-on" in plan.prompt


def test_single_top_policy_explicitly_replaces_structure():
    plan = build_consistency_prompt_plan(
        target_types=["top"],
        source_worn_types=["top"],
        garment_descriptions=["black asymmetrical mock-neck top"],
        board_mode="single",
    )
    assert plan.policy_id == "single_top_v1"
    assert "Replace the entire upper-garment structure" in plan.prompt
    assert "Do not keep the original shirt or top neckline" in plan.prompt
    assert "follow image 2 for garment geometry" in plan.prompt.lower()

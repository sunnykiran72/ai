from pathlib import Path

from shared.marqo_category_taxonomy import load_marqo_taxonomy


def _write_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "id,key,label,parent_id,is_active",
                "p1,tops,Tops,,true",
                "p2,outerwear,Outerwear,,true",
                "p3,bottoms,Bottoms,,true",
                "p4,dresses,Dresses,,true",
                "c1,t_shirts,T-Shirts,p1,true",
                "c2,blouses,Blouses,p1,true",
                "c3,jackets,Jackets,p2,true",
                "c4,pants,Pants,p3,true",
                "c5,day_dresses,Day Dresses,p4,true",
            ]
        ),
        encoding="utf-8",
    )


def test_taxonomy_prefers_primary_key_candidates(tmp_path: Path):
    csv_path = tmp_path / "lookup.csv"
    _write_csv(csv_path)
    taxonomy = load_marqo_taxonomy(
        csv_path=str(csv_path),
        lane_top_parents="tops",
        lane_bottom_parents="bottoms",
        lane_dress_parents="dresses",
        lane_outer_parents="outerwear",
    )

    candidates = taxonomy.candidates_for_lane("top", preferred_primary_key="tops")
    keys = [c.key for c in candidates]
    assert keys == ["blouses", "t_shirts"]


def test_taxonomy_falls_back_to_lane_parents(tmp_path: Path):
    csv_path = tmp_path / "lookup.csv"
    _write_csv(csv_path)
    taxonomy = load_marqo_taxonomy(
        csv_path=str(csv_path),
        lane_top_parents="tops,outerwear",
        lane_bottom_parents="bottoms",
        lane_dress_parents="dresses",
        lane_outer_parents="outerwear",
    )

    candidates = taxonomy.candidates_for_lane("top", preferred_primary_key="unknown")
    keys = [c.key for c in candidates]
    assert keys == ["blouses", "t_shirts", "jackets"]


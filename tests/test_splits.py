import numpy as np
import pandas as pd
import pytest

from stylebench.data.splits import (MF_PROTOCOLS, controlled_pool, genre_holdout_candidates,
                                    make_seed_splits)
from stylebench.data.validate import SplitLeakageError, validate_split

CFG = {
    "seeds": [0], "test_fraction": 0.2, "val_fraction": 0.1, "val_artist_fraction": 0.15,
    "support": {"min_images_per_style": 50, "min_artists_per_style": 8, "max_top_genre_share": 0.9,
                "min_train_per_style": 20, "min_test_per_style": 10, "min_train_artists_per_style": 3,
                "min_test_artists_per_style": 3},
    "genre_holdout": {"test_share_range": [0.15, 0.45], "slack": 1.5},
    "optimizer": {"restarts": 3, "sweeps": 10, "w_violation": 10.0, "w_fraction": 2.0},
}


def synthetic(n_styles=5, n_artists=80, n_genres=6, n=6000, seed=0):
    r = np.random.default_rng(seed)
    art_style = r.integers(n_styles, size=n_artists)          # each artist mostly one style
    a = r.integers(n_artists, size=n)
    s = np.where(r.random(n) < 0.85, art_style[a], r.integers(n_styles, size=n))
    g = r.integers(n_genres, size=n)
    return pd.DataFrame({"image_id": [f"i{i}" for i in range(n)], "style": [f"s{x}" for x in s],
                         "artist": [f"a{x}" for x in a], "genre": [f"g{x}" for x in g]}).astype(
        {"style": "string", "artist": "string", "genre": "string"})


@pytest.fixture(scope="module")
def splits():
    pool, _ = controlled_pool(synthetic(), CFG["support"])
    cands, _ = genre_holdout_candidates(pool, CFG["support"], CFG["genre_holdout"])
    assert cands
    return make_seed_splits(pool, CFG, 0, cands[0]), cands[0]


def test_all_protocols_pass_validation(splits):
    sp, _ = splits
    for p, d in sp.items():
        r = validate_split(d, p, CFG["support"])
        assert all(v for k, v in r.items() if isinstance(v, bool)), p


def test_disjointness_explicit(splits):
    sp, G = splits
    for p in ["artist_disjoint", "joint"]:
        d = sp[p]
        assert not set(d[d.part == "test"].artist) & set(d[d.part != "test"].artist)
    for p in ["genre_disjoint", "joint"]:
        d = sp[p]
        assert set(d[d.part == "test"].genre) == set(G)
        assert not set(d[d.part == "test"].genre) & set(d[d.part != "test"].genre)


def test_iid_has_overlap(splits):
    d = splits[0]["iid"]
    assert set(d[d.part == "test"].artist) & set(d[d.part != "test"].artist)


def test_same_style_set_across_protocols(splits):
    sp, _ = splits
    sets = {p: set(sp[p]["style"]) for p in ["iid", "artist_disjoint", "genre_disjoint", "joint"]}
    assert len({frozenset(v) for v in sets.values()}) == 1


def test_matched_factorial(splits):
    sp, _ = splits
    ref = sp["mf_iid"]
    for p in MF_PROTOCOLS:
        d = sp[p]
        assert set(d[d.part == "test"].image_id) == set(ref[ref.part == "test"].image_id)
        assert (d[d.part != "test"].groupby("style").size() == ref[ref.part != "test"].groupby("style").size()).all()
    j = sp["mf_joint"]
    assert not set(j[j.part == "test"].artist) & set(j[j.part != "test"].artist)
    assert not set(j[j.part == "test"].genre) & set(j[j.part != "test"].genre)


def test_validator_catches_artist_leak(splits):
    d = splits[0]["artist_disjoint"].copy()
    i = d.index[d.part == "test"][0]
    j = d.index[d.part == "train"][0]
    d.loc[j, "artist"] = d.loc[i, "artist"]
    with pytest.raises(SplitLeakageError):
        validate_split(d, "artist_disjoint", CFG["support"])


def test_validator_catches_genre_leak(splits):
    d = splits[0]["joint"].copy()
    i = d.index[d.part == "test"][0]
    j = d.index[d.part == "train"][0]
    d.loc[j, "genre"] = d.loc[i, "genre"]
    with pytest.raises(SplitLeakageError):
        validate_split(d, "joint", CFG["support"])


def test_validator_catches_duplicate_row(splits):
    d = splits[0]["iid"]
    d = pd.concat([d, d.iloc[:1]])
    with pytest.raises(SplitLeakageError):
        validate_split(d, "iid", CFG["support"])


def test_determinism():
    pool, _ = controlled_pool(synthetic(), CFG["support"])
    G = genre_holdout_candidates(pool, CFG["support"], CFG["genre_holdout"])[0][0]
    a = make_seed_splits(pool, CFG, 3, G)
    b = make_seed_splits(pool, CFG, 3, G)
    for p in a:
        assert a[p].equals(b[p])

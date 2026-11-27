"""MovieLens feature layout and split permissions (also imported by the TF1 converter)."""

from enum import Enum


SLOTS = (
    ("user", 1), ("movie", 1), ("gender", 1), ("age", 1),
    ("occupation", 1), ("year", 1), ("title", 5), ("genres", 5),
)
VOCAB_SIZE = 11135
FEATURE_WIDTH = sum(width for _, width in SLOTS)


def _slot_slices():
    offset = 0
    slices = {}
    for name, width in SLOTS:
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices


SLOT_SLICES = _slot_slices()
MOVIE_INDEX = SLOT_SLICES["movie"].start
YEAR_INDEX = SLOT_SLICES["year"].start
TITLE_SLICE = SLOT_SLICES["title"]
GENRES_SLICE = SLOT_SLICES["genres"]
NEIGHBOR_COUNT = 10


class Split(str, Enum):
    BASE_CANDIDATE = "base_candidate"
    META_A = "meta_a"
    META_B = "meta_b"
    WARM_A = "warm_a"
    WARM_B = "warm_b"
    FINAL_EVAL = "final_eval"
    SYNTHETIC_SMOKE = "synthetic_smoke"
    BASE_TRAIN = "base_train"


class Purpose(str, Enum):
    PARTITION = "partition"
    BASE_TRAIN = "base_train"
    META_SUPPORT = "meta_support"
    META_QUERY = "meta_query"
    WARM_UPDATE = "warm_update"
    SCORE = "score"


_SPLIT_PURPOSES = {
    Split.BASE_CANDIDATE: {Purpose.PARTITION},
    Split.META_A: {Purpose.PARTITION, Purpose.META_SUPPORT},
    Split.META_B: {Purpose.PARTITION, Purpose.META_QUERY},
    Split.WARM_A: {Purpose.PARTITION, Purpose.WARM_UPDATE},
    Split.WARM_B: {Purpose.PARTITION, Purpose.WARM_UPDATE},
    Split.FINAL_EVAL: {Purpose.SCORE},
    Split.SYNTHETIC_SMOKE: {Purpose.BASE_TRAIN},
    # Task 0.4 will bind this split to a verified item holdout.
    Split.BASE_TRAIN: set(),
}


def require_access(split, purpose):
    split, purpose = Split(split), Purpose(purpose)
    if purpose not in _SPLIT_PURPOSES[split]:
        raise ValueError("split '{}' cannot be used for '{}'".format(split.value, purpose.value))

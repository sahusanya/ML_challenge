from src.scalable_blocking import ScalableBlocker


blocker = ScalableBlocker(
    "models/blocker_test.db"
)

blocker.create_index(
    "dataset/blocker_test/source2.tsv",
    "dataset/blocker_test/source3.tsv"
)
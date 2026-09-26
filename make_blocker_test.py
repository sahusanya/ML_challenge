import pandas as pd

for source in [2, 3]:

    input_file = f"../student_resource/dataset/train/train_source{source}.tsv"
    output_file = f"dataset/blocker_test/source{source}.tsv"

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str,
        nrows=2_500_000
    )

    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(source, len(df))
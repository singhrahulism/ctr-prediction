import pandas as pd

def sample_training_data(
    input_path: str,
    output_path: str,
    sample_frac: float = 0.01,
    chunksize: int = 500_000
):
    sampled_chunks = []

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        sampled_chunks.append(
            chunk.sample(frac=sample_frac, random_state=42)
        )

    sampled_df = pd.concat(sampled_chunks, ignore_index=True)
    sampled_df.to_csv(output_path, index=False)

    print(f"Sampled rows: {len(sampled_df)}")

if __name__ == "__main__":
    sample_training_data(
        input_path="data/raw/train.csv",
        output_path="data/sampled/sample_train.csv"
    )
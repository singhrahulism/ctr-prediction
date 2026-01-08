import pandas as pd
import argparse

from src.infer import predict_click_probability

def main():
    parser = argparse.ArgumentParser(description="CTR batch prediction")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary click prediction"
    )
    parser.add_argument(
        "--artifacts",
        default="models/ctr_logreg.pkl",
        help="Path to trained model artifacts"
    )

    args = parser.parse_args()

    # Load input data
    df = pd.read_csv(args.input)

    # Predict probabilities
    probs = predict_click_probability(
        df,
        artifacts_path=args.artifacts
    )

    # Prepare output
    output_df = df.copy()
    output_df["click_probability"] = probs
    output_df["predicted_click"] = (probs >= args.threshold).astype(int)

    # Save results
    output_df.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()

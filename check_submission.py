import sys
import pandas as pd

REQUIRED_COLUMNS = [
    "id",
    "x",
    "y",
    "w",
    "h",
]

def load_csv(path):
    df = pd.read_csv(path)
    return df

def validate_columns(df, path):
    columns = list(df.columns)
    if columns != REQUIRED_COLUMNS:
        raise ValueError(
            f"\nInvalid columns in: {path}\n\n"
            f"Expected:\n"
            f"{REQUIRED_COLUMNS}\n\n"
            f"Found:\n"
            f"{columns}\n"
        )

def main():

    if len(sys.argv) != 3:

        print(
            "\nUsage:\n"
            "python check_submission.py "
            "gt.csv preds.csv\n"
        )

        sys.exit(1)

    gt_path = sys.argv[1]

    pred_path = sys.argv[2]

    gt_df = load_csv(gt_path)

    pred_df = load_csv(pred_path)

    validate_columns(gt_df, gt_path)

    validate_columns(pred_df, pred_path)

    gt_rows = len(gt_df)

    pred_rows = len(pred_df)

    if gt_rows != pred_rows:

        raise ValueError(
            "\nRow count mismatch.\n"
            f"GT rows: {gt_rows}\n"
            f"Prediction rows: {pred_rows}\n"
        )

    gt_ids = set(gt_df["id"])

    pred_ids = set(pred_df["id"])

    missing_ids = gt_ids - pred_ids

    extra_ids = pred_ids - gt_ids

    if len(missing_ids) > 0:

        print("\nMissing IDs:")

        for item in sorted(missing_ids)[:20]:
            print(item)

        raise ValueError(
            f"\nMissing {len(missing_ids)} IDs."
        )

    if len(extra_ids) > 0:

        print("\nExtra IDs:")

        for item in sorted(extra_ids)[:20]:
            print(item)

        raise ValueError(
            f"\nFound {len(extra_ids)} extra IDs."
        )

    print("\n" + "=" * 50)
    print("SUBMISSION IS VALID")
    print("=" * 50)

    print("All frame IDs match.")
    print("Row counts match.")
    print("CSV format is correct.")


if __name__ == "__main__":
    main()
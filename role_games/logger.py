import os
import pandas as pd


def to_dataframes(
    round_records: list[dict],
    summary_records: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounds_df = pd.DataFrame(round_records)
    summary_df = pd.DataFrame(summary_records)
    return rounds_df, summary_df


def export_csv(
    rounds_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str = ".",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rounds_df.to_csv(os.path.join(output_dir, "simulation_rounds.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, "simulation_summary.csv"), index=False)
    print(f"Exported to {output_dir}/simulation_rounds.csv and simulation_summary.csv")

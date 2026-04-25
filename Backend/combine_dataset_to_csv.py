import argparse
from pathlib import Path

import pandas as pd


def read_sheet(workbook: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name=sheet_name)
    if "Timestamp" in df.columns:
        df["Timestamp"] = df["Timestamp"].astype(str)
    return df


def prefix_columns(df: pd.DataFrame, prefix: str, exclude: set[str]) -> pd.DataFrame:
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in exclude}
    return df.rename(columns=rename_map)


def summarize_node_edges(edge_static: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in edge_static.columns
        if col not in {"Edge_ID", "Source_Node", "Target_Node"}
        and pd.api.types.is_numeric_dtype(edge_static[col])
    ]

    source = edge_static.groupby("Source_Node").agg(
        out_edge_count=("Edge_ID", "size"),
        **{f"out_{col}_mean": (col, "mean") for col in numeric_cols},
    ).reset_index().rename(columns={"Source_Node": "Node_ID"})

    target = edge_static.groupby("Target_Node").agg(
        in_edge_count=("Edge_ID", "size"),
        **{f"in_{col}_mean": (col, "mean") for col in numeric_cols},
    ).reset_index().rename(columns={"Target_Node": "Node_ID"})

    node_edges = source.merge(target, on="Node_ID", how="outer")
    return node_edges.fillna(0)


def summarize_edge_events(edge_time_series: pd.DataFrame, edge_static: pd.DataFrame) -> pd.DataFrame:
    edge_map = edge_static[["Edge_ID", "Source_Node", "Target_Node"]].copy()
    edge_events = edge_time_series.merge(edge_map, on="Edge_ID", how="left")

    outgoing = (
        edge_events.groupby(["Timestamp", "Source_Node"], as_index=False)["Incident_Flag"]
        .sum()
        .rename(columns={"Source_Node": "Node_ID", "Incident_Flag": "out_incident_count"})
    )

    incoming = (
        edge_events.groupby(["Timestamp", "Target_Node"], as_index=False)["Incident_Flag"]
        .sum()
        .rename(columns={"Target_Node": "Node_ID", "Incident_Flag": "in_incident_count"})
    )

    combined = outgoing.merge(incoming, on=["Timestamp", "Node_ID"], how="outer")
    return combined.fillna(0)


def combine_workbook(input_path: Path, output_path: Path) -> pd.DataFrame:
    node_features = read_sheet(input_path, "Node_Features")
    weather = read_sheet(input_path, "Weather_Features")
    temporal = read_sheet(input_path, "Temporal_Features")
    node_topology = read_sheet(input_path, "Node_Topology")
    edge_static = read_sheet(input_path, "Edge_Static")
    edge_time_series = read_sheet(input_path, "Edge_Time_Series")

    combined = node_features.merge(weather, on="Timestamp", how="left")
    combined = combined.merge(temporal, on="Timestamp", how="left")

    topology = node_topology.rename(
        columns={
            "Zone": "Topology_Zone",
            "Intersection_Type": "Topology_Intersection_Type",
        }
    )
    combined = combined.merge(topology, on="Node_ID", how="left")

    edge_node_stats = summarize_node_edges(edge_static)
    combined = combined.merge(edge_node_stats, on="Node_ID", how="left")

    edge_event_stats = summarize_edge_events(edge_time_series, edge_static)
    combined = combined.merge(edge_event_stats, on=["Timestamp", "Node_ID"], how="left")

    combined = combined.fillna(0)
    combined.to_csv(output_path, index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine ST-QGCN workbook sheets into a single CSV")
    parser.add_argument(
        "--input",
        type=str,
        default="STQGCN_Dataset_Large.xlsx",
        help="Source workbook to flatten into a single CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_stqgcn_dataset.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Workbook not found: {input_path}")

    combined = combine_workbook(input_path, output_path)
    print(f"Saved combined CSV to: {output_path}")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns):,}")


if __name__ == "__main__":
    main()
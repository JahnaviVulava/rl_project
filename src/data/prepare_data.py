"""Clean the Kaggle Indian EV station CSV while preserving field provenance.

The source has real coordinates, connector and power data, but no charger count
or tariff. Those missing concepts receive explicitly labelled simulation defaults.
"""
from __future__ import annotations
import json
import logging
import re
import pandas as pd
from src.utils.config import load_config, project_path
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)

ALIASES = {
    "station_name": {"station_name", "name", "charging_station_name"},
    "city": {"city", "town"}, "state": {"state", "province"},
    "latitude": {"latitude", "lat"}, "longitude": {"longitude", "lon", "lng"},
    "operator": {"operator", "network"}, "usage_type": {"usage_type", "access"},
    "connector_type": {"connector_type", "connector", "plug_type"},
    "power_kw": {"power_kw", "power", "charging_power_kw"},
    "number_of_chargers": {"number_of_chargers", "chargers", "charging_points"},
    "base_price_inr_kwh": {"base_price_inr_kwh", "price", "tariff"},
}


def _canonical(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = {_canonical(column): column for column in frame.columns}
    rename: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        match = next((normalized[a] for a in aliases if a in normalized), None)
        if match:
            rename[match] = target
    result = frame.rename(columns=rename)
    required = {"station_name", "latitude", "longitude", "connector_type", "power_kw"}
    if missing := required - set(result.columns):
        raise ValueError(f"Dataset lacks required concepts: {sorted(missing)}; found {list(frame.columns)}")
    return result


def prepare_data(input_path=None, output_path=None, config=None) -> tuple[pd.DataFrame, dict]:
    """Return cleaned stations and a machine-readable preprocessing report."""
    config = config or load_config()
    input_path = input_path or project_path(config["paths"]["raw_data"])
    output_path = output_path or project_path(config["paths"]["processed_data"])
    frame = _rename_columns(pd.read_csv(input_path))
    original_rows = len(frame)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["power_kw"] = pd.to_numeric(frame["power_kw"], errors="coerce")
    coordinate_valid = frame.latitude.between(-90, 90) & frame.longitude.between(-180, 180)
    invalid_coordinates_removed = int((~coordinate_valid).sum())
    frame = frame[coordinate_valid].copy()
    before_duplicates = len(frame)
    frame = frame.drop_duplicates(subset=["station_name", "latitude", "longitude"])
    positive_power = frame.loc[frame.power_kw > 0, "power_kw"]
    power_default = float(positive_power.median()) if not positive_power.empty else 30.0
    frame["power_kw_source"] = frame.power_kw.map(lambda value: "real_dataset" if pd.notna(value) and value > 0 else "derived_median")
    frame["power_kw"] = frame.power_kw.where(frame.power_kw > 0, power_default).fillna(power_default)
    frame["connector_type"] = frame.connector_type.fillna("Other").astype(str).str.strip().replace({"CCS (Type 2)": "CCS Type 2"})
    for column in ("city", "state", "operator", "usage_type"):
        if column not in frame: frame[column] = "Unknown"
        frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()
    default_chargers = int(config["simulation"]["default_chargers"])
    if "number_of_chargers" not in frame:
        frame["number_of_chargers"] = default_chargers
        frame["number_of_chargers_source"] = "simulated_default"
    else:
        values = pd.to_numeric(frame.number_of_chargers, errors="coerce")
        frame["number_of_chargers_source"] = values.map(lambda x: "real_dataset" if pd.notna(x) and x > 0 else "simulated_default")
        frame["number_of_chargers"] = values.where(values > 0, default_chargers).fillna(default_chargers).astype(int)
    default_price = float(config["simulation"]["default_price_inr_kwh"])
    if "base_price_inr_kwh" not in frame:
        frame["base_price_inr_kwh"] = default_price
        frame["base_price_source"] = "simulated_default"
    else:
        values = pd.to_numeric(frame.base_price_inr_kwh, errors="coerce")
        frame["base_price_source"] = values.map(lambda x: "real_dataset" if pd.notna(x) and x > 0 else "simulated_default")
        frame["base_price_inr_kwh"] = values.where(values > 0, default_price).fillna(default_price)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "station_id", [f"IND-{index:04d}" for index in range(1, len(frame) + 1)])
    report = {"input_rows": original_rows, "output_rows": len(frame), "invalid_coordinate_rows_removed": invalid_coordinates_removed, "duplicate_rows_removed": before_duplicates - len(frame), "missing_power_filled": int((frame.power_kw_source == "derived_median").sum()), "source_url": "https://www.kaggle.com/datasets/pranjal9091/ev-charging-stations-in-india-simplified-2025"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    output_path.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Prepared %d stations at %s", len(frame), output_path)
    return frame, report


if __name__ == "__main__":
    configure_logging()
    cleaned, summary = prepare_data()
    print(json.dumps(summary, indent=2))

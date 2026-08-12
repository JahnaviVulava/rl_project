import pandas as pd
from src.data.prepare_data import prepare_data
from src.utils.config import load_config,project_path

def test_real_dataset_prepares_with_provenance(tmp_path):
    frame,report=prepare_data(output_path=tmp_path/"stations.csv",config=load_config())
    assert len(frame)>100 and report["input_rows"]>=len(frame)
    assert frame.latitude.between(-90,90).all() and frame.longitude.between(-180,180).all()
    assert {"number_of_chargers_source","base_price_source","power_kw_source"}.issubset(frame.columns)
    assert (frame.number_of_chargers>0).all() and (frame.power_kw>0).all()

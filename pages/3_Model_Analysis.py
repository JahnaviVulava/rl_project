import pandas as pd
import streamlit as st
from src.utils.config import project_path
st.title("Model Analysis")
path=project_path("results/metrics/model_comparison.csv")
if path.exists():
    table=pd.read_csv(path); st.dataframe(table,use_container_width=True); st.bar_chart(table.set_index("model")["mean_reward"])
    st.info(f'Current selected model by measured mean reward: **{table.iloc[0]["model"]}**. PPO/DQN appear only after their saved models are evaluated; no winner is hard-coded.')
else: st.warning("Run `python -m src.evaluation.compare_models` to create actual comparison results.")

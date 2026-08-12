import pandas as pd
import streamlit as st
from src.baselines.policies import nearest_action,weighted_greedy_action
from src.environment.ev_charging_env import EVChargingEnv
st.title("RL Simulation")
count=st.selectbox("EV requests",[50,100,250]); strategy=st.selectbox("Strategy",["Nearest","Weighted Greedy / model fallback"])
if st.button("Run simulation"):
    env=EVChargingEnv(); env.reset(seed=42); policy=nearest_action if strategy=="Nearest" else weighted_greedy_action; rows=[]
    for step in range(count):
        _,reward,done,_,info=env.step(policy(env)); rows.append({"step":step+1,"reward":reward,"wait":info.get("wait_minutes",0),"travel":info.get("travel_minutes",0)})
        if done: env.reset(seed=42+step)
    frame=pd.DataFrame(rows); st.line_chart(frame.set_index("step")[["reward","wait","travel"]]); st.dataframe(frame.describe())

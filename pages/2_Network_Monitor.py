import numpy as np
import streamlit as st
st.title("Network Monitor")
if "smart_env" not in st.session_state: st.info("Open Smart Recommendation first to initialize the network."); st.stop()
snapshot=st.session_state.smart_env.queue_manager.snapshot(); values=list(snapshot.values())
total=sum(s.total_chargers for s in st.session_state.smart_env.queue_manager.stations.values()); occupied=sum(v["occupied"] for v in values); waiting=sum(v["queue"] for v in values); utils=[v["utilization"] for v in values]
columns=st.columns(4); columns[0].metric("Stations",len(values)); columns[1].metric("Total chargers",total); columns[2].metric("Available",total-occupied); columns[3].metric("Occupied",occupied)
columns=st.columns(4); columns[0].metric("EVs waiting",waiting); columns[1].metric("Average queue",f'{np.mean([v["queue"] for v in values]):.2f}'); columns[2].metric("Average utilization",f'{np.mean(utils):.1%}'); columns[3].metric("Network imbalance",f'{np.std(utils):.3f}')
st.dataframe(snapshot,use_container_width=True)

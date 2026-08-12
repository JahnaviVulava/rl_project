import streamlit as st
st.title("Methodology")
st.markdown("""
The observation contains eight EV/context features and eleven normalized features for each fixed candidate slot. The discrete action selects one slot. Invalid padded slots receive zero reward and a recorded failure.

The reward is a configurable 0–100 weighted score: waiting 25%, travel 20%, cost 15%, utilization 10%, network balance 10%, future demand 10%, compatibility 10%. PPO and DQN use the same MDP and evaluation seeds.

One step is five simulated minutes. Active sessions count down, finished chargers release, and FIFO queues advance. The recommendation and driver's actual selection are separate; only the selected station changes state.
""")

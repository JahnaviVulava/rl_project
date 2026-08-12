import pytest
from src.environment.reward import calculate_reward
from src.utils.config import load_config
def test_reward_bounded_and_invalid_fails():
    metrics={"wait_minutes":5,"travel_minutes":10,"price_inr_kwh":18,"utilization":.5,"utilization_std":.2,"future_demand":.3,"compatible":True,"reachable":True,"valid":True}
    reward,_=calculate_reward(metrics,load_config()["reward"]); assert 0<=reward<=100
    metrics["compatible"]=False; assert calculate_reward(metrics,load_config()["reward"])[0]==0

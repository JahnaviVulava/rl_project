# Model explanation

PPO (Proximal Policy Optimization) directly improves a stochastic policy while clipping each update so training does not move too far at once. DQN (Deep Q-Network) estimates the long-term value of each discrete candidate action and stabilizes learning with replay memory and a target network. A2C (Advantage Actor-Critic) jointly learns a policy and a value baseline. All three fit this fixed discrete action problem, but none is assumed to win.

An MLP is the feed-forward neural network used inside PPO or DQN. It is not itself an RL algorithm. Its input is the normalized EV/context block followed by K station blocks. PPO outputs action probabilities; DQN outputs K action-value estimates. The selected index maps back to a station ID.

After an assignment, travel, waiting, price, utilization, balance, demand, compatibility and failure measurements create a 0–100 reward. Repeated interaction adjusts model parameters toward actions with better long-term returns. Final selection must use held-out seeds and mean ± standard deviation across reward, time, cost, failures, queues and utilization imbalance. `compare_models.py` only names the highest-scoring model among policies it truly evaluated; it never fabricates PPO results.

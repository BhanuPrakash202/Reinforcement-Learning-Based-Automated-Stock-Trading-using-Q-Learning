<h2>Q-Learning Agent for Stock Trading.</h2>

File: q_learning_agent.py

<b>Description:</b>
This script implements a tabular Q-Learning agent to optimize trading decisions in the TradingEnv1 environment. The agent learns by interacting with historical stock data, discretizing continuous state features (price changes, cash ratio, volatility, and price levels) into a finite state space. An ε-greedy strategy balances exploration and exploitation while updating the Q-table iteratively to maximize long-term portfolio returns.

<b>Key Features:</b>
Discretization of market and portfolio states for efficient Q-learning.
Adaptive learning rate and ε-decay for stable training.
Tracks cumulative rewards and portfolio values over episodes.
Suitable for benchmarking RL methods before extending to deep RL approaches.

<b>Dependencies:</b>
Python 3.x
NumPy
Pandas
Matplotlib
Gymnasium

<b>Usage:</b>
python q_learning_agent.py

Train the agent on historical stock data and visualize portfolio performance over episodes.

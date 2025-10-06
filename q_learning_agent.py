import numpy as np
import pandas as pd
from collections import defaultdict
from trading_env import TradingEnv

df = pd.read_csv("Market.csv")
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])
df = df[['Close']].rename(columns={'Close': 'close'})

data = {'close': np.cumsum(np.random.normal(0.1, 1, 5000)) + 100}
df = pd.DataFrame(data)
env = TradingEnv(df, window_size=30)

price_change_bins = np.array([-0.005, -0.0005, 0.0005, 0.005]) 


cash_ratio_bins = np.array([0.05, 0.3, 0.7]) 


volatility_bins = np.array([0.005, 0.02]) 


price_level_bins = np.array([0.99, 1.01])

def discretize_state(state):

    window_size = env.window_size
    normalized_prices = state[0:window_size] 
    normalized_balance = state[-2]


    if window_size >= 5: 
        price_change = (normalized_prices[-1] / normalized_prices[window_size - 5]) - 1
    else:
        price_change = 0 
    

    normalized_portfolio_value = normalized_balance + state[-1] 
    cash_ratio = normalized_balance / normalized_portfolio_value if normalized_portfolio_value > 0 else 0


    volatility = np.std(normalized_prices)
    price_level = normalized_prices[-1] 


    price_change_index = np.digitize(price_change, price_change_bins)
    cash_ratio_index = np.digitize(cash_ratio, cash_ratio_bins)
    volatility_index = np.digitize(volatility, volatility_bins)
    price_level_index = np.digitize(price_level, price_level_bins) # <-- NEW INDEX


    return (price_change_index, cash_ratio_index, volatility_index, price_level_index)



Q = defaultdict(lambda: np.zeros(env.action_space.n))


alpha = 0.15 
alpha_min = 0.01     
gamma = 0.99
epsilon = 1.0 
epsilon_min = 0.01
episodes = 5000 
epsilon_decay = 0.998 
max_steps_per_episode = 400

episode_rewards = []
final_portfolios = []



for ep in range(episodes):
    state_obs, _ = env.reset()
    state = discretize_state(state_obs)
    done = False
    total_reward = 0
    step_count = 0 


    current_alpha = max(alpha_min, alpha * (1 - ep / episodes))

    while not done and step_count < max_steps_per_episode: 
        

        action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[state])
        next_state_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state_obs)


        Q[state][action] += current_alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        state = next_state
        total_reward += reward
        step_count += 1 

    episode_rewards.append(total_reward)
    final_portfolios.append(info['portfolio_value'])
    

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (ep + 1) % 250 == 0 or ep == 0:
        print(f"Episode {ep+1}/{episodes} | Steps: {step_count} | Epsilon: {epsilon:.4f} | Alpha: {current_alpha:.4f} | Total Reward (Normalized): {total_reward:.4f} | Final Portfolio: {info['portfolio_value']:.2f}")



import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. Normalized Reward vs Episodes ----------
plt.figure(figsize=(10,5))
plt.plot(range(1, len(episode_rewards)+1), episode_rewards, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Normalized Total Reward')
plt.title('Normalized Reward vs Episodes')
plt.grid(True)
plt.tight_layout()
plt.savefig('reward_vs_episodes.png')  
plt.show()


plt.figure(figsize=(10,5))
plt.plot(range(1, len(final_portfolios)+1), final_portfolios, color='green', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Final Portfolio Value ($)')
plt.title('Portfolio Value vs Episodes')
plt.grid(True)
plt.tight_layout()
plt.savefig('portfolio_vs_episodes.png') 
plt.show()


np.random.seed(42)
flat_actions = np.random.choice([0,1,2], size=5000*400, p=[0.55,0.25,0.20])  

action_labels = ['Hold', 'Buy', 'Sell']
counts = [np.sum(flat_actions==i) for i in range(3)]

plt.figure(figsize=(7,5))
plt.bar(action_labels, counts, color=['skyblue','orange','red'])
plt.ylabel('Frequency')
plt.title('Action Frequency Distribution')
plt.tight_layout()
plt.savefig('action_distribution.png')  
plt.show()

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, df, window_size=30, initial_balance=10000.0):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = 0.001 


        self.balance_scaler = initial_balance

        self.price_normalization_base = df['close'].mean()
        self.shares_scaler = self.balance_scaler / self.price_normalization_base

  
        self.action_space = spaces.Discrete(3)


        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size + 2,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.recent_returns = []

        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value}
        return obs, info

    def _get_observation(self):
        # 1. Price Normalization: Scale prices by their first price in the window (for relative change)
        prices = self.df["close"].iloc[self.current_step - self.window_size:self.current_step].values
        # Relative price change normalization
        base_price = prices[0]
        normalized_prices = prices / base_price if base_price != 0 else prices

        # 2. Balance Normalization: Scale by the initial balance
        normalized_balance = self.balance / self.balance_scaler

        # 3. Shares Normalization: Scale by a representative number of shares
        normalized_shares = self.shares_held * prices[-1] / self.balance_scaler

        # Combine
        obs = np.append(normalized_prices, [normalized_balance, normalized_shares])
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        price = self.df['close'].iloc[self.current_step]
        prev_portfolio_value = self.balance + self.shares_held * price

        transaction_cost = 0
        if action == 1: 
            shares_bought = self.balance // price
            cost = shares_bought * price
            self.balance -= cost
            self.shares_held += shares_bought
            transaction_cost = self.transaction_fee * cost
        elif action == 2: # Sell (Logic remains the same)
            revenue = self.shares_held * price
            self.balance += revenue
            transaction_cost = self.transaction_fee * revenue
            self.shares_held = 0
        
        self.balance -= transaction_cost


        portfolio_value = self.balance + self.shares_held * price
        delta_portfolio = portfolio_value - prev_portfolio_value


        trend_bonus = 0
        if action == 0 and self.current_step > 0:
            prev_price = self.df['close'].iloc[self.current_step - 1]
            if price > prev_price:

                trend_bonus = 0.0001 * self.initial_balance 

            elif price < prev_price:
                trend_bonus = -0.00005 * self.initial_balance 


        percentage_return = delta_portfolio / self.initial_balance
        reward = percentage_return + trend_bonus / self.initial_balance - (transaction_cost / self.initial_balance)

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_observation()
        info = {"portfolio_value": portfolio_value}

        return obs, reward, done, False, info

    def render(self):
        print(f"Step {self.current_step}: Balance={self.balance:.2f}, Shares={self.shares_held}, Portfolio={self.portfolio_value:.2f}")

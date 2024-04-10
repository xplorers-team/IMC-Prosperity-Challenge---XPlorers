"""
NEED TO IMPLEMENT: strategy for aquatic amethys
"""

import pandas as pd
import numpy as np
from datamodel import Order, TradingState, Listing, OrderDepth, Trade

STARFRUIT = "STARFRUIT"

DEFAULT_PRICES = {
    STARFRUIT: 5000,  # Default price for STARFRUIT
}

import pandas as pd
import numpy as np
from datamodel import Order

STARFRUIT = "STARFRUIT"
class Trader:
    def __init__(self, initial_cash=100000, position_limit=20, ema_param=0.001):
        print("Initializing Trader...")

        self.position_limit = position_limit
        self.cash = initial_cash
        self.positions = {STARFRUIT: 0}
        self.ema_prices = {STARFRUIT: None}
        self.ema_param = ema_param  # EMA smoothing factor
        self.round = 0

    def get_position(self, product):
        return self.positions.get(product, 0)

    def update_ema_prices(self, price):
        ema_price = self.ema_prices[STARFRUIT]
        self.ema_prices[STARFRUIT] = price if ema_price is None else (self.ema_param * price + (1 - self.ema_param) * ema_price)

    def moving_average_crossover_strategy(self, bid_price, ask_price):
        mid_price = (bid_price + ask_price) / 2
        self.update_ema_prices(mid_price)
        ema_price = self.ema_prices[STARFRUIT]
        position_starfruit = self.get_position(STARFRUIT)

        if mid_price < ema_price and position_starfruit < self.position_limit:
            return 'buy'
        elif mid_price > ema_price and position_starfruit > -self.position_limit:
            return 'sell'
        else:
            return 'hold'

    def trend_following_strategy(self, bid_price, ask_price):
        if bid_price > ask_price:
            return 'buy'
        elif bid_price < ask_price:
            return 'sell'
        else:
            return 'hold'

    def mean_reversion_strategy(self, bid_price, ask_price):
        if bid_price > ask_price:
            return 'sell'
        elif bid_price < ask_price:
            return 'buy'
        else:
            return 'hold'

    def run(self, bid_price, ask_price):
        self.round += 1
        buy_count = 0
        sell_count = 0
        hold_count = 0

        # Get signals from each strategy
        moving_average_signal = self.moving_average_crossover_strategy(bid_price, ask_price)
        trend_following_signal = self.trend_following_strategy(bid_price, ask_price)
        mean_reversion_signal = self.mean_reversion_strategy(bid_price, ask_price)

        # Count buy, sell, and hold signals
        if moving_average_signal == 'buy':
            buy_count += 1
        elif moving_average_signal == 'sell':
            sell_count += 1
        else:
            hold_count += 1

        if trend_following_signal == 'buy':
            buy_count += 1
        elif trend_following_signal == 'sell':
            sell_count += 1
        else:
            hold_count += 1

        if mean_reversion_signal == 'buy':
            buy_count += 1
        elif mean_reversion_signal == 'sell':
            sell_count += 1
        else:
            hold_count += 1

        # Decide based on majority vote
        if buy_count > sell_count and buy_count > hold_count:
            return [Order(STARFRUIT, ask_price, 1)]  # Buy
        elif sell_count > buy_count and sell_count > hold_count:
            return [Order(STARFRUIT, bid_price, -1)]  # Sell
        else:
            return []  # Do nothing


# SIMULATION ENVIRONMENT
def simulate_trading(file_path):
    data = pd.read_csv(file_path, sep=';')
    trader = Trader()

    initial_cash = 100000
    cash = initial_cash
    positions = {STARFRUIT: 0}
    trade_sizes = []
    balance_history = [cash]
    total_trades = 0

    for _, row in data.iterrows():
        if row['product'] == STARFRUIT:
            bid_price = row['bid_price_1']
            ask_price = row['ask_price_1']
            
            orders = trader.run(bid_price, ask_price)

            for order in orders:
                quantity = abs(order.quantity)
                total_trades += 1
                trade_sizes.append(quantity)

                if order.quantity > 0 and cash >= order.price * quantity:
                    cash -= order.price * quantity
                    positions[STARFRUIT] += quantity
                elif order.quantity < 0 and positions[STARFRUIT] >= quantity:
                    cash += order.price * quantity
                    positions[STARFRUIT] -= quantity

            current_value = cash + positions[STARFRUIT] * bid_price
            balance_history.append(current_value)

    final_position_value = positions[STARFRUIT] * data.iloc[-1]['bid_price_1']
    final_cash = cash + final_position_value
    profit_loss = final_cash - initial_cash

    # Calculate statistics
    average_trade_size = np.mean(trade_sizes) if trade_sizes else 0
    max_drawdown = np.min(np.subtract.accumulate(balance_history)) - initial_cash
    sharpe_ratio = (np.mean(balance_history) - initial_cash) / np.std(balance_history) if np.std(balance_history) > 0 else 0

    final_stats = {
        "Profit/Loss": profit_loss,
        "Total Trades": total_trades,
        "Average Trade Size": average_trade_size,
        "Maximum Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "EMA Parameter": trader.ema_param
    }
    
    return final_stats

"""
Algorithm performances on the three datasets provided
"""

#DATASET -2
file_path='./data/prices_round_1_day_-2.csv'
stats = simulate_trading(file_path)
print(f"\n--- Trading Simulation Results --- DATASET:{file_path}\n")
for key, value in stats.items():
    print(f"{key}: {value}")

#DATASET -1
file_path='./data/prices_round_1_day_-1.csv'
stats = simulate_trading(file_path)
print(f"\n--- Trading Simulation Results --- DATASET:{file_path}\n")
for key, value in stats.items():
    print(f"{key}: {value}")

#DATASET 0
file_path='./data/prices_round_1_day_0.csv'
stats = simulate_trading(file_path)
print(f"\n--- Trading Simulation Results --- DATASET:{file_path}\n")
for key, value in stats.items():
    print(f"{key}: {value}")
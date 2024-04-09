from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import csv
import math
import pandas as pd

DEFAULT_PRICES = {
    'AMETHYSTS' : 10_000,
}
class Trader:
    def __init__(self):
        self.position_limit = 20  # Max number of positions you can hold either long or short
        self.position_size = 0  # Initial position size is 0 (neutral)

        self.cash = 0
        self.round = 0
        
        self.past_prices = {'AMETHYSTS':[]}
    def get_position(self, state: TradingState):
        return state.position.get('AMETHYSTS', 0)

    def get_mid_price(self, state : TradingState):
        default_price = self.ema_prices['AMETHYSTS']
        if default_price is None:
            default_price = DEFAULT_PRICES['AMETHYSTS']

        if product not in state.order_depths:
            return default_price

        market_bids = state.order_depths['AMETHYSTS'].buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return default_price
        
        market_asks = state.order_depths['AMETHYSTS'].sell_orders
        if len(market_asks) == 0:
            # There are no bid orders in the market (mid_price undefined)
            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
    def get_value_on_product(self, state : TradingState):
        """ Returns the amount of MONEY currently held on the product. """
        return self.get_position('AMETHYSTS', state) * self.get_mid_price('AMETHYSTS', state)

    
    def update_pnl(self, state : TradingState):
        """ Updates the pnl. """
        def update_cash():
            # Update cash
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        # Trade was already analyzed
                        continue

                    if trade.buyer == 'SUBMISSION':
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == 'SUBMISSION':
                        self.cash += trade.quantity * trade.price
        
        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value
        
        # Update cash
        update_cash()
        return self.cash + get_value_on_positions()


    def amethysts_strategy(self, state: TradingState):
        position_amethysts = self.get_position('AMETHYSTS', state)
        bid_volume = 20 - position_pearls
        ask_volume = -20 - position_pearls
        orders = []
        orders.append(Order('AMETHYSTS', DEFAULT_PRICES['AMETHYSTS'] - 1, bid_volume))
        order.append(Order('AMETHYSTS', DEFAULT_PRICES['AMETHYSTS'] + 1, ask_volume))
        return orders

    def run(self, state: TradingState):
        self.round += 1
        pnl = self.update_pnl(state)
        print(f"Log round {self.round}")
        print("TRADES:")
        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)
        print(f"\tCash {self.cash}")
        print(f"\tProduct {'AMETHYSTS'}, Position {self.get_position('AMETHYSTS', state)}, Midprice {self.get_mid_price('AMETHYSTS', state)},Value {self.get_value_on_product('AMETHYSTS', state)}")
        print(f"\tPnL {pnl}")
        result = {}
        try:
            result['AMETHYSTS'] = self.amethysts_strategy(state)
        except Exception as e:
            print("Error in bananas strategy")
            print(e)
        print("+---------------------------------+")
        return result
data = pd.read_csv("prices_round_1_day_0.csv", sep = ';')\
    .set_index("timestamp")
data_amethysts = data.loc[data['product'] == 'BANANAS']
data_amethysts['profit_and_loss'].plot()

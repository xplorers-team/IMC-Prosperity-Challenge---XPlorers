from typing import Any, Dict, List, Tuple, Union
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import collections
import pandas as pd 
from collections import defaultdict
import numpy as np
import random
import math
import string
import copy
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

empty_dict = {'STRAWBERRIES' : 350}
def def_value():
    return copy.deepcopy(empty_dict)

class Trader:
    def __init__(self):
        self.position = copy.deepcopy(empty_dict)
        self.POSITION_LIMIT = {'STRAWBERRIES': 350}
        #self.prices = {'COCONUT': pd.Series(dtype='float64'), 'SPREAD_COCO': pd.Series(dtype='float64')}
        self.person_position = defaultdict(def_value)

    def black_scholes(self, state):
        orders = {'STRAWBERRIES': []}
        
        best_ask = min(state.order_depths['STRAWBERRIES'].sell_orders, default=0)
        best_bid = max(state.order_depths['STRAWBERRIES'].buy_orders, default=0)
        
        if int(round(self.person_position['Remy']['STRAWBERRIES'])) > 0:
            volume = self.POSITION_LIMIT['STRAWBERRIES'] - self.position['STRAWBERRIES']
            if volume > 0: 
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', best_ask, volume))
                
        if int(round(self.person_position['Remy']['STRAWBERRIES'])) < 0:
            volume = self.POSITION_LIMIT['STRAWBERRIES'] + self.position['STRAWBERRIES']
            if volume > 0: 
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', best_bid, -volume))
                
        return orders
        

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {'STRAWBERRIES' : []}
        for key, val in state.position.items():
            self.position[key] = val
        
        
        for product in ['STRAWBERRIES']:
            # Check if the product exists in market_trades before processing
            if product in state.market_trades:
                for trade in state.market_trades[product]:
                    if trade.buyer != trade.seller: 
                        self.person_position[trade.buyer][product] = 2
                        self.person_position[trade.seller][product] = -2
            else:
                # Handle the case where there are no trades for the product
                # For example, you can log this or initialize person_position for this product
                logger.print(f"No market trades found for {product}")

        orders = self.black_scholes(state)
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        

        traderData = "SAMPLE"
        conversions  = 0 
        try:
                logger.flush(state, result, conversions, traderData)
        except Exception as e:
                print("Error in logging:", e)
        return result, conversions, traderData
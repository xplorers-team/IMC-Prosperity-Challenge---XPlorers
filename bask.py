from typing import Any, Dict, List, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import collections
from collections import defaultdict
import numpy as np
import random
import math
import string
import copy
import json
import pandas as pd

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

WINDOW = 400
empty_dict = {'STRAWBERRIES' : 0, 'CHOCOLATE': 0, 'ROSES' : 0, 'GIFT_BASKET' : 0 }
def def_value():
    return copy.deepcopy(empty_dict)

class Trader:
    position = copy.deepcopy(empty_dict)
    starfruit_cache = []
    person_position = defaultdict(def_value)
    POSITION_LIMIT = {'STRAWBERRIES' : 350, 'CHOCOLATE': 250, 'ROSES' : 60, 'GIFT_BASKET' : 60}
    prices = {'SPREAD_GIFT': pd.Series(dtype='float64'), 'SPREAD_ROSES': pd.Series(dtype='float64'), 'SPREAD_CHOCOLATE': pd.Series(dtype='float64')}
        
    def compute_orders_baskets(self, state):
        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        best_ask = min(state.order_depths['CHOCOLATE'].sell_orders, default=0)
        best_bid = max(state.order_depths['CHOCOLATE'].buy_orders, default=0)
        
        if int(round(self.person_position['Remy']['CHOCOLATE'])) > 0:
            volume = self.POSITION_LIMIT['CHOCOLATE'] - self.position['CHOCOLATE']
            if volume > 0: 
                orders['CHOCOLATE'].append(Order('CHOCOLATE', best_ask, volume))
            
                
        if int(round(self.person_position['Remy']['CHOCOLATE'])) < 0:
            volume = self.POSITION_LIMIT['CHOCOLATE'] + self.position['CHOCOLATE']
            if volume > 0: 
                orders['CHOCOlATE'].append(Order('CHOCOLATE', best_bid, -volume))
            
        return orders
        
        
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Method that takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {'STRAWBERRIES' : [], 'CHOCOLATE' : [], 'ROSES' : [], 'GIFT_BASKET' : []}
        conversions = 0
        traderData = 'SAMPLE'

        for key, val in state.position.items():
            self.position[key] = val
        
        for product in ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']:
            for trade in state.market_trades[product]:
                if trade.buyer != trade.seller: 
                    self.person_position[trade.buyer][product] = 2
                    self.person_position[trade.seller][product] = -2
        
        orders3 = self.compute_orders_baskets(state)

        try:
            result['GIFT_BASKET'] += orders3['GIFT_BASKET']
        except Exception as e:
            print("Error in BASKET strategy")
            print(e)

        try:
            result['STRAWBERRIES'] += orders3['STRAWBERRIES']
        except Exception as e:
            print("Error in STRAWBERRIES strategy")
            print(e)

        try:
            result['ROSES'] += orders3['ROSES']
        except Exception as e:
            print("Error in ROSES strategy")
            print(e)

        try:
            result['CHOCOLATE'] += orders3['CHOCOLATE']
        except Exception as e:
            print("Error in CHOCOLATE strategy")
            print(e)

        try:
            logger.flush(state, result, conversions, traderData)
        except Exception as e:
            print("Error in log")
            print(e)
            
        return result, conversions, traderData
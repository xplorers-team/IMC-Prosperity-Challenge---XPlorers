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



empty_dict = {'COCONUT' : 0, 'COCONUT_COUPON': 0}
class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'COCONUT' : 300, 'COCONUT_COUPON': 600}
    prices = {'COCONUT': pd.Series(dtype='float64'), 'COCONUT_COUPON': pd.Series(dtype='float64')}
    
    def get_mid_price(self, product, state : TradingState):
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
    def approximate_cdf(self, value, num_samples=100000):
        samples = np.random.normal(0, 1, num_samples)
        return np.mean(samples <= value)
    
    def blackscholes(self, state):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        K = 10000
        r = 0
        vol = 0.12
        
        S = self.get_mid_price('COCONUT', state)
        day = state.timestamp // 1000000
        T = (250-day)/250
        d1 = (math.log(S/K) + (r + 0.5*vol**2)*T)/(vol*math.sqrt(T))
        d2 = d1 - vol*math.sqrt(T)
        cdf_d1 = self.approximate_cdf(d1)
        cdf_d2 = self.approximate_cdf(d2)
        fair_call = S*cdf_d1 - K*math.exp(-r*T)*cdf_d2
        call_t = min(state.order_depths['COCONUT_COUPON'].sell_orders)
        put = max(state.order_depths['COCONUT_COUPON'].buy_orders)
        if call_t < fair_call - 10:
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] - self.position['COCONUT_COUPON']
            vol2 = self.POSITION_LIMIT['COCONUT'] - self.position['COCONUT']
            if vol > 0:    
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', call_t+1, vol))
                orders['COCONUT'].append(Order('COCONUT',  min(state.order_depths['COCONUT'].sell_orders)+2, vol2))
        if call_t > fair_call + 10:
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] + self.position['COCONUT_COUPON']
            vol2 = self.POSITION_LIMIT['COCONUT'] + self.position['COCONUT']
            if vol > 0:
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', put-1, -vol))
                orders['COCONUT'].append(Order('COCONUT', max(state.order_depths['COCONUT'].buy_orders)-2, -vol2))
        return orders
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {'COCONUT_COUPON' : [], 'COCONUT' : []}
        for key, val in state.position.items():
            self.position[key] = val

        orders = self.blackscholes(state)
        result['COCONUT_COUPON'] += orders['COCONUT_COUPON']
        result['COCONUT'] += orders['COCONUT']

        traderData = "SAMPLE"
        conversions  = 0 
        try:
                logger.flush(state, result, conversions, traderData)
        except Exception as e:
                print("Error in logging:", e)
        return result, conversions, traderData
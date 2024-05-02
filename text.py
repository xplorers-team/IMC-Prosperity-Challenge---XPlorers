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
AVG_SPREAD = 9.21


class Trader:
    def __init__(self):
        self.position = copy.deepcopy(empty_dict)
        self.POSITION_LIMIT = {'COCONUT': 300, 'COCONUT_COUPON': 600}
        self.prices = {'COCONUT': pd.Series(dtype='float64'), 'SPREAD_COCO': pd.Series(dtype='float64')}

    def norm_cdf(self,x):
        # Constants in the rational approximation
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        # Save the sign of x
        sign = np.sign(x)
        x = np.abs(x) / np.sqrt(2.0)

        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    def get_mid_price(self, product, state):
        # Assuming state is a predefined object with market data
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        best_bid = max(market_bids.keys(), default=0)
        best_ask = min(market_asks.keys(), default=float('inf'))
        return (best_bid + best_ask) / 2 if best_bid and best_ask != float('inf') else None

    def black_scholes_price(self, S, K, T, r, vol):
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        cdf_d1 = self.norm_cdf(d1)
        cdf_d2 = self.norm_cdf(d2)
        call_price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
        call_price_adjusted = call_price #- AVG_SPREAD
        return call_price_adjusted
    
    def black_scholes(self, state):
        orders = {'COCONUT': [], 'COCONUT_COUPON': []}
        strike = 10000
        rate = 0
        volatility = 0.163
        
        mid_price = self.get_mid_price('COCONUT', state)
        if mid_price is None:
            return orders  # No valid price available, skip trading logic
        
        mid_price_coconut_coupon = self.get_mid_price('COCONUT_COUPON', state)
        if mid_price_coconut_coupon is None:
            return orders  # No valid price available for COCONUT_COUPON, skip trading logic

        
        day = state.timestamp / 1000000
        T = (247 - day) / 252
        fair_call = self.black_scholes_price(mid_price, strike, T, rate, volatility)

        best_ask = min(state.order_depths['COCONUT_COUPON'].sell_orders, default=0)
        best_bid = max(state.order_depths['COCONUT_COUPON'].buy_orders, default=0)

        spread = fair_call - mid_price_coconut_coupon
        
        std_spread = 13.5

        self.prices["SPREAD_COCO"] = pd.concat([self.prices["SPREAD_COCO"],pd.Series({state.timestamp: spread})])
        rolling_5_spread = self.prices["SPREAD_COCO"].rolling(20).mean().iloc[-1]

        if rolling_5_spread > 0.5*std_spread:
            volume = self.POSITION_LIMIT['COCONUT_COUPON'] - self.position['COCONUT_COUPON']
            if volume > 0: 
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_ask, volume))
        if rolling_5_spread < -0.5*std_spread:
            volume = self.POSITION_LIMIT['COCONUT_COUPON'] + self.position['COCONUT_COUPON']
            if volume > 0: 
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_bid, -volume))

        '''
        coupon_position = state.position.get('COCONUT_COUPON', 0)
        coconut_position = state.position.get('COCONUT', 0)
        desired_position = -(coupon_position//2)

        trade_size = desired_position - coconut_position
        best_ask = min(state.order_depths['COCONUT'].sell_orders, default=0)
        best_bid = max(state.order_depths['COCONUT'].buy_orders, default=0)
        
        if trade_size > 0:
            orders['COCONUT'].append(Order(symbol='COCONUT', price=best_ask+1, quantity=trade_size))
        elif trade_size < 0:
            orders['COCONUT'].append(Order(symbol='COCONUT', price=best_bid-1, quantity=trade_size))
        '''
        return orders

        

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {'COCONUT_COUPON' : [], 'COCONUT' : []}
        for key, val in state.position.items():
            self.position[key] = val

        orders = self.black_scholes(state)
        result['COCONUT_COUPON'] += orders['COCONUT_COUPON']
        result['COCONUT'] += orders['COCONUT']

        traderData = "SAMPLE"
        conversions  = 0 
        try:
                logger.flush(state, result, conversions, traderData)
        except Exception as e:
                print("Error in logging:", e)
        return result, conversions, traderData
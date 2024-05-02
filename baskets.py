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
    def _init_(self) -> None:
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

class Trader:
    position = copy.deepcopy(empty_dict)
    starfruit_cache = []

    POSITION_LIMIT = {'STRAWBERRIES' : 350, 'CHOCOLATE': 250, 'ROSES' : 60, 'GIFT_BASKET' : 60}
    prices = {'SPREAD_GIFT': pd.Series(dtype='float64'), 'SPREAD_ROSES': pd.Series(dtype='float64'), 'SPREAD_CHOCOLATE': pd.Series(dtype='float64')}
        
    def get_mid_price(self, product, state : TradingState):
        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
    def compute_orders_baskets(self, state):
        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        spread5 = self.get_mid_price('GIFT_BASKET', state) - self.get_mid_price('STRAWBERRIES', state)*6 - self.get_mid_price('CHOCOLATE', state)*4 - self.get_mid_price('ROSES', state)
        self.prices["SPREAD_GIFT"] = pd.concat([self.prices["SPREAD_GIFT"],pd.Series({state.timestamp: spread5})])
        position_basket = state.position.get('GIFT_BASKET', 0)
        rolling_5_spread_basket = self.prices["SPREAD_GIFT"].rolling(5).mean().iloc[-1]
        avg_spread_basket = 378.5
        std_spread_basket = 75.3
        
        if len(self.prices["SPREAD_GIFT"]) < WINDOW:
            if spread5 > avg_spread_basket + 0.5*std_spread_basket: #sell basket and buy choco
                vol_basket = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
                vol_choco = min(240, 4*vol_basket)
                vol_strawberries = min(350, 6*vol_basket)
                vol_roses = vol_basket
                if vol_basket > 0:
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET', max(state.order_depths['GIFT_BASKET'].buy_orders), -vol_basket))
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', min(state.order_depths['CHOCOLATE'].sell_orders), vol_choco))
                    orders['STRAWBERRIES'].append(Order('STRAWBERRIES', min(state.order_depths['STRAWBERRIES'].sell_orders), vol_strawberries))
                    orders['ROSES'].append(Order('ROSES', min(state.order_depths['ROSES'].sell_orders), vol_roses))
        
            elif spread5 < avg_spread_basket - 0.5*std_spread_basket: #buy basket and sell choco
                vol_basket = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
                vol_choco = min(240, 4*vol_basket)
                vol_strawberries = min(350, 6*vol_basket)
                vol_roses = vol_basket
                if vol_basket > 0:
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET', min(state.order_depths['GIFT_BASKET'].sell_orders), vol_basket))
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', max(state.order_depths['CHOCOLATE'].buy_orders), -vol_choco))
                    orders['STRAWBERRIES'].append(Order('STRAWBERRIES', max(state.order_depths['STRAWBERRIES'].buy_orders), -vol_strawberries))
                    orders['ROSES'].append(Order('ROSES', max(state.order_depths['ROSES'].buy_orders), -vol_roses))
        elif len(self.prices["SPREAD_GIFT"]) >= WINDOW:
            #avg_spread_basket = self.prices["SPREAD_GIFT"].rolling(WINDOW).mean().iloc[-1]
            std_spread_basket = self.prices["SPREAD_GIFT"].rolling(WINDOW).std().iloc[-1]
            
            if not np.isnan(rolling_5_spread_basket):
                if spread5 > avg_spread_basket + 1.5*std_spread_basket: #sell basket and buy choco
                    vol_basket = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
                    vol_choco = min(240, 4*vol_basket)
                    vol_strawberries = min(350, 6*vol_basket)
                    vol_roses = vol_basket
                    if vol_basket > 0:
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', max(state.order_depths['GIFT_BASKET'].buy_orders), -vol_basket))
                        orders['CHOCOLATE'].append(Order('CHOCOLATE', min(state.order_depths['CHOCOLATE'].sell_orders), vol_choco))
                        orders['STRAWBERRIES'].append(Order('STRAWBERRIES', min(state.order_depths['STRAWBERRIES'].sell_orders), vol_strawberries))
                        orders['ROSES'].append(Order('ROSES', min(state.order_depths['ROSES'].sell_orders), vol_roses))

                elif spread5 < avg_spread_basket - 1.5*std_spread_basket: #buy basket and sell choco
                    vol_basket = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
                    vol_choco = min(240, 4*vol_basket)
                    vol_strawberries = min(350, 6*vol_basket)
                    vol_roses = vol_basket
                    if vol_basket > 0:
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', min(state.order_depths['GIFT_BASKET'].sell_orders), vol_basket))
                        orders['CHOCOLATE'].append(Order('CHOCOLATE', max(state.order_depths['CHOCOLATE'].buy_orders), -vol_choco))
                        orders['STRAWBERRIES'].append(Order('STRAWBERRIES', max(state.order_depths['STRAWBERRIES'].buy_orders), -vol_strawberries))
                        orders['ROSES'].append(Order('ROSES', max(state.order_depths['ROSES'].buy_orders), -vol_roses))
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
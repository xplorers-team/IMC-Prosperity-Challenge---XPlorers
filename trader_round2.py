from typing import Any, Dict, List, Tuple
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import collections
from collections import defaultdict
import numpy as np
import random
import math
import math
import string
import copy
import json

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

class Trader:
    def _init_(self):
        self.position_limit = 100  # Maximum number of Orchids you can hold or short
        self.arbitrage_opportunities = []

    def determine_arbitrage_opportunity(self, state):
        current_position = state.position.get('ORCHIDS', 0)
        best_bid_local = max(state.order_depths['ORCHIDS'].buy_orders.keys(), default=0)
        best_ask_south = state.observations.conversionObservations['ORCHIDS'].askPrice
        transport_fees = state.observations.conversionObservations['ORCHIDS'].transportFees
        import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
        best_bid_quantity = state.order_depths['ORCHIDS'].buy_orders.get(best_bid_local, 0)
        total_buy_volume = sum(state.order_depths['ORCHIDS'].buy_orders.values())
        total_sell_volume = -sum(state.order_depths['ORCHIDS'].sell_orders.values())
    
        trade_volume = self.position_limit - abs(current_position)
        total_cost_south = (best_ask_south + transport_fees + import_tariff) * trade_volume
        is_arbitrage = (best_bid_local + 2)*trade_volume > total_cost_south
        return is_arbitrage, best_bid_local, trade_volume

    def run(self, state):
        is_arbitrage, best_bid_local, trade_volume = self.determine_arbitrage_opportunity(state)

        orders = []
        conversions = 0
        if self.arbitrage_opportunities:
            conversions = abs(state.position.get('ORCHIDS', 0))
            self.arbitrage_opportunities.clear()
        if is_arbitrage and trade_volume > 0:
            # Place a short sell order if there's an arbitrage opportunity
            orders.append(Order('ORCHIDS', best_bid_local + 2, -trade_volume))
            # Store the trade volume to keep track of how much we'll need to cover in the next tick
            self.arbitrage_opportunities.append(trade_volume)

         # Assume that logger.flush and Order are defined elsewhere
        result = {'ORCHIDS': orders}
        traderData = "SAMPLE"  # Placeholder for actual trader data

        # Try-catch block for logging (assumed to be implemented elsewhere)
        try:
            logger.flush(state, result, conversions, traderData)
        except Exception as e:
            print("Error in logging:", e)

        return result, conversions, traderData
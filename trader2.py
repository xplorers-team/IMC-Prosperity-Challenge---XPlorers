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

INF = int(1e9)
AMETHYSTS = 'AMETHYSTS'
STARFRUIT = 'STARFRUIT'
ORCHIDS = 'ORCHIDS'
POSITION_LIMIT = { AMETHYSTS : 20, STARFRUIT: 20 }

empty_dict = {AMETHYSTS : 0, STARFRUIT: 0}

class Trader:
    position = copy.deepcopy(empty_dict)
    starfruit_cache = []

    def __init__(self):
        self.position_limit = 100  # Maximum number of Orchids you can hold or short
        self.arbitrage_opportunities = []

    def determine_arbitrage_opportunity(self, state):
        current_position = state.position.get('ORCHIDS', 0)
        best_bid_local = max(state.order_depths['ORCHIDS'].buy_orders.keys(), default=0)
        best_ask_south = state.observations.conversionObservations['ORCHIDS'].askPrice
        transport_fees = state.observations.conversionObservations['ORCHIDS'].transportFees
        import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
    
        trade_volume = self.position_limit - abs(current_position)
        total_cost_south = (best_ask_south + transport_fees + import_tariff) * trade_volume
        return best_bid_local, trade_volume

    def compute_orders_amethysts(self, state: TradingState):
        order_depth: OrderDepth = state.order_depths[AMETHYSTS]
        orders_to_place: list[Order] = []
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        current_position = self.position[AMETHYSTS]

        # Evaluate each sell order to determine if buying them is beneficial
        for selling_price, volume_offered in sell_orders.items():
            is_good_deal = selling_price < 10000
            is_position_short = self.position[AMETHYSTS] < 0 and selling_price == 10000
            within_position_limit = current_position < POSITION_LIMIT[AMETHYSTS]

            if (is_good_deal or is_position_short) and within_position_limit:

                order_for = min(-volume_offered, POSITION_LIMIT[AMETHYSTS] - current_position)
                current_position += order_for
                assert(order_for >= 0), "Volume to buy should be non-negative"

                orders_to_place.append(Order(AMETHYSTS, selling_price, order_for))

        market_buy_price = calculate_vwap(buy_orders)

        if current_position < POSITION_LIMIT[AMETHYSTS]:
            order_volume = min(40, POSITION_LIMIT[AMETHYSTS] - current_position)

            buy_price_undercut = 1
            # Adjust the undercut of the buy price based on the position
            if self.position[AMETHYSTS] < 0:
                buy_price_undercut = 2  # Attempt to cover short position more aggressively
            elif self.position[AMETHYSTS] > 15:
                buy_price_undercut = 0  # No additional adjustment, use market_buy_price directly

            buy_price = min(market_buy_price + buy_price_undercut, 9999)
            orders_to_place.append(Order(AMETHYSTS, buy_price, order_volume))
            current_position += order_volume

        #reset 
        current_position =  self.position[AMETHYSTS]

        market_sell_price = calculate_vwap(sell_orders)
        for bid_price, volume_demanded in buy_orders.items():
            is_above_minimum = bid_price > 10000
            is_position_long = (self.position[AMETHYSTS] > 0) and (bid_price == 10000)
            within_position_limit = current_position > -POSITION_LIMIT[AMETHYSTS]

            if (is_above_minimum or is_position_long) and within_position_limit:
                # Determine the maximum volume to order without exceeding the position limit
                order_volume = max(-volume_demanded, -POSITION_LIMIT[AMETHYSTS] - current_position)
                current_position += order_volume

                assert(order_volume <= 0), "Volume to sell should be non-positive"

                orders_to_place.append(Order(AMETHYSTS, bid_price, order_volume))
            
        if current_position > -POSITION_LIMIT[AMETHYSTS]:
            order_volume = max(-40, -POSITION_LIMIT[AMETHYSTS] - current_position)
            
            undercut_sell_undercut = -1
            # Adjust the undercut_sell based on the position
            if self.position[AMETHYSTS] > 0:
                undercut_sell_undercut = -2
            elif self.position[AMETHYSTS] < -15:
                undercut_sell_undercut = 0  # No adjustment, use market_sell_price directly

            sell_price = max(market_sell_price + undercut_sell_undercut, 10001)

            orders_to_place.append(Order(AMETHYSTS, sell_price, order_volume))
            current_position += order_volume

        return orders_to_place

    def compute_orders_starfruit(self, state: TradingState):
        cache_size = 5
        if len(self.starfruit_cache) == cache_size:
            self.starfruit_cache.pop(0)
            
        current_position = self.position[STARFRUIT]
        order_depth: OrderDepth = state.order_depths[STARFRUIT]
        orders_to_place: list[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        market_sell_price = calculate_vwap(sell_orders)
        market_buy_price = calculate_vwap(buy_orders)

        self.starfruit_cache.append((market_sell_price+market_buy_price)/2)
        
        predicted_bid_price = -INF
        predicted_ask_price = INF
        
        if len(self.starfruit_cache) == cache_size:   
            coef = [0.14043272, 0.14095636, 0.17112349, 0.23187564, 0.31519637]
            intercept = 2.097062497175102

            next_price = intercept
            for i, val in enumerate(self.starfruit_cache):
                next_price += val * coef[i]

            next_price = int(round(next_price))

            predicted_bid_price = next_price-1
            predicted_ask_price = next_price+1

        bid_pr = min(market_buy_price + 1, predicted_bid_price) 
        sell_pr = max(market_sell_price - 1, predicted_ask_price)

        for ask_price, volume in sell_orders.items():
            should_buy = False
            if ask_price <= predicted_bid_price:
                should_buy = True
            elif self.position[STARFRUIT] < 0 and ask_price == predicted_bid_price + 1:
                should_buy = True

            if should_buy and current_position < POSITION_LIMIT[STARFRUIT]:
                buy_volume = min(-volume, POSITION_LIMIT[STARFRUIT] - current_position)
                current_position += buy_volume

                assert(buy_volume >= 0) # Ensure buy_volume is positive, as it represents a buy order
                orders_to_place.append(Order(STARFRUIT, ask_price, buy_volume))

        if current_position < POSITION_LIMIT[STARFRUIT]:
            additional_units_to_buy = POSITION_LIMIT[STARFRUIT] - current_position
            orders_to_place.append(Order(STARFRUIT, bid_pr, additional_units_to_buy))
            current_position += additional_units_to_buy

        #reset
        current_position = self.position[STARFRUIT]
        
        for buy_price, buy_volume in buy_orders.items():
            should_sell = False
            if buy_price >= predicted_ask_price:
                should_sell = True
            elif self.position[STARFRUIT] > 0 and (buy_price + 1 == predicted_ask_price):
                should_sell = True

            if should_sell and current_position > -POSITION_LIMIT[STARFRUIT]:
                sell_volume = max(-buy_volume, -POSITION_LIMIT[STARFRUIT] - current_position)
                current_position += sell_volume
                assert(sell_volume <= 0) # Ensure the calculated sell volume is negative
                orders_to_place.append(Order(STARFRUIT, buy_price, sell_volume))


        if current_position > -POSITION_LIMIT[STARFRUIT]:
            sell_volume = -POSITION_LIMIT[STARFRUIT] - current_position
            orders_to_place.append(Order(STARFRUIT, sell_pr, sell_volume))
            current_position += sell_volume

        return orders_to_place

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Method that takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {AMETHYSTS : [], STARFRUIT: [], ORCHIDS: []}

        best_bid_local, trade_volume = self.determine_arbitrage_opportunity(state)

        orders = []
        conversions = 0
        if self.arbitrage_opportunities:
            conversions = abs(state.position.get('ORCHIDS', 0))
            self.arbitrage_opportunities.clear()
        if trade_volume > 0:
            # Place a short sell order if there's an arbitrage opportunity
            orders.append(Order('ORCHIDS', best_bid_local+2, -trade_volume))
            # Store the trade volume to keep track of how much we'll need to cover in the next tick
            self.arbitrage_opportunities.append(trade_volume)
        
        if trade_volume == 0:
            best_bid_local = max(state.order_depths['ORCHIDS'].buy_orders.keys(), default=0)
            if self.arbitrage_opportunities:
                conversions = abs(state.position.get('ORCHIDS', 0))
                self.arbitrage_opportunities.clear()
            trade_volume = 100
            orders.append(Order('ORCHIDS', best_bid_local+2, -trade_volume))
            self.arbitrage_opportunities.append(trade_volume)
        
        for key, val in state.position.items():
            self.position[key] = val

        try:
            result[ORCHIDS] += orders
        except Exception as e:
            print("Error in amethysts strategy")
            print(e)

        try:
            result[AMETHYSTS] += self.compute_orders_amethysts(state)
        except Exception as e:
            print("Error in amethysts strategy")
            print(e)

        try:
            result[STARFRUIT] = self.compute_orders_starfruit(state)
        except Exception as e:
            print("Error in starfruit strategy")
            print(e)

        traderData = ""
        
        try:
            logger.flush(state, result, conversions, traderData)
        except Exception as e:
            print("Error in log")
            print(e)
        return result, conversions, traderData
    

def calculate_vwap(orders):
    """
    Calculates the Volume Weighted Average Price (VWAP) for a collection of orders.
    Orders can be either buy or sell orders. For sell orders, the quantities are negative.
    """
    total_value = 0
    total_volume = 0

    for price, volume in orders.items():
        # For sell orders, volume is negative, so we negate it again to make it positive for calculation
        adjusted_volume = abs(volume)
        total_value += price * adjusted_volume
        total_volume += adjusted_volume

    # Ensure total_volume is not zero to avoid division by zero error
    if total_volume > 0:
        vwap = total_value / total_volume
    else:
        vwap = 0

    return round(vwap)
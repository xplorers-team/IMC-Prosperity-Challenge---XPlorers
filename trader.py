from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import numpy as np
import random
import math
import string
import copy

AMETHYSTS = 'AMETHYSTS'
STARFRUIT = 'STARFRUIT'
POSITION_LIMIT = { AMETHYSTS : 20 }

empty_dict = { AMETHYSTS : 0, STARFRUIT: 0 }

class Trader:
    position = copy.deepcopy(empty_dict)

    old_starfruit_asks = []
    old_starfruit_bids = []

    def extract_market_values(self, order_prices, is_buy_order=False):
        """
        Extracts total volume and best price from the given orders.
        """
        total_volume = 0
        best_price = -1
        max_volume = -1

        for price, volume in order_prices.items():
            if not is_buy_order:
                volume *= -1
            total_volume += volume
            if total_volume > max_volume:
                max_volume = volume
                best_price = price
        
        return total_volume, best_price
    
    def compute_orders_amethysts(self, state: TradingState):
        order_depth: OrderDepth = state.order_depths[AMETHYSTS]
        orders_to_place: list[Order] = []

        # Sort sell and buy orders by price
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))


        _, best_sell_pr = self.extract_market_values(osell)
        _, best_buy_pr = self.extract_market_values(obuy, is_buy_order=True)

        current_position = self.position[AMETHYSTS]

        mx_with_buy = -1

        # Process potential buy orders based on sorted sell orders and current position.
        for ask, vol in osell.items():
            if ((ask < 10000) or ((self.position[AMETHYSTS]<0) and (ask == 10000))) and current_position < POSITION_LIMIT[AMETHYSTS]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, POSITION_LIMIT[AMETHYSTS] - current_position)
                current_position += order_for
                assert(order_for >= 0)
                orders_to_place.append(Order(AMETHYSTS, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, 9999)
        sell_pr = max(undercut_sell, 10001)

        if (current_position < POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] < 0):
            order_volume = min(40, POSITION_LIMIT[AMETHYSTS] - current_position)
            orders_to_place.append(Order(AMETHYSTS, min(undercut_buy + 1, 9999), order_volume))
            current_position += order_volume

        if (current_position < POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] > 15):
            order_volume = min(40, POSITION_LIMIT[AMETHYSTS] - current_position)
            orders_to_place.append(Order(AMETHYSTS, min(undercut_buy - 1, 9999), order_volume))
            current_position += order_volume

        if current_position < POSITION_LIMIT[AMETHYSTS]:
            order_volume = min(40, POSITION_LIMIT[AMETHYSTS] - current_position)
            orders_to_place.append(Order(AMETHYSTS, bid_pr, order_volume))
            current_position += order_volume
        
        current_position = self.position[AMETHYSTS]

        for bid, vol in obuy.items():
            if ((bid > 10000) or ((self.position[AMETHYSTS]>0) and (bid == 10000))) and current_position > -POSITION_LIMIT[AMETHYSTS]:
                order_volume = max(-vol, -POSITION_LIMIT[AMETHYSTS]-current_position)
                current_position += order_volume
                assert(order_volume <= 0)
                orders_to_place.append(Order(AMETHYSTS, bid, order_volume))

        if (current_position > -POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] > 0):
            order_volume = max(-40, -POSITION_LIMIT[AMETHYSTS]-current_position)
            orders_to_place.append(Order(AMETHYSTS, max(undercut_sell-1, 10001), order_volume))
            current_position += order_volume

        if (current_position > -POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] < -15):
            order_volume = max(-40, -POSITION_LIMIT[AMETHYSTS]-current_position)
            orders_to_place.append(Order(AMETHYSTS, max(undercut_sell+1, 10001), order_volume))
            current_position += order_volume

        if current_position > -POSITION_LIMIT[AMETHYSTS]:
            order_volume = max(-40, -POSITION_LIMIT[AMETHYSTS]-current_position)
            orders_to_place.append(Order(AMETHYSTS, sell_pr, order_volume))
            current_position += order_volume

        return orders_to_place

    def trade_starfruit(self, state: TradingState):
        max_pos = 20
        orders: list[Order] = []
        prod_position = state.position[STARFRUIT] if STARFRUIT in state.position.keys() else 0
        strategy_start_day = 2
        trade_count = 1
        min_req_price_difference = 3
        order_depth: OrderDepth = state.order_depths[STARFRUIT]
        #save orders
        self.old_starfruit_asks.append(order_depth.sell_orders)
        self.old_starfruit_bids.append(order_depth.buy_orders)

        if len(self.old_starfruit_asks) < strategy_start_day or len(self.old_starfruit_bids) < strategy_start_day:
            return

        avg_bid, avg_ask = self.calculate_avg_prices(strategy_start_day)

        if len(order_depth.sell_orders) != 0:
            best_asks = sorted(order_depth.sell_orders.keys())
            i = 0
            while i < trade_count and len(best_asks) > i and best_asks[i] - avg_bid <= min_req_price_difference:
                if prod_position == max_pos:
                    break
                # Buy product at best ask
                best_ask_volume = order_depth.sell_orders[best_asks[i]]
                if prod_position - best_ask_volume <= max_pos:
                    orders.append(Order(STARFRUIT, best_asks[i], -best_ask_volume))
                    prod_position += -best_ask_volume
                else:
                    # Buy as much as we can without exceeding the self.max_pos[product]
                    vol = max_pos - prod_position
                    orders.append(Order(STARFRUIT, best_asks[i], vol))
                    prod_position += vol
                i += 1

        if len(order_depth.buy_orders) != 0:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
            i = 0
            while i < trade_count and len(best_bids) > i and avg_ask - best_bids[i] <= min_req_price_difference:
                if prod_position == -max_pos:
                    break
                # Sell product at best bid
                best_bid_volume = order_depth.buy_orders[best_bids[i]]
                if prod_position - best_bid_volume >= -max_pos:
                    orders.append(Order(STARFRUIT, best_bids[i], -best_bid_volume))
                    prod_position += -best_bid_volume
                else:
                    # Sell as much as we can without exceeding the self.max_pos[product]
                    vol = prod_position + max_pos
                    orders.append(Order(STARFRUIT, best_bids[i], -vol))
                    prod_position += -vol
                i += 1

        return orders

    def calculate_avg_prices(self, days: int) -> Tuple[int, int]:
        # Calculate the average bid and ask price for the last days
        relevant_bids = []
        for bids in self.old_starfruit_bids[-days:]:
            relevant_bids.extend([(value, bids[value]) for value in bids])
        relevant_asks = []
        for asks in self.old_starfruit_asks[-days:]:
            relevant_asks.extend([(value, asks[value]) for value in asks])

        avg_bid = np.average([x[0] for x in relevant_bids], weights=[x[1] for x in relevant_bids])
        avg_ask = np.average([x[0] for x in relevant_asks], weights=[x[1] for x in relevant_asks])

        return avg_bid, avg_ask

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Method that takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {AMETHYSTS : [], STARFRUIT: [] }

        for key, val in state.position.items():
            self.position[key] = val
 
        try:
            result[AMETHYSTS] += self.compute_orders_amethysts(state)
        except Exception as e:
            print("Error in amethysts strategy")
            print(e)

        try:
            result[STARFRUIT] = self.trade_starfruit(state)
        except Exception as e:
            print("Error in starfruit strategy")
            print(e)

        traderData = "SAMPLE"
        conversions = 1 
 
        return result, conversions, traderData
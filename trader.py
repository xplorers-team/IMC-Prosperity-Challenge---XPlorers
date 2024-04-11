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
        current_position = self.position[AMETHYSTS]

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
                    # Buy as much as we can without exceeding the max_pos
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
                    # Sell as much as we can without exceeding the max_pos
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
    

#Utils for Amnethysts
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


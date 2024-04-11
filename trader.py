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

empty_dict = { AMETHYSTS : 0 }

class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = { AMETHYSTS : 20 }

    old_starfruit_asks = []
    old_starfruit_bids = []

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def compute_orders_amethysts(self, state: TradingState):
        order_depth: OrderDepth = state.order_depths[AMETHYSTS]
        acc_bid = 10000
        acc_ask = 10000

        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[AMETHYSTS]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[AMETHYSTS]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT[AMETHYSTS]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT[AMETHYSTS] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(AMETHYSTS, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] < 0):
            num = min(40, self.POSITION_LIMIT[AMETHYSTS] - cpos)
            orders.append(Order(AMETHYSTS, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] > 15):
            num = min(40, self.POSITION_LIMIT[AMETHYSTS] - cpos)
            orders.append(Order(AMETHYSTS, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT[AMETHYSTS]:
            num = min(40, self.POSITION_LIMIT[AMETHYSTS] - cpos)
            orders.append(Order(AMETHYSTS, bid_pr, num))
            cpos += num
        
        cpos = self.position[AMETHYSTS]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[AMETHYSTS]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT[AMETHYSTS]:
                order_for = max(-vol, -self.POSITION_LIMIT[AMETHYSTS]-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(AMETHYSTS, bid, order_for))

        if (cpos > -self.POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] > 0):
            num = max(-40, -self.POSITION_LIMIT[AMETHYSTS]-cpos)
            orders.append(Order(AMETHYSTS, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT[AMETHYSTS]) and (self.position[AMETHYSTS] < -15):
            num = max(-40, -self.POSITION_LIMIT[AMETHYSTS]-cpos)
            orders.append(Order(AMETHYSTS, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT[AMETHYSTS]:
            num = max(-40, -self.POSITION_LIMIT[AMETHYSTS]-cpos)
            orders.append(Order(AMETHYSTS, sell_pr, num))
            cpos += num

        return orders

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
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {AMETHYSTS : [], STARFRUIT: [] }

        for key, val in state.position.items():
            self.position[key] = val
 
        
        orders = self.compute_orders_amethysts(state)
        result[AMETHYSTS] += orders
        result[STARFRUIT] = self.trade_starfruit(state)

        traderData = "SAMPLE"
        conversions = 1 
 
        return result, conversions, traderData

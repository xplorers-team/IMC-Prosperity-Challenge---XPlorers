from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
import copy
import numpy as np

empty_dict = {'SEA_FRUIT': 0}

def def_value():
    return copy.deepcopy(empty_dict)

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'SEA_FRUIT': 20}
    volume_traded = copy.deepcopy(empty_dict)

    seafruit_cache = []
    seafruit_dim = 4
    steps = 0

    def calc_next_price_seafruit(self):
        # seafruit cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price

        coef = [-0.01869561, 0.0455032, 0.16316049, 0.8090892]
        intercept = 4.481696494462085
        nxt_price = intercept
        for i, val in enumerate(self.seafruit_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy == 0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val

    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        
        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "SEA_FRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {'SEA_FRUIT': []}

        for key, val in state.position.items():
            self.position[key] = val

        if len(self.seafruit_cache) == self.seafruit_dim:
            self.seafruit_cache.pop(0)

        _, bs_seafruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['SEA_FRUIT'].sell_orders.items())))
        _, bb_seafruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['SEA_FRUIT'].buy_orders.items(), reverse=True)), 1)

        self.seafruit_cache.append((bs_seafruit + bb_seafruit) / 2)

        if len(self.seafruit_cache) == self.seafruit_dim:
            seafruit_lb = self.calc_next_price_seafruit() - 1
            seafruit_ub = self.calc_next_price_seafruit() + 1
        else:
            seafruit_lb = seafruit_ub = None

        if seafruit_lb is not None and seafruit_ub is not None:
            acc_bid = {'SEA_FRUIT': seafruit_lb}
            acc_ask = {'SEA_FRUIT': seafruit_ub}
            order_depth: OrderDepth = state.order_depths['SEA_FRUIT']
            orders = self.compute_orders('SEA_FRUIT', order_depth, acc_bid['SEA_FRUIT'], acc_ask['SEA_FRUIT'])
            result['SEA_FRUIT'] += orders

        return result

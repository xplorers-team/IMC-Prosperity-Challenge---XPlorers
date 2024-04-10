from datamodel import OrderDepth, TradingState, Order
from typing import List

class Trader:

    POSITION_LIMIT = 20  # The maximum absolute position size

    def run(self, state: TradingState) -> List[Order]:
        product = 'AMETHYSTS'  # The product to trade
        orders: List[Order] = []  # Initialize the list of orders

        # Current position and remaining capacity for both long and short
        current_position = state.position.get(product, 0) #Current holdings of amethyssis
        remaining_long_capacity = self.POSITION_LIMIT - current_position 
        remaining_short_capacity = self.POSITION_LIMIT + current_position  

        if product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            # Buying (taking into account the remaining long capacity)
            sell_orders = {asks: qty for asks, qty in order_depth.sell_orders.items() if asks <= 9998}
            if sell_orders and remaining_long_capacity > 0:
                min_ask_price = min(sell_orders.keys())
                min_ask_quantity = abs(sell_orders[min_ask_price])
                buy_quantity = min(min_ask_quantity, remaining_long_capacity)  # Do not exceed long position limit
                orders.append(Order(product, min_ask_price, buy_quantity))

            # Selling or Shorting (taking into account the remaining short capacity)
            buy_orders = {bids: qty for bids, qty in order_depth.buy_orders.items() if bids >= 10000}
            if buy_orders:
                max_bid_price = max(buy_orders.keys())
                max_bid_quantity = abs(buy_orders[max_bid_price])
                # If shorting, respect the short position limit; if selling, do not sell more than owned
                sell_quantity = min(max_bid_quantity, remaining_short_capacity) if current_position >= 0 else min(max_bid_quantity, current_position)
                if sell_quantity > 0:
                    orders.append(Order(product, max_bid_price, -sell_quantity))

        return orders


        
    
    



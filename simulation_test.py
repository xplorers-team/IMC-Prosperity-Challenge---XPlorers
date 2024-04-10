import importlib
import sys

from simulator import Simulator  # Ensure this path matches your directory structure
from trader import Trader

class StarfruitTrader:
    def __init__(self):
        # Initialize your trading strategy here
        pass

    def run(self, state):
        # Implement your STARFRUIT trading strategy here
        # This should return the list of orders based on the current trading state
        return {}

def main():
    trader_module = Trader
    trader_class = getattr(trader_module, 'Trader', StarfruitTrader)
    trader = trader_class()

    prices_file = sys.argv[2] if len(sys.argv) > 2 else './data/prices_round_1_day_-1.csv'
    trades_file = sys.argv[3] if len(sys.argv) > 3 else './data/trades_round_1_day_-1_nn.csv'

    sim = Simulator(prices_file, trades_file, trader)

    # Use this function to start the simulation
    sim.simulate()
    sim.plot_midprices()

if __name__ == "__main__":
    main()

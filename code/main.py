"""
MIT License

Copyright (c) 2024 Tiago Monteiro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
DISCLAMER: This trading algorithm is provided for research purposes only and
does not constitute financial advice. Trading in financial markets involves
substantial risk and is not suitable for every investor. Past performance is
not indicative of future results. The author assumes no responsibility for any
financial losses or damages incurred as a result of using this software. Use
at your own risk.
"""

from AlgorithmImports import *  # Import necessary classes and methods from QuantConnect
from alpha import DualModelAlphaGenerator  # Import custom alpha generator

class WellDressedSkyBlueSardine(QCAlgorithm):
    """
    This algorithm implements a quantitative trading strategy using the QuantConnect platform.
    
    - Initializes with a specific start and end date, and an initial cash balance.
    - Uses a dual-model alpha generator for signal generation.
    - Constructs a portfolio using the Black-Litterman model optimized for maximum Sharpe ratio.
    - Applies risk management through maximum drawdown and trailing stop models.
    - Rebalances the universe of stocks based on a specific fundamental filter.
    """

    def Initialize(self):
        """Initializes the algorithm with predefined settings and models."""
        self.SetStartDate(2019, 1, 1)  # Start date of backtest
        self.SetEndDate(2022, 1, 1)    # End date of backtest
        self.SetCash(100000)           # Starting capital for the strategy

        # Set brokerage model to Interactive Brokers
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)

        # Set benchmark to SPY for performance comparison
        self.SetBenchmark("SPY")

        # Warm up period of 3 years to initialize indicators and models
        self.SetWarmUp(365 * 3)

        # Add universe selection model with fundamental filtering
        self.AddUniverse(self.FundamentalUniverseSelection)
        self.UniverseSettings.Resolution = Resolution.Daily  # Set resolution for universe data

        # Add custom alpha generator
        self.AddAlpha(DualModelAlphaGenerator())

        # Set up portfolio optimizer and construction model
        optimizer = MaximumSharpeRatioPortfolioOptimizer()
        self.SetPortfolioConstruction(
            BlackLittermanOptimizationPortfolioConstructionModel(optimizer=optimizer)
        )

        # Add risk management models
        self.AddRiskManagement(MaximumDrawdownPercentPerSecurity())
        self.AddRiskManagement(TrailingStopRiskManagementModel())

        self.portfolio_targets = []  # List to store portfolio targets
        self.active_stocks = set()   # Set to store active stocks in the universe

        # Schedule universe rebalancing every Monday at midnight
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.At(0, 0),
            self.RebalanceUniverse
        )

    def FundamentalUniverseSelection(self, fundamental):
        """
        Selects stocks based on fundamental data.
        
        - Filters stocks in the energy sector with a positive market cap.
        - Sorts filtered stocks by market capitalization in descending order.
        - Selects the top 20 stocks by market cap.
        
        :param fundamental: List of fundamental data objects
        :return: List of selected stock symbols
        """
        energy_sector_code = MorningstarSectorCode.ENERGY  # Define sector code for energy

        # Filter stocks based on sector and market cap
        filtered = [
            x for x in fundamental
            if x.AssetClassification.MorningstarSectorCode == energy_sector_code and x.MarketCap > 0
        ]

        # Sort filtered stocks by market capitalization
        sorted_by_market_cap = sorted(filtered, key=lambda x: x.MarketCap, reverse=True)
        
        # Return top 20 stocks by market capitalization
        return [x.Symbol for x in sorted_by_market_cap][:20]

    def RebalanceUniverse(self):
        """Rebalances the universe of stocks at the specified schedule."""
        self.UniverseSettings.Rebalance = Resolution.Daily
        self.Debug("Universe rebalanced at: " + str(self.Time))

    def OnSecuritiesChanged(self, changes):
        """
        Handles changes in the securities universe.
        
        - Updates active stocks set based on added securities.
        - Liquidates removed securities from the portfolio.
        - Computes equal weight for each active stock and sets portfolio targets.
        
        :param changes: SecurityChanges object containing added and removed securities
        """
        # Update active stocks based on added securities
        self.active_stocks = {x.Symbol for x in changes.AddedSecurities}

        # Liquidate removed securities
        for x in changes.RemovedSecurities:
            self.Liquidate(x.Symbol)

        # Compute equal weight for each active stock
        if self.active_stocks:
            weight = 1.0 / len(self.active_stocks)
            self.portfolio_targets = [
                PortfolioTarget(symbol, weight) for symbol in self.active_stocks
            ]

    def OnData(self, data):
        """
        Handles incoming data and executes trades based on portfolio targets.
        
        - Skips processing if warming up or if data for all active stocks is not available.
        - Calculates the required trade quantity to achieve target portfolio weights.
        - Executes market orders to adjust holdings based on calculated quantities.
        
        :param data: Slice object containing current market data
        """
        # Skip processing if warming up
        if self.IsWarmingUp:
            return

        # Skip processing if no targets or incomplete data
        if not self.portfolio_targets or not all(symbol in data for symbol in self.active_stocks):
            return

        # Iterate over portfolio targets and adjust holdings
        for target in self.portfolio_targets:
            symbol, target_weight = target.Symbol, target.Quantity

            # Skip if data for symbol is not available
            if not data.ContainsKey(symbol):
                continue

            # Calculate current and target values
            current_price = data[symbol].Price
            current_value = self.Portfolio[symbol].HoldingsValue
            target_value = self.Portfolio.TotalPortfolioValue * target_weight
            quantity = (target_value - current_value) / current_price

            # Execute market orders to adjust holdings
            if quantity > 0 and self.Portfolio.Cash >= quantity * current_price:
                self.MarketOrder(symbol, quantity)
            elif quantity < 0:
                self.MarketOrder(symbol, quantity)

        # Clear portfolio targets after orders are placed
        self.portfolio_targets = []

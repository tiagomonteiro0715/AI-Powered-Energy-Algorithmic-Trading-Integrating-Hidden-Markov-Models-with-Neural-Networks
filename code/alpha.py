"""
This code is made public in accordance with the terms and conditions outlined by QuantConnect.
For more information on these terms, please visit the QuantConnect terms of service page at:
https://www.quantconnect.com/terms/
"""

"""
DISCLAMER: This trading algorithm is provided for research purposes only and
does not constitute financial advice. Trading in financial markets involves
substantial risk and is not suitable for every investor. Past performance is
not indicative of future results. The author assumes no responsibility for any
financial losses or damages incurred as a result of using this software. Use
at your own risk.
"""

import numpy as np  # Import NumPy for numerical operations
from hmmlearn.hmm import GaussianHMM  # Import Gaussian Hidden Markov Model from hmmlearn
import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch's neural network module
import torch.optim as optim  # Import optimization algorithms from PyTorch
from AlgorithmImports import *  # Import necessary classes and methods from QuantConnect
from neural_network import NeuralNetwork  # Import the custom neural network

class DualModelAlphaGenerator(AlphaModel):
    """
    A dual-model alpha generator combining Gaussian Hidden Markov Model (HMM) and a neural network.

    - Utilizes HMM to predict market states based on historical returns.
    - Uses a neural network to generate predictions based on recent price movements.
    - Generates trading insights by combining predictions from both models.
    """

    def __init__(self, lookback=20, hmm_components=5, nn_input_size=5, nn_hidden_size=10, retrain_interval=30):
        """
        Initializes the dual-model alpha generator with specified parameters.

        :param lookback: Number of periods to look back for rolling window data
        :param hmm_components: Number of hidden states in the HMM
        :param nn_input_size: Input size for the neural network
        :param nn_hidden_size: Hidden layer size for the neural network
        :param retrain_interval: Days between retraining the models
        """
        super().__init__()  # Initialize the parent AlphaModel class

        # Model parameters
        self.lookback = lookback
        self.hmm_components = hmm_components
        self.nn_input_size = nn_input_size
        self.nn_hidden_size = nn_hidden_size
        self.retrain_interval = retrain_interval

        # Initialize the models
        self.hmm = GaussianHMM(n_components=self.hmm_components)
        self.nn = NeuralNetwork(nn_input_size, nn_hidden_size, 1)

        # Set up the optimizer and loss function for the neural network
        self.optimizer = optim.Adam(self.nn.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

        # Data storage and training state
        self.data = {}
        self.last_train_time = None

    def add_data(self, symbol, trade_bar):
        """
        Add trade bar data to the rolling window for a given symbol.

        :param symbol: Symbol of the security
        :param trade_bar: TradeBar object containing the latest market data
        """
        try:
            if symbol not in self.data:
                self.data[symbol] = RollingWindow[TradeBar](self.lookback)
            if self.is_valid_trade_bar(trade_bar):
                self.data[symbol].Add(trade_bar)
            else:
                print(f"Invalid trade bar for symbol {symbol}: {trade_bar}")
        except Exception as e:
            print(f"Error adding data for symbol {symbol}: {e}")

    def is_valid_trade_bar(self, trade_bar):
        """
        Check if the trade bar has a valid close price.

        :param trade_bar: TradeBar object
        :return: True if valid, False otherwise
        """
        return trade_bar.Close > 0

    def is_valid_data(self, close_prices):
        """
        Validate if close prices are all positive.

        :param close_prices: Array of close prices
        :return: True if all prices are positive, False otherwise
        """
        return np.all(close_prices > 0)

    def update_models(self, algorithm):
        """
        Retrain models if the retrain interval has passed.

        :param algorithm: The algorithm instance providing the current time
        """
        try:
            if self.last_train_time is None or (algorithm.Time - self.last_train_time).days >= self.retrain_interval:
                self.retrain_models()
                self.last_train_time = algorithm.Time
        except Exception as e:
            print(f"Error updating models: {e}")

    def update(self, algorithm, data):
        """
        Update the model and generate insights based on incoming data.

        :param algorithm: The algorithm instance
        :param data: Slice object containing the current market data
        :return: List of generated insights
        """
        insights = []
        self.update_models(algorithm)

        for symbol in data.Bars.Keys:
            trade_bar = data.Bars[symbol]
            self.add_data(symbol, trade_bar)

            if not self.data[symbol].IsReady:
                continue

            close_prices = np.array([bar.Close for bar in self.data[symbol]])

            if not self.is_valid_data(close_prices):
                continue

            try:
                insights.extend(self.generate_insights(symbol, close_prices))
            except Exception as e:
                print(f"Error generating insights for {symbol}: {e}")

        return insights

    def generate_insights(self, symbol, close_prices):
        """
        Generate insights for a given symbol based on model predictions.

        :param symbol: Symbol of the security
        :param close_prices: Array of close prices
        :return: List of generated insights
        """
        insights = []

        # Calculate log returns from close prices
        returns = self.calculate_returns(close_prices)

        # Predict market states using HMM
        hmm_states = self.hmm_predict_states(returns)

        # Determine the best HMM state with the highest mean return
        best_state = self.get_best_hmm_state(returns, hmm_states)

        # Train the neural network and get the output prediction
        nn_output = self.train_neural_network(close_prices)

        # Determine the direction of the insight
        insight_direction = self.determine_insight_direction(hmm_states, best_state, nn_output, close_prices[-1])

        # Append the generated insight with a confidence level
        insights.append(Insight.Price(symbol, timedelta(days=1), insight_direction, confidence=1))

        return insights

    def calculate_returns(self, close_prices):
        """
        Calculate log returns from close prices.

        :param close_prices: Array of close prices
        :return: Array of log returns
        """
        return np.diff(np.log(close_prices))

    def hmm_predict_states(self, returns):
        """
        Fit HMM and predict states based on returns.

        :param returns: Array of log returns
        :return: Array of predicted HMM states
        """
        try:
            self.hmm.fit(returns.reshape(-1, 1))
            return self.hmm.predict(returns.reshape(-1, 1))
        except Exception as e:
            print(f"Error in HMM state prediction: {e}")
            return np.zeros(len(returns), dtype=int)

    def get_best_hmm_state(self, returns, hmm_states):
        """
        Identify the best HMM state with the highest mean return.

        :param returns: Array of log returns
        :param hmm_states: Array of HMM states
        :return: The state with the highest mean return
        """
        try:
            category_means = [(state, np.mean(returns[hmm_states == state])) for state in np.unique(hmm_states)]
            return max(category_means, key=lambda x: x[1])[0]
        except Exception as e:
            print(f"Error determining best HMM state: {e}")
            return 0

    def train_neural_network(self, close_prices):
        """
        Train the neural network and return the output prediction.

        :param close_prices: Array of close prices
        :return: Output prediction from the neural network
        """
        # Generate features for the neural network
        features = self.get_nn_features(close_prices)
        target = close_prices[-1]

        feature_tensor = torch.tensor(features, dtype=torch.float32)
        target_tensor = torch.tensor([target], dtype=torch.float32)

        try:
            # Zero the gradients, perform forward pass, compute loss, and backpropagate
            self.optimizer.zero_grad()
            nn_output = self.nn(feature_tensor)
            loss = self.loss_function(nn_output, target_tensor)
            loss.backward()
            self.optimizer.step()
            return nn_output.item()
        except Exception as e:
            print(f"Error training neural network: {e}")
            return target  # Default to last known price if training fails

    def get_nn_features(self, close_prices):
        """
        Generate features for the neural network from close prices.

        :param close_prices: Array of close prices
        :return: Array of features for the neural network
        """
        try:
            return np.diff(close_prices[-(self.nn_input_size + 1):])
        except Exception as e:
            print(f"Error generating NN features: {e}")
            return np.zeros(self.nn_input_size)  # Return zeros if there's an error

    def determine_insight_direction(self, hmm_states, best_state, nn_output, last_price):
        """
        Determine the direction of the insight based on model outputs.

        :param hmm_states: Array of HMM states
        :param best_state: Best HMM state with the highest mean return
        :param nn_output: Output from the neural network
        :param last_price: The last known price of the security
        :return: Insight direction (Up, Down, or Flat)
        """
        try:
            if hmm_states[-1] == best_state and nn_output > last_price:
                return InsightDirection.Up
            elif hmm_states[-1] != best_state and nn_output < last_price:
                return InsightDirection.Down
            else:
                return InsightDirection.Flat
        except Exception as e:
            print(f"Error determining insight direction: {e}")
            return InsightDirection.Flat  # Default to flat if an error occurs

    def retrain_models(self):
        """
        Retrain both the HMM and neural network models with current data.
        """
        try:
            for symbol, window in self.data.items():
                if not window.IsReady:
                    continue

                # Extract close prices and calculate returns
                close_prices = np.array([bar.Close for bar in window])
                returns = self.calculate_returns(close_prices)

                # Fit HMM and train the neural network with current data
                self.hmm.fit(returns.reshape(-1, 1))
                self.train_neural_network(close_prices)
        except Exception as e:
            print(f"Error retraining models: {e}")

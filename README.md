# AI-Powered-Energy-Algorithmic-Trading-Integrating-Hidden-Markov-Models-with-Neural-Networks

#### This code is made public in accordance with the terms and conditions outlined by QuantConnect.For more information on these terms, please visit the QuantConnect terms of service page at: https://www.quantconnect.com/terms/

#### DISCLAMER: This trading algorithm is provided for research purposes only and does not constitute financial advice. Trading in financial markets involves substantial risk and is not suitable for every investor. Past performance is not indicative of future results. The author assumes no responsibility for any financial losses or damages incurred as a result of using this software. Use at your own risk.

*Full code and backtest data in the quantconnect plataform to garantee scientific reproduction*: 

- https://www.quantconnect.com/terminal/processCache/?request=embedded_backtest_68f71147c8b5a8948dd3884c33e06c41.html

Article ArXiv: https://arxiv.org/abs/2407.19858 

## Abstract

In quantitative finance, machine learning methods are essential for alpha generation. This study introduces a new approach that combines Hidden Markov Models (HMM) and neural networks, integrated with Black- Litterman  portfolio optimization. During the COVID period (2019-2022), this dual-model approach achieved a 97% return with a Sharpe ratio of 0.992. It incorporates two risk models to enhance risk management, showing efficiency during volatile periods. The methodology was implemented on the QuantConnect platform, which was chosen for its robust framework and experimental reproducibility. The system, which predicts future price movements, includes a three-year warm-up to ensure proper algorithm function. It targets highly liquid, large-cap energy stocks to ensure stable and predictable performance while also considering broker payments. The dual-model alpha system utilizes log returns to select the optimal state based on the historical performance. It combines state predictions with neural network outputs, which are based on historical data, to generate trading signals. This study examined the architecture of the trading system, data pre-processing, training, and performance. The full code and backtesting data are available under the QuantConnect terms.

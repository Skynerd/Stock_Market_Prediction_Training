# Stock Market Prediction Using LSTM

This project implements a Long Short-Term Memory (LSTM) model to predict stock prices based on historical closing prices. The model is trained on Bitcoin's daily closing prices and can be used to forecast future prices.

## Features

- **Data Preprocessing**: Loads and scales the data using MinMaxScaler.
- **Sequence Generation**: Creates input sequences for training the LSTM model.
- **Model Training**: Builds and trains an LSTM model on the training dataset.
- **Prediction**: Makes predictions on the test dataset and evaluates the model's performance.
- **Visualization**: Plots the training data, actual prices, and predicted prices for visual analysis.
- **Model Saving**: Saves the trained model for future predictions.

## Project Structure

- **input/BTC-USD.csv**: Historical Bitcoin price data used for training.
- **output/stock_price_lstm_model.h5**: The saved LSTM model in HDF5 format after training.
- **stock_market_prediction_train.py**: The main Python script that contains the code for data processing, model training, and visualization.

## Prerequisites

To run this project, you'll need to have the following Python libraries installed:

- `tensorflow 2.15.0`
- `pandas 2.2.2`
- `numpy 1.26.2`
- `scikit-learn 1.4.2`
- `matplotlib 3.8.1`

You can install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## How to Use

1. **Clone this repository**:

```bash
git clone https://github.com/yourusername/Stock_Market_Prediction_Train.git
```

2. **Navigate to the project directory**:

```bash
cd Stock_Market_Prediction_Train
```

3. **Ensure you have the input dataset**: Place your historical Bitcoin price data in the `input/` folder as `BTC-USD.csv`.

4. **Run the training script**:

```bash
python stock_market_prediction_train.py
```

5. **View the results**: After training, the script will display the RMSE (Root Mean Squared Error) and plot the actual vs. predicted stock prices.

## Example Output

```
Root Mean Squared Error: 1234.56  # Example RMSE value
```

The plot will display the training dataset, actual closing prices, and predictions made by the LSTM model.

## Visualization

The script generates a plot that includes:

- **Training Data**: Shown in the training blue colour.
- **Actual Prices**: Displayed in the actual orange colour.
- **Predicted Prices**: Shown in the prediction green colour.

![Stock Price Prediction](https://github.com/Skynerd/Stock_Market_Prediction_Training/blob/main/ReadMe/DemoPlot.png)  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

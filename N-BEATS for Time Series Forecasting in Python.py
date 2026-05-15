"""Generated from Jupyter notebook: N-BEATS for Time Series Forecasting in Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import NBEATSModel
from darts.utils.statistics import plot_acf


def main():
    # Generate synthetic time series data
    np.random.seed(42)
    time = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = (
        10
        + 0.5 * np.arange(100)
        + 5 * np.sin(2 * np.pi * np.arange(100) / 30)
        + np.random.normal(scale=2, size=100)
    )
    df = pd.DataFrame({"Date": time, "Value": data})

    # Create a TimeSeries object
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Value")

    # Split the data into train and test sets
    train, test = series.split_before(0.8)

    # Initialize the N-BEATS model
    model = NBEATSModel(
        input_chunk_length=30, output_chunk_length=10, n_epochs=50, random_state=42
    )

    # Fit the model
    model.fit(train)

    # Perform backtesting
    backtest_results = model.backtest(
        series,
        start=0.8,  # Use 80% of the data for the first training
        forecast_horizon=10,
        stride=1,
        retrain=False,  # We've already fitted the model, so we don't need to retrain
        verbose=True,
        metric=[mape, rmse, mae],  # Using multiple metrics
    )

    # Print the backtest results
    print("Backtest Results:")
    print(f"MAPE: {backtest_results[0]:.2f}%")
    print(f"RMSE: {backtest_results[1]:.2f}")
    print(f"MAE: {backtest_results[2]:.2f}")

    # Generate historical forecasts for plotting
    historical_forecasts = model.historical_forecasts(
        series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
    )

    # Plot the results
    plt.figure(figsize=(12, 6))
    series.plot(label="Actual")
    historical_forecasts.plot(label="Forecast")
    plt.title("N-BEATS - Historical Forecasts")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("NBEATS_Backtest.png")
    plt.show()

    # Plot ACF
    plt.figure(figsize=(10, 6))
    plot_acf(series, max_lag=50)
    plt.title("Autocorrelation Function")
    plt.savefig("ACF_plot.png")
    plt.show()


    # --- code cell ---

    # Multivariate forecasting
    # Add a secondary variable (e.g., external temperature)
    temperature = (
        20
        + 2 * np.sin(2 * np.pi * np.arange(100) / 30)
        + np.random.normal(scale=1, size=100)
    )
    df["Temperature"] = temperature

    # Create a multivariate TimeSeries
    multivariate_series = TimeSeries.from_dataframe(
        df, time_col="Date", value_cols=["Value", "Temperature"]
    )

    # Split the multivariate data into train and test sets
    multivariate_train, multivariate_test = multivariate_series.split_before(0.8)

    # Initialize and train the N-BEATS model for multivariate data
    multivariate_model = NBEATSModel(
        input_chunk_length=30, output_chunk_length=10, n_epochs=50, random_state=42
    )
    multivariate_model.fit(multivariate_train)

    # Perform backtesting on multivariate data
    multivariate_backtest_results = multivariate_model.backtest(
        multivariate_series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
        metric=[mape, rmse, mae],
    )

    # Print the multivariate backtest results
    print("\nMultivariate Backtest Results:")
    print(f"MAPE: {multivariate_backtest_results[0]:.2f}%")
    print(f"RMSE: {multivariate_backtest_results[1]:.2f}")
    print(f"MAE: {multivariate_backtest_results[2]:.2f}")

    # Generate historical forecasts for multivariate data
    multivariate_historical_forecasts = multivariate_model.historical_forecasts(
        multivariate_series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
    )

    # Plot results for the primary series in multivariate forecast
    plt.figure(figsize=(12, 6))
    multivariate_series["Value"].plot(label="Actual")
    multivariate_historical_forecasts["Value"].plot(label="Forecast")
    plt.title("N-BEATS Multivariate - Historical Forecasts (Primary Series)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("NBEATS_Multivariate_Backtest.png")
    plt.show()


if __name__ == "__main__":
    main()

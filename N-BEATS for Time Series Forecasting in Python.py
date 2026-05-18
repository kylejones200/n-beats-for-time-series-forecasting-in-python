"""Generated from Jupyter notebook: N-BEATS for Time Series Forecasting in Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import NBEATSModel
from darts.utils.statistics import plot_acf


def generate_synthetic_time_series_data() -> None:
    np.random.seed(42)
    time = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = (
        10
        + 0.5 * np.arange(100)
        + 5 * np.sin(2 * np.pi * np.arange(100) / 30)
        + np.random.normal(scale=2, size=100)
    )
    df = pd.DataFrame({"Date": time, "Value": data})
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Value")
    train, test = series.split_before(0.8)
    model = NBEATSModel(
        input_chunk_length=30, output_chunk_length=10, n_epochs=50, random_state=42
    )
    model.fit(train)
    backtest_results = model.backtest(
        series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
        metric=[mape, rmse, mae],
    )
    print("Backtest Results:")
    print(f"MAPE: {backtest_results[0]:.2f}%")
    print(f"RMSE: {backtest_results[1]:.2f}")
    print(f"MAE: {backtest_results[2]:.2f}")
    historical_forecasts = model.historical_forecasts(
        series, start=0.8, forecast_horizon=10, stride=1, retrain=False, verbose=True
    )
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
    plt.figure(figsize=(10, 6))
    plot_acf(series, max_lag=50)
    plt.title("Autocorrelation Function")
    plt.savefig("ACF_plot.png")
    plt.show()


def multivariate_forecasting() -> None:
    temperature = (
        20
        + 2 * np.sin(2 * np.pi * np.arange(100) / 30)
        + np.random.normal(scale=1, size=100)
    )
    df["Temperature"] = temperature
    multivariate_series = TimeSeries.from_dataframe(
        df, time_col="Date", value_cols=["Value", "Temperature"]
    )
    multivariate_train, multivariate_test = multivariate_series.split_before(0.8)
    multivariate_model = NBEATSModel(
        input_chunk_length=30, output_chunk_length=10, n_epochs=50, random_state=42
    )
    multivariate_model.fit(multivariate_train)
    multivariate_backtest_results = multivariate_model.backtest(
        multivariate_series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
        metric=[mape, rmse, mae],
    )
    print("\nMultivariate Backtest Results:")
    print(f"MAPE: {multivariate_backtest_results[0]:.2f}%")
    print(f"RMSE: {multivariate_backtest_results[1]:.2f}")
    print(f"MAE: {multivariate_backtest_results[2]:.2f}")
    multivariate_historical_forecasts = multivariate_model.historical_forecasts(
        multivariate_series,
        start=0.8,
        forecast_horizon=10,
        stride=1,
        retrain=False,
        verbose=True,
    )
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


def main() -> None:
    generate_synthetic_time_series_data()
    multivariate_forecasting()


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

from sales_forecast_system.features.calendar_features import CalendarFeatureCreator
from sales_forecast_system.features.lag_features import LagFeatureCreator
from sales_forecast_system.features.rolling_features import RollingFeatureCreator


def make_sample_data(periods=143):
    dates = pd.date_range(
        start="2010-02-05",
        periods=periods,
        freq="7D",
    )

    np.random.seed(42)

    return pd.DataFrame({
        "Store": 1,
        "Dept": 1,
        "Date": dates,
        "Weekly_Sales": np.random.normal(20000, 3000, periods),
        "Temperature": np.random.normal(60, 10, periods),
        "Fuel_Price": np.random.normal(3, 0.3, periods),
        "CPI": np.random.normal(200, 5, periods),
        "Unemployment": np.random.normal(7, 1, periods),
        "IsHoliday": 0,
        "Month": dates.month,
        "DayOfWeek": dates.dayofweek,
        "Quarter": dates.quarter,
    })


def test_calendar_features():
    df = make_sample_data()

    creator = CalendarFeatureCreator()
    result = creator.create_all_calendar_features(df)

    assert len(result) == len(df)

    for column in [
        "month_sin",
        "month_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "quarter_sin",
        "quarter_cos",
    ]:
        assert column in result.columns

    for column in [
        "sin_annual_1",
        "cos_annual_1",
        "sin_annual_2",
        "cos_annual_2",
        "sin_annual_3",
        "cos_annual_3",
    ]:
        assert column in result.columns


def test_lag_features():
    df = make_sample_data()

    generator = LagFeatureCreator()

    result = generator.create_lag_features(
        df,
        group_cols=["Store", "Dept"],
    )

    assert len(result) == len(df)

    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        assert f"lag_{lag}" in result.columns

    assert pd.isna(result["lag_1"].iloc[0])
    assert result["lag_1"].iloc[1] == df["Weekly_Sales"].iloc[0]


def test_rolling_features():
    df = make_sample_data()

    generator = RollingFeatureCreator()

    result = generator.create_rolling_features(
        df,
        group_cols=["Store", "Dept"],
    )

    assert len(result) == len(df)

    for window in [4, 8, 12, 26, 52]:
        assert f"rolling_mean_{window}" in result.columns

    assert "rolling_std_4" in result.columns


def test_feature_generation_handles_short_series():
    df = make_sample_data(periods=20)

    generator = LagFeatureCreator()

    result = generator.create_lag_features(
        df,
        group_cols=["Store", "Dept"],
    )

    assert len(result) == 20
    assert "lag_52" in result.columns
    assert result["lag_52"].isna().all()

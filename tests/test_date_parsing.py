from forecasting_service.data import complete_weekly_series, load_sales_data


def test_dayfirst_dates_make_weekly_series():
    df = complete_weekly_series(load_sales_data("data/sales_history.csv"))
    california = df[df["state"] == "California"].sort_values("date")
    gaps = california["date"].diff().dropna().dt.days.unique().tolist()
    assert gaps == [7]
    assert california["date"].min().strftime("%Y-%m-%d") == "2019-10-06"
    assert california["date"].max().strftime("%Y-%m-%d") == "2023-05-07"

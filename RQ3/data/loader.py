
from datetime import datetime
import os
import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

def get_ohlcv(pair, granularity, start, end, api_key=None):
    api_key = api_key or os.getenv("OANDA_API_KEY")
    client = API(access_token=api_key)
    params = {
        "from": datetime.fromisoformat(start).isoformat(),
        "to": datetime.fromisoformat(end).isoformat(),
        "granularity": granularity,
        "price": "M",
    }
    r = InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    candles = r.response["candles"]
    df = pd.DataFrame([
        {
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": c["volume"],
        }
        for c in candles
    ])
    df["time"] = pd.to_datetime(df["time"])
    return df

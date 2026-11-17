from typing import Protocol

import pandas as pd
import yfinance as yf

class  MarketDataProvider(Protocol):
    def fetch(
        self,
        symbol: str, 
        start: str,
        end: str,
        interval: str = "1d",

    )-> pd.DataFrame:
        ...


class YahooFinanceProvider:    

    def fetch(
        self,
        symbol: str, 
        start: str,
        end: str,
        interval: str = "1d",

    )-> pd.DataFrame:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            
        )   
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        if isinstance(df.columns,pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(column).strip()for column in df.columns]     
        if "Date" not in df.columns:

            if "Datetime" in df.columns:
                df = df.rename(
                    columns={"Datetime": "Date"}
                )

            else:
                raise ValueError(
                    "Yahoo Finance response has no Date column"
                )
        return df

import yfinance as yf
from pathlib import Path

folder = Path.home()/"MARK"/"data"/"raw"/"market"
folder.mkdir(parents=True,exist_ok=True)

df = yf.download(

    "^NSEI",
    start="2011-10-01",
    end="2026-10-10",
    interval="1d",
    auto_adjust=False,
    progress=False

)
if df.empty:
    raise RuntimeError("No download check yahoo finance availability")
df.to_csv(folder/"nifty_50_daily.csv")






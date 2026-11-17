import pandas as pd
import matplotlib.pyplot as plt

def plot_price_trend(df: pd.DataFrame, title="Historical closingprice trend"):

    """
    Plot the historical closing-price trend.

    Parameters:
        df: Cleaned market DataFrame

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axes
    """

    required_columns = {"Date", "Close"}

    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain Date and Close columns")

    dates = pd.to_datetime(df["Date"])
    closing_prices = df["Close"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        dates,
        closing_prices,
        label="NIFTY 50 Close",
        linewidth=1
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig, ax

def plot_daily_returns(df: pd.DataFrame, title = "Daily Returns Distribution"):
    """
    plot the daily return of the  stock
    Parameters:
            df: Cleaned market DataFrame
    
        Returns:
            fig: Matplotlib figure
            ax: Matplotlib axes

    """

    close = df["Close"].copy()

    daily_returns = close.pct_change().dropna()*100

    fig,ax = plt.subplots(figsize = (14,6))

    ax.hist (daily_returns, bins = 50, alpha =0.7, color = 'steelblue', edgecolor = 'white')
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Return")
    ax.axvline(daily_returns.mean(), color="orange", linestyle="--",linewidth=1.5,
                label="Mean Return"
)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Daily Return (%)")
    ax.legend()
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()


    return fig , ax

def plot_returns_boxplot(df: pd.DataFrame, title="Daily Returns Distribution"):
    """
    plot the boxplot of the daily return of the stock
    Parameters:
            df: Cleaned market DataFrame
    
        Returns:
            fig: Matplotlib figure
            ax: Matplotlib axes

    """

    close =df["Close"].copy()
    daily_returns = close.pct_change().dropna()*100

    if daily_returns.empty:
        raise ValueError("Daily returns data is empty. Ensure the DataFrame has sufficient data.")

    fig, ax = plt.subplots(figsize=(8, 6))
    flierprops={
    "marker": "o",
    "markersize": 3,
    "markerfacecolor": "none",
    "markeredgecolor": "gray",
    "alpha": 0.5
}

    ax.boxplot(daily_returns,
    patch_artist=True,
    showmeans=True,
    showfliers=True,
    widths=0.5,
    whis=1.5,flierprops=flierprops,)
    ax.grid(True, alpha=0.3)
    

    ax.set_title(title)
    ax.set_ylabel("Daily Return (%)")
    ax.set_xticks([1])
    ax.set_xticklabels(["Daily Returns"])
    return fig, ax



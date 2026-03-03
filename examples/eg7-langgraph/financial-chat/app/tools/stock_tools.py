# =============================================================================
# 股票分析工具函数
# 使用 @tool 装饰器定义，供 LangGraph Agent 调用
# 注意：当前为模拟数据，生产环境应替换为真实数据 API
# =============================================================================

from typing import List

from langchain.agents import tool


# 模拟股票数据（生产环境应替换为真实 API）
MOCK_STOCKS = {
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "price": 178.50,
        "change": 2.3,
        "market_cap": "2.8T",
        "pe_ratio": 28.5,
        "dividend_yield": 0.52,
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "sector": "Technology",
        "price": 141.80,
        "change": -0.5,
        "market_cap": "1.7T",
        "pe_ratio": 24.2,
        "dividend_yield": 0.0,
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "price": 378.90,
        "change": 1.2,
        "market_cap": "2.8T",
        "pe_ratio": 35.1,
        "dividend_yield": 0.74,
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "sector": "Consumer Cyclical",
        "price": 178.25,
        "change": 1.8,
        "market_cap": "1.8T",
        "pe_ratio": 62.3,
        "dividend_yield": 0.0,
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "sector": "Automotive",
        "price": 248.50,
        "change": -2.1,
        "market_cap": "790B",
        "pe_ratio": 72.4,
        "dividend_yield": 0.0,
    },
}


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic information for a given stock symbol."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}. Available symbols: AAPL, GOOGL, MSFT, AMZN, TSLA\n</observation>\n"

    stock = MOCK_STOCKS[symbol]
    return f"\n<observation>\nSymbol: {symbol}\nCompany: {stock['name']}\nSector: {stock['sector']}\nPrice: ${stock['price']}\nChange: {stock['change']}%\nMarket Cap: {stock['market_cap']}\nP/E Ratio: {stock['pe_ratio']}\nDividend Yield: {stock['dividend_yield']}%\n</observation>\n"


@tool
def get_stock_price_history(symbol: str) -> str:
    """Get historical price data for a stock."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    history = """
| Date     | Open  | High  | Low   | Close | Volume   |
|----------|-------|-------|-------|-------|----------|
| 2024-01-01 | 170.00 | 172.50 | 169.00 | 171.25 | 45000000 |
| 2024-01-02 | 171.50 | 173.00 | 170.80 | 172.40 | 42000000 |
| 2024-01-03 | 172.60 | 175.20 | 171.90 | 174.80 | 48000000 |
| 2024-01-04 | 175.00 | 176.50 | 173.20 | 175.60 | 51000000 |
| 2024-01-05 | 175.80 | 178.50 | 174.90 | 178.50 | 55000000 |"""

    return f"\n<observation>\nHistorical prices for {symbol}:\n{history}\n</observation>\n"


@tool
def get_key_metrics(symbol: str) -> str:
    """Get fundamental metrics for a stock."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    metrics = """
| Metric              | Value    |
|---------------------|----------|
| Market Cap          | 2.8T     |
| P/E Ratio           | 28.5     |
| EPS                 | 6.26     |
| Beta                | 1.21     |
| 52-Week High        | 199.62   |
| 52-Week Low         | 124.17   |
| Avg Volume          | 52.4M    |"""

    return f"\n<observation>\nKey Metrics for {symbol}:\n{metrics}\n</observation>\n"


@tool
def get_stock_ratios(symbol: str) -> str:
    """Get financial ratios for a stock."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    ratios = """
| Ratio               | Value    |
|---------------------|----------|
| Current Ratio       | 1.15     |
| Quick Ratio         | 0.98     |
| Debt/Equity         | 1.75     |
| ROE                 | 147.00%  |
| ROA                 | 26.00%   |
| Gross Margin        | 45.00%   |
| Net Profit Margin   | 24.00%   |"""

    return f"\n<observation>\nFinancial Ratios for {symbol}:\n{ratios}\n</observation>\n"


@tool
def get_stock_sector_info(symbol: str) -> str:
    """Get sector information for a stock."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    sector_info = """
| Field          | Value              |
|----------------|-------------------|
| Sector         | Technology        |
| Industry       | Consumer Electronics|
| Website        | apple.com         |
| Description    | Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. |
| Employees      | 164,000           |
| CEO            | Tim Cook          |"""

    return f"\n<observation>\nSector Information for {symbol}:\n{sector_info}\n</observation>\n"


@tool
def get_valuation_multiples(symbol: str) -> str:
    """Get valuation multiples for a stock."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    multiples = """
| Multiple          | Value    |
|-------------------|----------|
| P/E               | 28.5     |
| P/B               | 45.2     |
| P/S               | 7.8      |
| EV/EBITDA         | 22.1     |
| EV/Revenue        | 8.2      |"""

    return f"\n<observation>\nValuation Multiples for {symbol}:\n{multiples}\n</observation>\n"


@tool
def get_gainers() -> str:
    """Get top gaining stocks."""

    gainers = """
| Symbol | Name              | Price  | Change |
|--------|-------------------|--------|--------|
| NVDA   | NVIDIA Corp       | 495.22 | +4.56% |
| AMD    | AMD Inc           | 147.41 | +3.21% |
| META   | Meta Platforms   | 474.99 | +2.89% |"""

    return f"\n<observation>\nTop Gainers:\n{gainers}\n</observation>\n"


@tool
def get_losers() -> str:
    """Get top losing stocks."""

    losers = """
| Symbol | Name              | Price  | Change |
|--------|-------------------|--------|--------|
| PYPL   | PayPal Holdings   | 62.45  | -3.21% |
| SNAP   | Snap Inc          | 11.23  | -2.87% |
| RIVN   | Rivian Automotive | 16.78  | -2.34% |"""

    return f"\n<observation>\nTop Losers:\n{losers}\n</observation>\n"


@tool
def get_stock_universe() -> str:
    """Get a list of stocks in the market universe."""

    universe = """
| Symbol | Name              | Sector            | Market Cap |
|--------|-------------------|-------------------|------------|
| AAPL   | Apple Inc.        | Technology        | 2.8T       |
| MSFT   | Microsoft Corp    | Technology        | 2.8T       |
| GOOGL  | Alphabet Inc.     | Technology        | 1.7T       |
| AMZN   | Amazon.com Inc.   | Consumer Cyclical | 1.8T       |
| TSLA   | Tesla Inc.        | Automotive        | 790B       |"""

    return f"\n<observation>\nStock Universe:\n{universe}\n</observation>\n"


@tool
def get_relative_strength(symbol: str) -> str:
    """Calculate relative strength compared to S&P 500."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    rs_data = """
| Time Period | RS Rating |
|-------------|-----------|
| 1 Month     | 72        |
| 3 Months    | 85        |
| 6 Months    | 68        |
| 1 Year      | 78        |"""

    return f"\n<observation>\nRelative Strength for {symbol} vs S&P 500:\n{rs_data}\n</observation>\n"


@tool
def get_news_sentiment(symbol: str) -> str:
    """Get news sentiment for a stock."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    news = """
| Date       | Title                                    | Sentiment |
|------------|------------------------------------------|-----------|
| 2024-01-05 | Apple announces new product line        | Positive  |
| 2024-01-04 | Quarterly earnings exceed expectations  | Positive  |
| 2024-01-03 | Analyst upgrades to Buy                  | Positive  |
| 2024-01-02 | Supply chain concerns persist            | Negative  |"""

    return f"\n<observation>\nNews and Sentiment for {symbol}:\n{news}\nOverall Sentiment: Positive (0.72)\n</observation>\n"


@tool
def get_stock_chart_analysis(symbol: str) -> str:
    """Generate technical analysis from stock chart."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    analysis = """
Technical Analysis for {symbol}:

Trend: BULLISH
- Price is trading above 20, 50, and 200-day moving averages
- 50-day SMA (170.25) is above 200-day SMA (165.80)
- Short-term momentum is positive

Indicators:
- RSI(14): 62.5 (Neutral, not overbought)
- MACD: Bullish crossover, histogram turning positive
- ATR: 3.45 (moderate volatility)

Support Levels:
- Near-term support: $172.50
- Strong support: $168.00

Resistance Levels:
- First resistance: $180.00
- Major resistance: $185.00

Conclusion: The stock shows bullish technical setup with strong support levels and room for upside movement.""".format(symbol=symbol)

    return f"\n<observation>\n{analysis}\n</observation>\n"


@tool
def calculate_technical_stops(symbol: str) -> str:
    """Calculate stop-loss levels based on technical analysis."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    stops = f"""
Technical Stop Levels for {symbol}:

| Level     | Price   | Description              |
|-----------|---------|--------------------------|
| Aggressive| $170.00 | Recent swing low         |
| Moderate  | $168.00 | 20-day SMA               |
| Conservative| $165.00 | 50-day SMA             |
| Safe      | $160.00 | 200-day SMA + buffer    |

Current Price: $178.50
Recommended Stop: $168.00 (Moderate)
Risk/Reward Ratio: 2.5:1"""

    return f"\n<observation>\n{stops}\n</observation>\n"


@tool
def calculate_r_multiples(symbol: str, entry_price: float, stop_price: float, risk_multiple: int = 2) -> str:
    """Calculate profit targets based on R multiples."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    risk = abs(entry_price - stop_price)
    reward = risk * risk_multiple

    result = f"""
R Multiples for {symbol}:

Entry Price: ${entry_price:.2f}
Stop Price: ${stop_price:.2f}
Risk (1R): ${risk:.2f}

| Target  | Multiple | Price    | Profit  |
|---------|----------|----------|---------|
| Target 1| 1R      | ${entry_price + risk:.2f} | ${risk:.2f} |
| Target 2| 2R      | ${entry_price + (risk * 2):.2f} | ${risk * 2:.2f} |
| Target 3| 3R      | ${entry_price + (risk * 3):.2f} | ${risk * 3:.2f} |
| Target 4| 4R      | ${entry_price + (risk * 4):.2f} | ${risk * 4:.2f} |"""

    return f"\n<observation>\n{result}\n</observation>\n"


@tool
def calculate_position_size(symbol: str, entry_price: float, stop_price: float, account_size: float = 100000.0, risk_percent: float = 1.0) -> str:
    """Calculate optimal position size based on risk management."""

    symbol = symbol.upper()
    if symbol not in MOCK_STOCKS:
        return f"\n<observation>\nNo data found for symbol {symbol}\n</observation>\n"

    risk_amount = account_size * (risk_percent / 100)
    risk_per_share = abs(entry_price - stop_price)
    position_size = int(risk_amount / risk_per_share)
    total_position_value = position_size * entry_price

    result = f"""
Position Sizing for {symbol}:

Account Size: ${account_size:,.2f}
Risk Percentage: {risk_percent}%
Risk Amount: ${risk_amount:,.2f}

Entry Price: ${entry_price:.2f}
Stop Price: ${stop_price:.2f}
Risk Per Share: ${risk_per_share:.2f}

Position Size: {position_size} shares
Total Position Value: ${total_position_value:,.2f}
Potential Loss: ${risk_amount:,.2f} ({risk_percent}%)"""

    return f"\n<observation>\n{result}\n</observation>\n"

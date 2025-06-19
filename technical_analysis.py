import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# NIFTY 50 stock list
nifty_50 = {
    "IT & Tech": {"INFY.NS": "Infosys", "TCS.NS": "TCS", "WIPRO.NS": "Wipro", "HCLTECH.NS": "HCL Tech", "LTIM.NS": "LTIMindtree"},
    "Banking & Finance": {"HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "SBIN.NS": "SBI", "AXISBANK.NS": "Axis Bank", "KOTAKBANK.NS": "Kotak Bank", "BAJFINANCE.NS": "Bajaj Finance"},
    "Energy & Oil": {"RELIANCE.NS": "Reliance", "ONGC.NS": "ONGC", "BPCL.NS": "BPCL", "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid"},
    "Auto": {"TATAMOTORS.NS": "Tata Motors", "M&M.NS": "Mahindra & Mahindra", "EICHERMOT.NS": "Eicher Motors", "MARUTI.NS": "Maruti Suzuki", "BAJAJ-AUTO.NS": "Bajaj Auto"},
    "Consumer & FMCG": {"ITC.NS": "ITC", "HINDUNILVR.NS": "HUL", "NESTLEIND.NS": "Nestle", "TITAN.NS": "Titan", "BRITANNIA.NS": "Britannia"},
    "Pharma": {"SUNPHARMA.NS": "Sun Pharma", "CIPLA.NS": "Cipla", "DRREDDY.NS": "Dr. Reddy's"},
    "Others": {"ADANIENT.NS": "Adani Ent.", "ADANIPORTS.NS": "Adani Ports", "LT.NS": "L&T", "JSWSTEEL.NS": "JSW Steel", "HINDALCO.NS": "Hindalco", "COALINDIA.NS": "Coal India", "ULTRACEMCO.NS": "Ultratech Cement", "GRASIM.NS": "Grasim", "DIVISLAB.NS": "Divi's Lab", "HEROMOTOCO.NS": "Hero Moto"}
}

# Page configuration
st.set_page_config(page_title="Technical Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Navigation
page = st.sidebar.selectbox("Navigate", ["📈 Chart Analysis", "💰 Trading Simulator"])

if page == "📈 Chart Analysis":
    st.title("📈 Technical Analysis Dashboard")
    
    # Sidebar inputs
    st.sidebar.header("Stock Selection")
    category = st.sidebar.selectbox("Select Category", list(nifty_50.keys()))
    company_display = list(nifty_50[category].values())
    company_codes = list(nifty_50[category].keys())
    selected_name = st.sidebar.selectbox("Select Company", company_display)
    ticker = company_codes[company_display.index(selected_name)]
    period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    # Indicator selection
    st.sidebar.header("Technical Indicators")
    show_sma = st.sidebar.checkbox("Simple Moving Average (20, 50)")
    show_ema = st.sidebar.checkbox("Exponential Moving Average (20)")
    show_bollinger = st.sidebar.checkbox("Bollinger Bands")
    show_rsi = st.sidebar.checkbox("RSI", value=True)
    show_macd = st.sidebar.checkbox("MACD")
    show_volume = st.sidebar.checkbox("Volume", value=True)
    
    # Pattern detection settings
    st.sidebar.header("Pattern Detection")
    detect_patterns = st.sidebar.checkbox("Show Trend Change Patterns", value=True)

    @st.cache_data
    def load_data(ticker, period):
        try:
            data = yf.download(ticker, period=period, interval="1d")
            if data.empty:
                return pd.DataFrame()
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            data.dropna(inplace=True)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

    def calculate_bollinger_bands(prices, window=20, num_std=2):
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band, sma

    def detect_trend_change_patterns(df):
        """Detect only significant trend change patterns"""
        patterns = []
        
        for i in range(len(df)):
            if i < 2:  # Need at least 2 previous candles
                patterns.append(None)
                continue
                
            try:
                # Current candle
                o, h, l, c = df.iloc[i][['Open', 'High', 'Low', 'Close']]
                # Previous candles
                o1, h1, l1, c1 = df.iloc[i-1][['Open', 'High', 'Low', 'Close']]
                o2, h2, l2, c2 = df.iloc[i-2][['Open', 'High', 'Low', 'Close']]
                
                body = abs(c - o)
                prev_body = abs(c1 - o1)
                candle_range = h - l
                
                pattern = None
                
                # Only detect significant trend change patterns
                
                # Bullish Engulfing (strong reversal)
                if (c > o) and (c1 < o1) and (o < c1) and (c > o1) and body > prev_body * 1.5:
                    pattern = "Bullish Engulfing"
                
                # Bearish Engulfing (strong reversal)
                elif (c < o) and (c1 > o1) and (o > c1) and (c < o1) and body > prev_body * 1.5:
                    pattern = "Bearish Engulfing"
                
                # Hammer at support levels (bullish reversal)
                elif (body < 0.3 * candle_range and 
                      (l - min(c, o)) > 2 * body and 
                      (h - max(c, o)) < body and
                      c < df['Close'].iloc[i-5:i].min() * 1.02):  # Near recent lows
                    pattern = "Hammer"
                
                # Shooting Star at resistance (bearish reversal)
                elif (body < 0.3 * candle_range and 
                      (h - max(c, o)) > 2 * body and 
                      (min(c, o) - l) < body and
                      c > df['Close'].iloc[i-5:i].max() * 0.98):  # Near recent highs
                    pattern = "Shooting Star"
                
                # Morning Star (3-candle bullish reversal)
                elif (i >= 2 and c2 < o2 and abs(c1 - o1) < 0.1 * (h1 - l1) and 
                      c > o and c > (c2 + o2) / 2):
                    pattern = "Morning Star"
                
                # Evening Star (3-candle bearish reversal)
                elif (i >= 2 and c2 > o2 and abs(c1 - o1) < 0.1 * (h1 - l1) and 
                      c < o and c < (c2 + o2) / 2):
                    pattern = "Evening Star"
                
                patterns.append(pattern)
                
            except:
                patterns.append(None)
        
        return patterns

    # Load and process data
    with st.spinner("Loading data..."):
        df = load_data(ticker, period)

    if df.empty:
        st.error("❌ No data available for the selected stock.")
        st.stop()

    # Calculate indicators
    close_prices = df['Close']
    
    if show_rsi:
        df['RSI'] = calculate_rsi(close_prices)
    
    if show_macd:
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(close_prices)
    
    if show_sma:
        df['SMA_20'] = close_prices.rolling(window=20).mean()
        df['SMA_50'] = close_prices.rolling(window=50).mean()
    
    if show_ema:
        df['EMA_20'] = close_prices.ewm(span=20).mean()
    
    if show_bollinger:
        df['BB_Upper'], df['BB_Lower'], df['BB_Middle'] = calculate_bollinger_bands(close_prices)
    
    if detect_patterns:
        df['Pattern'] = detect_trend_change_patterns(df)

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:.2f}%")
    
    with col2:
        if show_rsi and 'RSI' in df.columns:
            current_rsi = df['RSI'].iloc[-1] if pd.notna(df['RSI'].iloc[-1]) else 50
            rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            st.metric("RSI", f"{current_rsi:.1f}", rsi_status)
    
    with col3:
        day_high = df['High'].iloc[-1]
        day_low = df['Low'].iloc[-1]
        st.metric("Day Range", f"₹{day_low:.2f} - ₹{day_high:.2f}")
    
    with col4:
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = (volume / avg_volume) if avg_volume > 0 else 1
        st.metric("Volume", f"{volume:,.0f}", f"{volume_ratio:.1f}x avg")

    # Create dynamic subplot structure
    subplot_count = 1  # Always have price chart
    subplot_titles = ['Price Chart']
    row_heights = [0.7]
    
    if show_volume:
        subplot_count += 1
        subplot_titles.append('Volume')
        row_heights.append(0.15)
    
    if show_rsi:
        subplot_count += 1
        subplot_titles.append('RSI')
        row_heights.append(0.15)
    
    if show_macd:
        subplot_count += 1
        subplot_titles.append('MACD')
        row_heights.append(0.15)
    
    # Normalize row heights
    total_height = sum(row_heights)
    row_heights = [h/total_height for h in row_heights]

    # Create the chart
    fig = make_subplots(
        rows=subplot_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00D4AA',
            decreasing_line_color='#FF6B6B'
        ),
        row=1, col=1
    )

    # Add selected indicators to price chart
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                                line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                                line=dict(color='blue', width=1)), row=1, col=1)
    
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', 
                                line=dict(color='red', width=1)), row=1, col=1)
    
    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                                line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                                line=dict(color='gray', width=1, dash='dash'), 
                                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    # Add trend change patterns
    if detect_patterns:
        pattern_data = df[df['Pattern'].notna() & (df['Pattern'] != '')]
        if not pattern_data.empty:
            colors = {
                'Bullish Engulfing': 'green',
                'Bearish Engulfing': 'red', 
                'Hammer': 'green',
                'Shooting Star': 'red',
                'Morning Star': 'green',
                'Evening Star': 'red'
            }
            
            for pattern in pattern_data['Pattern'].unique():
                pattern_subset = pattern_data[pattern_data['Pattern'] == pattern]
                fig.add_trace(
                    go.Scatter(
                        x=pattern_subset.index,
                        y=pattern_subset['High'] * 1.01,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up' if 'Bullish' in pattern or pattern in ['Hammer', 'Morning Star'] else 'triangle-down',
                            size=8,
                            color=colors.get(pattern, 'blue')
                        ),
                        name=pattern,
                        hovertemplate=f'{pattern}<br>Date: %{{x}}<br>Price: ₹%{{y:.2f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=1
                )

    current_row = 2

    # Volume chart
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['Volume'], 
                name='Volume', 
                marker_color='lightblue',
                opacity=0.7
            ), 
            row=current_row, col=1
        )
        current_row += 1

    # RSI chart
    if show_rsi:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['RSI'], 
                name='RSI', 
                line=dict(color='purple', width=2)
            ), 
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1, opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1, opacity=0.5)
        fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.1, row=current_row, col=1)
        current_row += 1

    # MACD chart
    if show_macd:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['MACD'], 
                name='MACD', 
                line=dict(color='blue', width=2)
            ), 
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['MACD_Signal'], 
                name='Signal', 
                line=dict(color='red', width=2)
            ), 
            row=current_row, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['MACD_Histogram'], 
                name='Histogram', 
                marker_color='gray',
                opacity=0.6
            ), 
            row=current_row, col=1
        )

    # Update layout
    fig.update_layout(
        height=600 + (subplot_count - 1) * 150,
        title=f"{selected_name} ({ticker}) - Technical Analysis",
        showlegend=True,
        template="plotly_white",
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    # Remove weekend gaps for cleaner look
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    # Pattern summary
    if detect_patterns:
        recent_patterns = df[df['Pattern'].notna() & (df['Pattern'] != '')].tail(10)
        if not recent_patterns.empty:
            st.subheader("Recent Trend Change Patterns")
            pattern_cols = st.columns(len(recent_patterns))
            for i, (date, row) in enumerate(recent_patterns.iterrows()):
                if i < len(pattern_cols):
                    with pattern_cols[i]:
                        st.write(f"**{row['Pattern']}**")
                        st.write(f"{date.strftime('%Y-%m-%d')}")
                        st.write(f"₹{row['Close']:.2f}")

else:  # Trading Simulator Page
    st.title("💰 Trading Simulator")
    st.write("Test your trading strategy with historical data")
    
    # Trading parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Investment Parameters")
        investment_amount = st.slider("Investment Amount (₹)", 10000, 500000, 100000, step=10000)
        risk_appetite = st.selectbox("Risk Appetite", 
                                   ["Conservative", "Moderate", "Aggressive"])
        
    with col2:
        st.subheader("Stock Selection")
        sim_category = st.selectbox("Category", list(nifty_50.keys()), key="sim_cat")
        sim_company_display = list(nifty_50[sim_category].values())
        sim_company_codes = list(nifty_50[sim_category].keys())
        sim_selected_name = st.selectbox("Company", sim_company_display, key="sim_company")
        sim_ticker = sim_company_codes[sim_company_display.index(sim_selected_name)]
    
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running trading simulation..."):
            # Load 1 month data
            sim_data = yf.download(sim_ticker, period="1mo", interval="1d")
            
            if not sim_data.empty:
                if isinstance(sim_data.columns, pd.MultiIndex):
                    sim_data.columns = sim_data.columns.droplevel(1)
                
                # Simple simulation logic based on risk appetite
                risk_multiplier = {"Conservative": 0.02, "Moderate": 0.05, "Aggressive": 0.08}
                risk_per_trade = risk_multiplier[risk_appetite]
                
                initial_price = sim_data['Close'].iloc[0]
                final_price = sim_data['Close'].iloc[-1]
                
                # Simulate basic buy and hold
                shares_bought = investment_amount / initial_price
                final_value = shares_bought * final_price
                profit_loss = final_value - investment_amount
                return_pct = (profit_loss / investment_amount) * 100
                
                # Display results
                st.subheader("Simulation Results")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric("Investment", f"₹{investment_amount:,.0f}")
                    st.metric("Shares Bought", f"{shares_bought:.2f}")
                
                with result_col2:
                    st.metric("Final Value", f"₹{final_value:,.0f}")
                    st.metric("Entry Price", f"₹{initial_price:.2f}")
                
                with result_col3:
                    st.metric("Profit/Loss", f"₹{profit_loss:,.0f}", f"{return_pct:.2f}%")
                    st.metric("Exit Price", f"₹{final_price:.2f}")
                
                # Show the chart
                sim_fig = go.Figure()
                sim_fig.add_trace(
                    go.Candlestick(
                        x=sim_data.index,
                        open=sim_data['Open'],
                        high=sim_data['High'],
                        low=sim_data['Low'],
                        close=sim_data['Close'],
                        name='Price'
                    )
                )
                
                # Mark entry and exit points
                sim_fig.add_trace(
                    go.Scatter(
                        x=[sim_data.index[0]],
                        y=[initial_price],
                        mode='markers',
                        marker=dict(color='green', size=15, symbol='triangle-up'),
                        name='Entry Point'
                    )
                )
                
                sim_fig.add_trace(
                    go.Scatter(
                        x=[sim_data.index[-1]],
                        y=[final_price],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='triangle-down'),
                        name='Exit Point'
                    )
                )
                
                sim_fig.update_layout(
                    title=f"{sim_selected_name} - Trading Simulation (1 Month)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(sim_fig, use_container_width=True)
                
                # Strategy explanation
                st.subheader("Strategy Applied")
                strategy_text = {
                    "Conservative": "Buy and hold strategy with focus on stable stocks. Risk per trade: 2%",
                    "Moderate": "Balanced approach with some tactical entries. Risk per trade: 5%",
                    "Aggressive": "Active trading with higher risk tolerance. Risk per trade: 8%"
                }
                
                st.info(f"**{risk_appetite} Strategy:** {strategy_text[risk_appetite]}")
                
                if return_pct > 0:
                    st.success(f"🎉 Congratulations! You would have made ₹{profit_loss:,.0f} profit!")
                else:
                    st.warning(f"📉 You would have lost ₹{abs(profit_loss):,.0f}. Remember, trading involves risks!")
            
            else:
                st.error("Could not load simulation data. Please try again.")

st.markdown("---")
st.markdown("*Disclaimer: This is for educational purposes only. Past performance does not guarantee future results.*")
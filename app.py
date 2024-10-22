import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Union

def kahanes_algorithm(prices: List[float]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Implements Kahane's algorithm for stock price analysis
    Returns maximum profit and buy-sell points
    """
    n = len(prices)
    if n < 2:
        return 0, []
    
    transactions = []
    i = 0
    
    while i < n-1:
        # Find local minimum
        while i < n-1 and prices[i] >= prices[i+1]:
            i += 1
        if i == n-1:
            break
        buy = i
        
        # Find local maximum
        while i < n-1 and prices[i] <= prices[i+1]:
            i += 1
        sell = i
        
        if prices[sell] > prices[buy]:
            transactions.append((buy, sell))
    
    total_profit = sum(prices[sell] - prices[buy] for buy, sell in transactions)
    return total_profit, transactions

def dp_single_transaction(prices: List[float]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Dynamic programming approach for single transaction
    Returns maximum profit and buy-sell points
    """
    n = len(prices)
    if n < 2:
        return 0, []
    
    min_price_idx = 0
    min_price = prices[0]
    max_profit = 0
    best_transaction = None
    
    for i in range(1, n):
        current_profit = prices[i] - min_price
        if current_profit > max_profit:
            max_profit = current_profit
            best_transaction = [(min_price_idx, i)]
        if prices[i] < min_price:
            min_price = prices[i]
            min_price_idx = i
    
    return max_profit, best_transaction if best_transaction else []

def dp_multiple_transactions(prices: List[float]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Dynamic programming approach for multiple transactions
    Returns maximum profit and buy-sell points
    """
    n = len(prices)
    if n < 2:
        return 0, []
    
    transactions = []
    i = 1
    
    while i < n:
        while i < n and prices[i-1] >= prices[i]:
            i += 1
        if i == n:
            break
        buy = i - 1
        
        while i < n and prices[i-1] <= prices[i]:
            i += 1
        sell = i - 1
        
        if prices[sell] > prices[buy]:
            transactions.append((buy, sell))
    
    total_profit = sum(prices[sell] - prices[buy] for buy, sell in transactions)
    return total_profit, transactions

def greedy_approach(prices: List[float]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Greedy approach for stock trading
    Buy when price is less than next day, sell when price is greater than next day
    """
    n = len(prices)
    if n < 2:
        return 0, []
    
    transactions = []
    for i in range(n-1):
        if prices[i] < prices[i+1]:
            transactions.append((i, i+1))
    
    total_profit = sum(prices[sell] - prices[buy] for buy, sell in transactions)
    return total_profit, transactions

def state_machine(prices: List[float]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    State machine approach for stock trading
    States: HOLDING, NOT_HOLDING
    """
    n = len(prices)
    if n < 2:
        return 0, []
    
    transactions = []
    holding = False
    buy_idx = 0
    
    for i in range(n-1):
        if not holding and prices[i] < prices[i+1]:
            buy_idx = i
            holding = True
        elif holding and prices[i] > prices[i+1]:
            transactions.append((buy_idx, i))
            holding = False
    
    # Check if we need to sell on the last day
    if holding and prices[-1] > prices[buy_idx]:
        transactions.append((buy_idx, n-1))
    
    total_profit = sum(prices[sell] - prices[buy] for buy, sell in transactions)
    return total_profit, transactions

def preprocess_data(df: pd.DataFrame, days: int = 25) -> pd.DataFrame:
    """
    Preprocess the data to get the most recent n days
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=True)
    df = df.tail(days)
    return df.reset_index(drop=True)

def calculate_metrics(prices: List[float], transactions: List[Tuple[int, int]]) -> dict:
    """
    Calculate performance metrics for the trading algorithm
    """
    if not transactions:
        return {
            'total_profit': 0,
            'profit_per_trade': 0,
            'win_rate': 0,
            'avg_holding_period': 0,
            'max_profit_trade': 0,
            'max_loss_trade': 0
        }
    
    profits = [prices[sell] - prices[buy] for buy, sell in transactions]
    holding_periods = [sell - buy for buy, sell in transactions]
    
    return {
        'total_profit': sum(profits),
        'profit_per_trade': sum(profits) / len(transactions),
        'win_rate': sum(1 for p in profits if p > 0) / len(profits) * 100,
        'avg_holding_period': sum(holding_periods) / len(holding_periods),
        'max_profit_trade': max(profits),
        'max_loss_trade': min(profits)
    }

def plot_trades(df: pd.DataFrame, transactions: List[Tuple[int, int]], 
                algorithm_name: str) -> go.Figure:
    """
    Create interactive plot showing stock price and trading points
    """
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Stock Price',
        line=dict(color='blue', width=2)
    ))
    
    if transactions:
        # Add buy points
        buy_dates = [df['Date'].iloc[t[0]] for t in transactions]
        buy_prices = [df['Close'].iloc[t[0]] for t in transactions]
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode='markers',
            name='Buy Points',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
        
        # Add sell points
        sell_dates = [df['Date'].iloc[t[1]] for t in transactions]
        sell_prices = [df['Close'].iloc[t[1]] for t in transactions]
        fig.add_trace(go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            name='Sell Points',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title=f'{algorithm_name} - Trading Points (Last 25 Days)',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        ),
        showlegend=True
    )
    
    return fig

def create_performance_comparison(results: dict) -> go.Figure:
    """
    Create comparison chart for algorithm performance metrics
    """
    algorithms = list(results.keys())
    metrics = ['total_profit', 'profit_per_trade', 'win_rate', 'avg_holding_period']
    
    fig = go.Figure()
    for metric in metrics:
        values = [results[algo][1][metric] for algo in algorithms]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=algorithms,
            y=values
        ))
    
    fig.update_layout(
        title='Algorithm Performance Metrics Comparison',
        barmode='group',
        xaxis_title='Algorithm',
        yaxis_title='Value',
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Stock Market Analysis",
        layout="wide"
    )
    
    st.title('DAA Course Project (Maximizing Stock Market)')
    st.write("""
             
             
             Group 8, TY-CSAI


    This application analyzes the last 25 days of stock market data using various algorithms to maximize profit.
    Upload your CSV file to begin the analysis.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and preprocess data
        df_original = pd.read_csv(uploaded_file)
        df = preprocess_data(df_original, days=25)
        prices = df['Close'].values
        
        st.write(f"Analyzing data from {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Run algorithms
        algorithms = {
            "Kahane's Algorithm": kahanes_algorithm,
            "DP (Single Transaction)": dp_single_transaction,
            "DP (Multiple Transactions)": dp_multiple_transactions,
            "Greedy Approach": greedy_approach,
            "State Machine": state_machine
        }
        
        results = {}
        for algo_name, algo_func in algorithms.items():
            profit, transactions = algo_func(prices)
            metrics = calculate_metrics(prices, transactions)
            results[algo_name] = (profit, metrics, transactions)
        
        # Display stock overview
        st.subheader(' Recent Stock Price Movement')
        fig = px.line(df, x='Date', y='Close', 
                     title='Stock Price - Last 25 Days')
        st.plotly_chart(fig, use_container_width=True)
        
        # Algorithm analysis
        st.subheader(' Algorithm Analysis')
        for algo_name, (profit, metrics, transactions) in results.items():
            with st.expander(f"{algo_name} Analysis"):
                # Metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Profit", f"₹{metrics['total_profit']:.2f}")
                with col2:
                    st.metric("Profit/Trade", f"₹{metrics['profit_per_trade']:.2f}")
                with col3:
                    st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                with col4:
                    st.metric("Avg Hold Period", f"{metrics['avg_holding_period']:.1f}d")
                
                # Trading visualization
                fig = plot_trades(df, transactions, algo_name)
                st.plotly_chart(fig, use_container_width=True)
                
                # Transaction details
                if transactions:
                    st.write("💰 Transaction Details:")
                    for i, (buy, sell) in enumerate(transactions, 1):
                        profit = prices[sell] - prices[buy]
                        st.write(
                            f"Trade {i}: Buy @ ₹{prices[buy]:.2f} ({df['Date'].iloc[buy].date()}), "
                            f"Sell @ ₹{prices[sell]:.2f} ({df['Date'].iloc[sell].date()}), "
                            f"Profit: ₹{profit:.2f}"
                        )
                else:
                    st.write("ℹ No profitable transactions found in the last 25 days.")
        
        # Performance comparison
        st.subheader(' Algorithm Performance Comparison')
        metrics_fig = create_performance_comparison(results)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Volume Analysis')
            if 'Volume' in df.columns:
                vol_fig = px.bar(df, x='Date', y='Volume', 
                                title='Trading Volume - Last 25 Days')
                st.plotly_chart(vol_fig, use_container_width=True)
        
        with col2:
            st.subheader('Price Distribution')
            dist_fig = px.histogram(df, x='Close', 
                                  title='Price Distribution - Last 25 Days',
                                  nbins=20)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        # Statistics
        st.subheader(' Statistical Summary')
        st.write(df[['Close', 'Volume', 'VWAP']].describe())
        
        # Price change analysis
        st.subheader(' Daily Returns Analysis')
        df['Daily_Return'] = df['Close'].pct_change() * 100
        fig = px.line(df, x='Date', y='Daily_Return',
                     title='Daily Price Changes (%)')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
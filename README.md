
# DAA Course Project - Maximizing Stock Market Profit

This repository contains a Streamlit-based web application that analyzes stock market data using multiple algorithms to identify the best buying and selling points for maximizing profit. The application provides an interactive interface to upload stock data, visualize results, and compare performance metrics.

![homepage](UI.png)

## Features

- Analyze stock prices using multiple algorithms:
  - Kahane's Algorithm
  - Dynamic Programming (Single Transaction)
  - Dynamic Programming (Multiple Transactions)
  - Greedy Approach
  - State Machine Approach
- Visualize stock price trends and trading points.
- Compare algorithm performance metrics.
- Upload custom CSV files for analysis.

## Running the Application

1. Clone the repository:

   ```bash
   git clone https://github.com/parthkulkarni04/DAA_CP.git
   cd DAA_CP
   ```

2. Install the required packages using:

```bash
pip install streamlit pandas numpy plotly matplotlib seaborn
```

3. Run the application using Streamlit:

   ```bash
   streamlit run main.py
   ```

4. Upload CSV file containing stock data from the `Archive` folder


## Algorithms

### 1. Kahane's Algorithm

Identifies local minima and maxima to determine profitable buy-sell points.

### 2. Dynamic Programming (Single Transaction)

Finds the most profitable single buy-sell transaction.

### 3. Dynamic Programming (Multiple Transactions)

Allows multiple transactions over the period to maximize profit.

### 4. Greedy Approach

Buys stock if the next day's price is higher and sells if it's lower.

### 5. State Machine Approach

Simulates stock trading using a state machine model.

## Visualizations

The application provides interactive visualizations, including:

- Stock price trends for the last 25 days.
- Buy and sell points on the stock price graph.
- Comparative performance metrics of all implemented algorithms.


## Running the Java Program
To run the program:

1. Compile: 
```bash
javac StockMarketAnalysis.java
```

2. Run (Add the fill path to the CSV you want to analyze)
```bash
java StockMarketAnalysis path/to/your/stock_data.csv
```
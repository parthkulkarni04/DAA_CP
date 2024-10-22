import java.io.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

// Main data model class
class StockData {
    LocalDate date;
    double prevClose;
    double open;
    double high;
    double low;
    double last;
    double close;
    double vwap;
    long volume;
    double turnover;

    public StockData(String[] data) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        this.date = LocalDate.parse(data[0], formatter);
        this.prevClose = Double.parseDouble(data[3]);
        this.open = Double.parseDouble(data[4]);
        this.high = Double.parseDouble(data[5]);
        this.low = Double.parseDouble(data[6]);
        this.last = Double.parseDouble(data[7]);
        this.close = Double.parseDouble(data[8]);
        this.vwap = Double.parseDouble(data[9]);
        this.volume = Long.parseLong(data[10]);
        this.turnover = Double.parseDouble(data[11]);
    }
}

// Trading algorithms
class TradingAlgorithms {
    static class Transaction {
        int buyDay;
        int sellDay;
        double profit;

        Transaction(int buyDay, int sellDay, double profit) {
            this.buyDay = buyDay;
            this.sellDay = sellDay;
            this.profit = profit;
        }
    }

    static class Result {
        double totalProfit;
        List<Transaction> transactions;

        Result(double totalProfit, List<Transaction> transactions) {
            this.totalProfit = totalProfit;
            this.transactions = transactions;
        }
    }

    // Kahane's Algorithm
    public static Result kahanesAlgorithm(double[] prices) {
        List<Transaction> transactions = new ArrayList<>();
        int i = 0;
        double totalProfit = 0;

        while (i < prices.length - 1) {
            // Find local minimum
            while (i < prices.length - 1 && prices[i] >= prices[i + 1]) {
                i++;
            }
            if (i == prices.length - 1) break;
            int buy = i;

            // Find local maximum
            while (i < prices.length - 1 && prices[i] <= prices[i + 1]) {
                i++;
            }
            int sell = i;

            if (prices[sell] > prices[buy]) {
                double profit = prices[sell] - prices[buy];
                transactions.add(new Transaction(buy, sell, profit));
                totalProfit += profit;
            }
        }

        return new Result(totalProfit, transactions);
    }

    // Dynamic Programming - Single Transaction
    public static Result dpSingleTransaction(double[] prices) {
        if (prices.length < 2) return new Result(0, new ArrayList<>());

        int minPriceIdx = 0;
        double minPrice = prices[0];
        double maxProfit = 0;
        int bestBuyIdx = 0;
        int bestSellIdx = 0;

        for (int i = 1; i < prices.length; i++) {
            double currentProfit = prices[i] - minPrice;
            if (currentProfit > maxProfit) {
                maxProfit = currentProfit;
                bestBuyIdx = minPriceIdx;
                bestSellIdx = i;
            }
            if (prices[i] < minPrice) {
                minPrice = prices[i];
                minPriceIdx = i;
            }
        }

        List<Transaction> transactions = new ArrayList<>();
        if (maxProfit > 0) {
            transactions.add(new Transaction(bestBuyIdx, bestSellIdx, maxProfit));
        }

        return new Result(maxProfit, transactions);
    }

    // Dynamic Programming - Multiple Transactions
    public static Result dpMultipleTransactions(double[] prices) {
        List<Transaction> transactions = new ArrayList<>();
        double totalProfit = 0;
        
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                double profit = prices[i] - prices[i - 1];
                transactions.add(new Transaction(i - 1, i, profit));
                totalProfit += profit;
            }
        }

        return new Result(totalProfit, transactions);
    }
}

// Metrics calculator
class MetricsCalculator {
    static class Metrics {
        double totalProfit;
        double profitPerTrade;
        double winRate;
        double avgHoldingPeriod;
        double maxProfitTrade;
        double maxLossTrade;

        Metrics(List<TradingAlgorithms.Transaction> transactions) {
            if (transactions.isEmpty()) {
                return;
            }

            this.totalProfit = transactions.stream()
                .mapToDouble(t -> t.profit)
                .sum();

            this.profitPerTrade = totalProfit / transactions.size();

            long winningTrades = transactions.stream()
                .filter(t -> t.profit > 0)
                .count();
            this.winRate = (double) winningTrades / transactions.size() * 100;

            this.avgHoldingPeriod = transactions.stream()
                .mapToDouble(t -> t.sellDay - t.buyDay)
                .average()
                .orElse(0);

            this.maxProfitTrade = transactions.stream()
                .mapToDouble(t -> t.profit)
                .max()
                .orElse(0);

            this.maxLossTrade = transactions.stream()
                .mapToDouble(t -> t.profit)
                .min()
                .orElse(0);
        }
    }
}

// Main class
public class StockMarketAnalysis {
    private static List<StockData> readCsvFile(String filePath) throws IOException {
        List<StockData> stockDataList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line = reader.readLine(); // Skip header

        while ((line = reader.readLine()) != null) {
            String[] data = line.split(",");
            try {
                stockDataList.add(new StockData(data));
            } catch (Exception e) {
                System.err.println("Error parsing line: " + line);
            }
        }
        reader.close();
        return stockDataList;
    }

    private static void analyzeStock(List<StockData> stockDataList) {
        // Get closing prices
        double[] prices = stockDataList.stream()
            .mapToDouble(data -> data.close)
            .toArray();

        // Run algorithms
        System.out.println("\n=== Stock Market Analysis Results ===\n");

        // Kahane's Algorithm
        TradingAlgorithms.Result kahanesResult = TradingAlgorithms.kahanesAlgorithm(prices);
        printResults("Kahane's Algorithm", kahanesResult, stockDataList);

        // DP Single Transaction
        TradingAlgorithms.Result dpSingleResult = TradingAlgorithms.dpSingleTransaction(prices);
        printResults("DP Single Transaction", dpSingleResult, stockDataList);

        // DP Multiple Transactions
        TradingAlgorithms.Result dpMultipleResult = TradingAlgorithms.dpMultipleTransactions(prices);
        printResults("DP Multiple Transactions", dpMultipleResult, stockDataList);
    }

    private static void printResults(String algorithmName, TradingAlgorithms.Result result, 
                                   List<StockData> stockDataList) {
        System.out.println("=== " + algorithmName + " ===");
        System.out.printf("Total Profit: ₹%.2f%n", result.totalProfit);

        MetricsCalculator.Metrics metrics = new MetricsCalculator.Metrics(result.transactions);
        System.out.printf("Profit per Trade: ₹%.2f%n", metrics.profitPerTrade);
        System.out.printf("Win Rate: %.1f%%%n", metrics.winRate);
        System.out.printf("Average Holding Period: %.1f days%n", metrics.avgHoldingPeriod);

        System.out.println("\nTransaction Details:");
        for (int i = 0; i < result.transactions.size(); i++) {
            TradingAlgorithms.Transaction t = result.transactions.get(i);
            StockData buyDay = stockDataList.get(t.buyDay);
            StockData sellDay = stockDataList.get(t.sellDay);
            
            System.out.printf("Trade %d: Buy @ ₹%.2f (%s), Sell @ ₹%.2f (%s), Profit: ₹%.2f%n",
                i + 1, buyDay.close, buyDay.date, sellDay.close, sellDay.date, t.profit);
        }
        System.out.println();
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Please provide the path to the CSV file as an argument.");
            return;
        }

        try {
            List<StockData> stockDataList = readCsvFile(args[0]);
            System.out.println("Analyzing data from " + 
                stockDataList.get(0).date + " to " + 
                stockDataList.get(stockDataList.size() - 1).date);
            
            analyzeStock(stockDataList);
        } catch (IOException e) {
            System.err.println("Error reading CSV file: " + e.getMessage());
        }
    }
}
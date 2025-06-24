import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import os
import scipy.stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def detect_time_columns(data):
    """
    Automatically detect time/date columns based on common naming patterns
    """
    time_keywords = ['time', 'Time', 'TIME', 'date', 'Date', 'DATE', 
                     'timestamp', 'Timestamp', 'TIMESTAMP',
                     'time[s]', 'Time[s]', 'TIME[S]']
    
    time_columns = []
    for col in data.columns:
        # Check exact matches
        if col in time_keywords:
            time_columns.append(col)
        # Check if column contains time keywords
        elif any(keyword.lower() in col.lower() for keyword in ['time', 'date', 'timestamp']):
            time_columns.append(col)
        # Check if column is datetime type
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            time_columns.append(col)
    
    return time_columns

def clean_data(data):
    """
    Basic data cleaning: drop NA rows and remove time columns
    """
    print("Performing data cleaning...")
    
    # Detect and remove time columns
    time_columns = detect_time_columns(data)
    if time_columns:
        print(f"Detected time columns: {', '.join(time_columns)}")
        data_cleaned = data.drop(columns=time_columns)
    else:
        data_cleaned = data.copy()
        time_columns = []
    
    # Drop rows with any NA values
    initial_rows = len(data_cleaned)
    data_cleaned = data_cleaned.dropna()
    rows_dropped = initial_rows - len(data_cleaned)
    
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing NA values")
    
    print(f"Final dataset: {len(data_cleaned)} rows, {len(data_cleaned.columns)} columns")
    
    return data_cleaned, time_columns

def check_granger_causality(x, y, max_lag=2, f_stat_threshold=10, p_value_threshold=0.05):
    """
    Test if x Granger-causes y.
    Returns:
    - is_causal: Boolean indicating if x causes y
    - f_stat: F-statistic
    - p_value: p-value
    - lag: Optimal lag
    """
    try:
        # Test with specified max_lag
        test_result = grangercausalitytests(np.column_stack((y, x)), maxlag=max_lag, verbose=False)
        
        # Extract results for each lag
        f_stats = []
        p_values = []
        
        for lag in range(1, max_lag + 1):
            f_stat = test_result[lag][0]['ssr_ftest'][0]
            p_value = test_result[lag][0]['ssr_ftest'][1]
            f_stats.append(f_stat)
            p_values.append(p_value)
            
        # Find the highest F-statistic among all lags
        max_f_index = f_stats.index(max(f_stats))
        best_f = f_stats[max_f_index]
        best_p = p_values[max_f_index]
        best_lag = max_f_index + 1
        
        # Check if the best result meets our criteria
        is_causal = (best_f > f_stat_threshold) and (best_p < p_value_threshold)
        
        return is_causal, best_f, best_p, best_lag
    except:
        # Return False if there's an error (e.g., non-stationary data)
        return False, 0, 1, 0

def visualize_network(G, output_dir):
    """
    Create a structured network visualization with eigenvector centrality coloring
    """
    plt.figure(figsize=(16, 12))
    
    # Calculate eigenvector centrality for coloring
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        # Fallback to degree centrality if eigenvector fails
        eigenvector_centrality = nx.degree_centrality(G)
    
    # Create structured layout with more spacing
    pos = nx.spring_layout(G, k=3.0, iterations=500, seed=42)
    
    # Prepare node colors based on eigenvector centrality
    node_colors = [eigenvector_centrality[node] for node in G.nodes()]
    
    # Node sizes based on total degree (in + out)
    node_sizes = [300 + 1000 * G.degree(node) / max([G.degree(n) for n in G.nodes()]) 
                  for node in G.nodes()]
    
    # Draw nodes with eigenvector centrality coloring
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_sizes, 
                                  node_color=node_colors,
                                  cmap=plt.cm.viridis,
                                  alpha=0.8)
    
    # Separate edges by correlation type
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('correlation', 0) > 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('correlation', 0) < 0]
    
    # Draw positive edges in blue
    if positive_edges:
        nx.draw_networkx_edges(G, pos, edgelist=positive_edges, 
                              edge_color='blue', width=2, alpha=0.7,
                              arrows=True, arrowstyle='->', arrowsize=20,
                              connectionstyle='arc3,rad=0.1')
    
    # Draw negative edges in red
    if negative_edges:
        nx.draw_networkx_edges(G, pos, edgelist=negative_edges,
                              edge_color='red', width=2, alpha=0.7,
                              arrows=True, arrowstyle='->', arrowsize=20,
                              connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')
    
    # Add colorbar for eigenvector centrality
    if nodes:
        plt.colorbar(nodes, label='Eigenvector Centrality', shrink=0.8)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Positive Causal Effect'),
        Line2D([0], [0], color='red', lw=2, label='Negative Causal Effect')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title('Causal Network\n(Node color = Eigenvector Centrality, Node size = Total Degree)', 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_network.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Network visualization saved to {os.path.join(output_dir, 'causal_network.png')}")

def create_centrality_csv(G, output_dir):
    """
    Create CSV with centrality measures for each node
    """
    if G.number_of_nodes() == 0:
        print("No nodes in graph. Skipping centrality CSV.")
        return None
    
    # Calculate centrality measures
    out_degree = dict(G.out_degree())
    in_degree = dict(G.in_degree())
    
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
    except:
        betweenness_centrality = {node: 0 for node in G.nodes()}
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        # Fallback to degree centrality if eigenvector fails
        eigenvector_centrality = nx.degree_centrality(G)
    
    # Create DataFrame
    centrality_data = []
    for node in G.nodes():
        centrality_data.append({
            'Node': node,
            'Out_Degree': out_degree[node],
            'In_Degree': in_degree[node],
            'Betweenness_Centrality': betweenness_centrality[node],
            'Eigenvector_Centrality': eigenvector_centrality[node]
        })
    
    centrality_df = pd.DataFrame(centrality_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'node_centralities.csv')
    centrality_df.to_csv(csv_path, index=False)
    print(f"Node centralities saved to {csv_path}")
    
    return centrality_df

def print_hub_analysis(G):
    """
    Print top 5 nodes for each hub type
    """
    if G.number_of_nodes() == 0:
        print("No nodes in graph for hub analysis.")
        return
    
    # Calculate centrality measures
    out_degree = dict(G.out_degree())
    in_degree = dict(G.in_degree())
    
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
    except:
        betweenness_centrality = {node: 0 for node in G.nodes()}
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_centrality = nx.degree_centrality(G)
    
    print("\n" + "="*60)
    print("HUB ANALYSIS")
    print("="*60)
    
    # Causal Influence Hubs (high out-degree)
    print("\nTop 5 Causal Influence Hubs (High Out-Degree):")
    sorted_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)
    for i, (node, degree) in enumerate(sorted_out[:5]):
        if degree > 0:
            print(f"  {i+1}. {node}: {degree} outgoing connections")
    
    # Vulnerability Hubs (high in-degree)
    print("\nTop 5 Vulnerability Hubs (High In-Degree):")
    sorted_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)
    for i, (node, degree) in enumerate(sorted_in[:5]):
        if degree > 0:
            print(f"  {i+1}. {node}: {degree} incoming connections")
    
    # Bridge Hubs (high betweenness centrality)
    print("\nTop 5 Bridge Hubs (High Betweenness Centrality):")
    sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    for i, (node, centrality) in enumerate(sorted_betweenness[:5]):
        if centrality > 0:
            print(f"  {i+1}. {node}: {centrality:.4f}")
    
    # Importance Hubs (high eigenvector centrality)
    print("\nTop 5 Importance Hubs (High Eigenvector Centrality):")
    sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
    for i, (node, centrality) in enumerate(sorted_eigenvector[:5]):
        if centrality > 0:
            print(f"  {i+1}. {node}: {centrality:.4f}")

def Decypher(data, exclude_cols=None, corr_threshold=0.6, f_stat_threshold=10, 
            p_value_threshold=0.05, max_lag=2, output_dir=None):
    """
    Simplified Decypher function for causal network analysis
    
    Parameters:
    - data: DataFrame with time series data
    - exclude_cols: List of column names to exclude (in addition to auto-detected time columns)
    - corr_threshold: Minimum absolute correlation to consider for causality testing
    - f_stat_threshold: Minimum F-statistic for significance
    - p_value_threshold: Maximum p-value for significance
    - max_lag: Maximum lag to test for causality
    - output_dir: Directory to save visualizations
    
    Returns:
    - G: NetworkX DiGraph of causal relationships
    - causal_df: DataFrame with all causal relationships found
    - centrality_df: DataFrame with node centrality measures
    """
    # Set output directory
    if output_dir is None:
        output_dir = '.'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clean data (remove time columns and NA rows)
    data_cleaned, time_columns = clean_data(data)
    
    # Handle additional column exclusions
    if exclude_cols:
        if isinstance(exclude_cols, str):
            exclude_cols = [exclude_cols]
        excluded_additional = [col for col in exclude_cols if col in data_cleaned.columns]
        if excluded_additional:
            data_cleaned = data_cleaned.drop(columns=excluded_additional)
            print(f"Additionally excluded columns: {', '.join(excluded_additional)}")
    
    if data_cleaned.empty or len(data_cleaned.columns) < 2:
        print("Error: Insufficient data after cleaning. Need at least 2 columns.")
        return None, None, None
    
    print(f"Analyzing {len(data_cleaned.columns)} variables: {', '.join(data_cleaned.columns)}")
    
    # Calculate correlation matrix
    correlation_matrix = data_cleaned.corr(method='pearson')
    
    # Create a directed network graph
    G = nx.DiGraph()
    
    # Add nodes
    for column in data_cleaned.columns:
        G.add_node(column)
    
    # Test for causal relationships
    print("Testing causal relationships using Granger causality...")
    causal_edges = 0
    
    for i, col1 in enumerate(data_cleaned.columns):
        for j, col2 in enumerate(data_cleaned.columns):
            if i != j:  # Don't test self-causality
                corr_value = correlation_matrix.loc[col1, col2]
                # Only test causality if correlation meets threshold
                if abs(corr_value) >= corr_threshold:
                    # Check causality
                    is_causal, f_stat, p_value, lag = check_granger_causality(
                        data_cleaned[col1], data_cleaned[col2], 
                        max_lag=max_lag, 
                        f_stat_threshold=f_stat_threshold, 
                        p_value_threshold=p_value_threshold
                    )
                    
                    # Add edge if causality is detected
                    if is_causal:
                        G.add_edge(col1, col2, 
                                  weight=abs(corr_value),
                                  correlation=corr_value,
                                  f_stat=f_stat,
                                  p_value=p_value,
                                  lag=lag)
                        causal_edges += 1
    
    print(f"Found {causal_edges} causal relationships")
    
    # Remove nodes without connections
    nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)
    
    if nodes_to_remove:
        print(f"Removed {len(nodes_to_remove)} nodes without causal connections")
    
    # Create visualizations and analysis
    if G.number_of_nodes() > 0:
        # Network visualization
        visualize_network(G, output_dir)
        
        # Create centrality CSV
        centrality_df = create_centrality_csv(G, output_dir)
        
        # Print hub analysis
        print_hub_analysis(G)
        
        # Create causal relationships DataFrame
        causal_relationships = []
        for u, v, data in G.edges(data=True):
            causal_relationships.append({
                'Cause': u,
                'Effect': v,
                'Correlation': data['correlation'],
                'F_Statistic': data['f_stat'],
                'P_Value': data['p_value'],
                'Optimal_Lag': data['lag']
            })
        
        causal_df = pd.DataFrame(causal_relationships)
        causal_df = causal_df.sort_values('F_Statistic', ascending=False)
        
        # Save causal relationships
        causal_path = os.path.join(output_dir, 'causal_relationships.csv')
        causal_df.to_csv(causal_path, index=False)
        print(f"Causal relationships saved to {causal_path}")
        
        # Print network summary
        print(f"\nNetwork Summary:")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        
    else:
        print("No causal relationships found meeting the criteria.")
        causal_df = pd.DataFrame()
        centrality_df = pd.DataFrame()
    
    return G, causal_df, centrality_df

def query_causal_influences(target_variable, causal_graph, top_n=5):
    """
    Find the top N factors that causally influence a target variable
    
    Parameters:
    - target_variable: Name of the target variable
    - causal_graph: NetworkX DiGraph from Decypher analysis
    - top_n: Number of top influences to return
    
    Returns:
    - List of tuples: (cause, f_statistic)
    """
    if causal_graph is None or causal_graph.number_of_edges() == 0:
        print("No causal graph available. Run Decypher analysis first.")
        return []
    
    influences = []
    
    # Find all edges pointing to the target variable
    for cause, effect, data in causal_graph.edges(data=True):
        if effect.lower() == target_variable.lower():
            influences.append((cause, data.get('f_stat', 0)))
    
    if not influences:
        print(f"No causal influences found for '{target_variable}'")
        print(f"Available variables: {list(causal_graph.nodes())}")
        return []
    
    # Sort by F-statistic strength
    influences.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {min(top_n, len(influences))} factors causally influencing '{target_variable}':")
    print("-" * 50)
    for i, (cause, f_stat) in enumerate(influences[:top_n]):
        print(f"{i+1:2d}. {cause:<25} F-statistic: {f_stat:.2f}")
    
    return influences[:top_n]

def plot_bollinger_bands(data, metric_name, time_column=None, window=20, std_dev=1.5, output_dir='.'):
    """
    Plot Bollinger Bands for a specific metric
    
    Parameters:
    - data: Original DataFrame (before cleaning)
    - metric_name: Name of the metric to plot
    - time_column: Name of time column (auto-detected if None)
    - window: Rolling window for moving average (default: 20)
    - std_dev: Standard deviations for bands (default: 1.5)
    - output_dir: Directory to save plot
    """
    # Auto-detect time column if not specified
    if time_column is None:
        time_columns = detect_time_columns(data)
        if time_columns:
            time_column = time_columns[0]
        else:
            print("No time column found. Cannot create Bollinger Bands.")
            return
    
    # Check if metric exists
    if metric_name not in data.columns:
        print(f"Metric '{metric_name}' not found in data.")
        print(f"Available metrics: {list(data.columns)}")
        return
    
    # Prepare data
    df = data[[time_column, metric_name]].copy().dropna()
    
    # Convert time column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except:
            print(f"Could not convert {time_column} to datetime format")
            return
    
    # Sort by time
    df = df.sort_values(time_column)
    
    # Calculate Bollinger Bands
    df['Moving_Average'] = df[metric_name].rolling(window=window).mean()
    df['Upper_Band'] = df['Moving_Average'] + (df[metric_name].rolling(window=window).std() * std_dev)
    df['Lower_Band'] = df['Moving_Average'] - (df[metric_name].rolling(window=window).std() * std_dev)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot the metric
    plt.plot(df[time_column], df[metric_name], label=metric_name, color='green', linewidth=1.5)
    
    # Plot moving average
    plt.plot(df[time_column], df['Moving_Average'], label='Moving Average', color='blue', linewidth=2)
    
    # Plot Bollinger Bands
    plt.plot(df[time_column], df['Upper_Band'], label='Upper Band', color='red', linestyle='--', alpha=0.7)
    plt.plot(df[time_column], df['Lower_Band'], label='Lower Band', color='red', linestyle='--', alpha=0.7)
    
    # Fill between bands
    plt.fill_between(df[time_column], df['Lower_Band'], df['Upper_Band'], alpha=0.1, color='gray')
    
    plt.title(f'Bollinger Bands for {metric_name}\n({window}-period moving average, {std_dev} std dev)')
    plt.xlabel('Time')
    plt.ylabel(metric_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{metric_name}_bollinger_bands.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Bollinger Bands plot saved to {plot_path}")

def forecast_metric(data, metric_name, forecast_days=7, history_days=30, time_column=None, output_dir='.'):
    """
    Forecast future values for a metric using ARIMA
    
    Parameters:
    - data: Original DataFrame (before cleaning)
    - metric_name: Name of the metric to forecast
    - forecast_days: Number of days to forecast into the future
    - history_days: Number of recent days to use for training
    - time_column: Name of time column (auto-detected if None)
    - output_dir: Directory to save plot
    
    Returns:
    - forecast_df: DataFrame with forecasted values
    """
    # Auto-detect time column if not specified
    if time_column is None:
        time_columns = detect_time_columns(data)
        if time_columns:
            time_column = time_columns[0]
        else:
            print("No time column found. Cannot create forecast.")
            return None
    
    # Check if metric exists
    if metric_name not in data.columns:
        print(f"Metric '{metric_name}' not found in data.")
        print(f"Available metrics: {list(data.columns)}")
        return None
    
    # Prepare data
    df = data[[time_column, metric_name]].copy().dropna()
    
    # Convert time column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        try:
            df[time_column] = pd.to_datetime(df[time_column])
        except:
            print(f"Could not convert {time_column} to datetime format")
            return None
    
    # Sort by time and get recent data
    df = df.sort_values(time_column)
    recent_data = df.tail(history_days)
    
    if len(recent_data) < 10:
        print(f"Insufficient data for forecasting. Need at least 10 points, got {len(recent_data)}")
        return None
    
    # Fit ARIMA model
    try:
        # Try different ARIMA parameters and select best one
        best_aic = float('inf')
        best_model = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(recent_data[metric_name], order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is None:
            print("Could not fit ARIMA model. Using simple linear trend instead.")
            # Fallback to simple linear regression
            x = np.arange(len(recent_data))
            y = recent_data[metric_name].values
            coeffs = np.polyfit(x, y, 1)
            
            # Generate future dates
            last_date = recent_data[time_column].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            
            # Simple linear forecast
            future_x = np.arange(len(recent_data), len(recent_data) + forecast_days)
            forecast_values = np.polyval(coeffs, future_x)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast_values,
                'Lower_CI': forecast_values - np.std(y),
                'Upper_CI': forecast_values + np.std(y)
            })
        else:
            # Use ARIMA forecast
            forecast = best_model.forecast(steps=forecast_days)
            forecast_ci = best_model.get_forecast(steps=forecast_days).conf_int()
            
            # Generate future dates
            last_date = recent_data[time_column].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=forecast_days, freq='D')
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast,
                'Lower_CI': forecast_ci.iloc[:, 0],
                'Upper_CI': forecast_ci.iloc[:, 1]
            })
    
    except Exception as e:
        print(f"Error in forecasting: {e}")
        return None
    
    # Create forecast plot
    plt.figure(figsize=(14, 8))
    
    # Plot historical data
    plt.plot(recent_data[time_column], recent_data[metric_name], 
             label='Historical Data', color='blue', linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], 
             label='Forecast', color='red', linewidth=2, linestyle='--')
    
    # Plot confidence interval
    plt.fill_between(forecast_df['Date'], 
                     forecast_df['Lower_CI'], 
                     forecast_df['Upper_CI'], 
                     alpha=0.3, color='red', label='Confidence Interval')
    
    plt.title(f'{forecast_days}-Day Forecast for {metric_name}\n(Based on last {history_days} days)')
    plt.xlabel('Date')
    plt.ylabel(metric_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{metric_name}_forecast.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Forecast plot saved to {plot_path}")
    
    # Save forecast data
    forecast_path = os.path.join(output_dir, f'{metric_name}_forecast.csv')
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecast data saved to {forecast_path}")
    
    return forecast_df

# Query what influences a specific metric
G, causal_df, centrality_df = Decypher(data, output_dir=None)
query_causal_influences('metric', G, top_n=5)
plot_bollinger_bands(data, 'SmO2', std_dev=1.5)
forecast_df = forecast_metric(data, 'SmO2', forecast_days=7, history_days=30)

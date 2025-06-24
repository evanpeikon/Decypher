import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
import os
import scipy.stats
import plotly.graph_objects as go
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def check_multivariate_granger_causality(data, target, predictors, max_lag=2, f_stat_threshold=10, p_value_threshold=0.05):
    """
    Test if multiple variables jointly Granger-cause a target variable.
    
    Parameters:
    - data: DataFrame containing all variables
    - target: Target variable name
    - predictors: List of predictor variable names
    - max_lag: Maximum lag to test
    
    Returns:
    - is_causal: Boolean indicating if predictors cause target
    - f_stat: F-statistic
    - p_value: p-value
    - lag: Optimal lag
    """
    best_f = 0
    best_p = 1
    best_lag = 0
    is_causal = False
    
    try:
        # Test each lag from 1 to max_lag
        for lag in range(1, max_lag + 1):
            # Create lagged predictors
            X = pd.DataFrame()
            y = data[target][lag:]
            
            # Add lagged values for each predictor
            for pred in predictors:
                for l in range(1, lag + 1):
                    X[f"{pred}_lag{l}"] = data[pred].shift(l)[lag:]
            
            # Also include lagged values of the target variable
            for l in range(1, lag + 1):
                X[f"{target}_lag{l}"] = data[target].shift(l)[lag:]
            
            # Remove any NaN rows
            X = X.dropna()
            y = y[:len(X)]
            
            # Create two models: one with and one without the predictor variables
            restricted_cols = [col for col in X.columns if col.startswith(f"{target}_lag")]
            if len(restricted_cols) > 0:  # Only proceed if there are target lags
                model_restricted = sm.OLS(y, X[restricted_cols]).fit()
                model_unrestricted = sm.OLS(y, X).fit()
                
                # F-test for nested models
                df1 = model_restricted.df_resid - model_unrestricted.df_resid
                df2 = model_unrestricted.df_resid
                
                if df1 > 0 and df2 > 0:  # Only calculate if valid degrees of freedom
                    f_stat = ((model_restricted.ssr - model_unrestricted.ssr) / df1) / \
                            (model_unrestricted.ssr / df2)
                    
                    p_value = 1 - scipy.stats.f.cdf(f_stat, df1, df2)
                    
                    if f_stat > best_f:
                        best_f = f_stat
                        best_p = p_value
                        best_lag = lag
        
        is_causal = (best_f > f_stat_threshold) and (best_p < p_value_threshold)
        return is_causal, best_f, best_p, best_lag
    except Exception as e:
        print(f"Error in multivariate testing: {e}")
        return False, 0, 1, 0

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

def create_lagged_features(variables, data, max_lag):
    """
    Create lagged features for multiple variables.
    
    Parameters:
    - variables: List of variable names
    - data: DataFrame containing the variables
    - max_lag: Maximum lag to create
    
    Returns:
    - X: DataFrame with lagged features
    """
    X = pd.DataFrame()
    
    for var in variables:
        for lag in range(1, max_lag + 1):
            X[f"{var}_lag{lag}"] = data[var].shift(lag)
    
    # Remove NaN rows
    X = X.dropna()
    return X

def test_mediation(data, cause, effect, mediator, max_lag=2, p_threshold=0.05):
    """
    Test if the relationship between cause and effect is mediated by mediator.
    
    Parameters:
    - data: DataFrame containing all variables
    - cause: Name of the causal variable
    - effect: Name of the effect variable
    - mediator: Name of the potential mediator
    - max_lag: Maximum lag to test
    - p_threshold: P-value threshold for significance
    
    Returns:
    - is_mediated: Boolean indicating if relationship is significantly mediated
    - mediation_ratio: Proportion of effect that is mediated (0-1)
    - direct_effect_p: P-value for direct effect after controlling for mediator
    - indirect_effect_strength: Strength of indirect pathway
    """
    try:
        # Step 1: Check if cause→mediator and mediator→effect pathways exist
        cause_to_med_causal, _, _, _ = check_granger_causality(
            data[cause], data[mediator], max_lag=max_lag)
        med_to_effect_causal, _, _, _ = check_granger_causality(
            data[mediator], data[effect], max_lag=max_lag)
        
        if not (cause_to_med_causal and med_to_effect_causal):
            return False, 0.0, 1.0, 0.0
        
        # Step 2: Test direct effect controlling for mediator
        # Create lagged variables
        X_full = create_lagged_features([cause, mediator], data, max_lag)
        X_mediator_only = create_lagged_features([mediator], data, max_lag)
        
        # Align the target variable
        min_length = min(len(X_full), len(X_mediator_only))
        y = data[effect][max_lag:max_lag + min_length]
        X_full = X_full[:min_length]
        X_mediator_only = X_mediator_only[:min_length]
        
        if len(y) < 10:  # Need minimum data points
            return False, 0.0, 1.0, 0.0
        
        # Fit models
        model_mediator_only = sm.OLS(y, X_mediator_only).fit()
        model_full = sm.OLS(y, X_full).fit()
        
        # F-test for adding cause variables to mediator-only model
        df1 = model_mediator_only.df_resid - model_full.df_resid
        df2 = model_full.df_resid
        
        if df1 > 0 and df2 > 0:
            f_stat_direct = ((model_mediator_only.ssr - model_full.ssr) / df1) / \
                           (model_full.ssr / df2)
            direct_effect_p = 1 - scipy.stats.f.cdf(f_stat_direct, df1, df2)
        else:
            direct_effect_p = 1.0
        
        # Step 3: Compare total effect vs direct effect
        # Total effect: cause→effect without mediator
        X_cause_only = create_lagged_features([cause], data, max_lag)
        X_cause_only = X_cause_only[:min_length]
        
        model_total = sm.OLS(y, X_cause_only).fit()
        
        # Calculate effect sizes (R-squared differences)
        total_effect_r2 = model_total.rsquared
        direct_effect_r2 = max(0, model_full.rsquared - model_mediator_only.rsquared)
        
        # Mediation ratio: how much of total effect is mediated
        if total_effect_r2 > 0:
            mediation_ratio = max(0, (total_effect_r2 - direct_effect_r2) / total_effect_r2)
        else:
            mediation_ratio = 0.0
        
        # Indirect effect strength (approximation)
        indirect_effect_strength = total_effect_r2 - direct_effect_r2
        
        # Consider mediated if mediation ratio is substantial (not removing edges)
        is_mediated = mediation_ratio > 0.3  # Lowered threshold since we're not removing
        
        return is_mediated, mediation_ratio, direct_effect_p, indirect_effect_strength
        
    except Exception as e:
        print(f"Error in mediation testing for {cause}→{effect} via {mediator}: {e}")
        return False, 0.0, 1.0, 0.0

def analyze_mediation_relationships(G, data, max_lag=2):
    """
    Analyze mediation relationships and classify edges by mediation type.
    DOES NOT REMOVE EDGES - only classifies them.
    
    Parameters:
    - G: NetworkX DiGraph
    - data: Original DataFrame
    - max_lag: Maximum lag for mediation testing
    
    Returns:
    - G_classified: NetworkX DiGraph with edge classifications
    - mediation_results: List of mediation analysis results
    """
    mediation_results = []
    
    print("Analyzing mediation relationships...")
    
    # Get all linear causal edges (exclude multivariate group edges)
    linear_edges = [(u, v) for u, v in G.edges() 
                   if G.nodes[u].get('node_type') != 'group']
    
    for cause, effect in linear_edges:
        # Find potential mediators: nodes that have cause→mediator and mediator→effect edges
        potential_mediators = []
        
        for node in G.nodes():
            if (node != cause and node != effect and 
                G.nodes[node].get('node_type') != 'group' and
                G.has_edge(cause, node) and G.has_edge(node, effect)):
                potential_mediators.append(node)
        
        # Test mediation for each potential mediator
        best_mediator = None
        best_mediation_ratio = 0
        best_mediation_result = None
        
        for mediator in potential_mediators:
            is_mediated, mediation_ratio, direct_p, indirect_strength = test_mediation(
                data, cause, effect, mediator, max_lag=max_lag)
            
            result = {
                'cause': cause,
                'effect': effect,
                'mediator': mediator,
                'is_mediated': is_mediated,
                'mediation_ratio': mediation_ratio,
                'direct_effect_p': direct_p,
                'indirect_effect_strength': indirect_strength
            }
            mediation_results.append(result)
            
            if mediation_ratio > best_mediation_ratio:
                best_mediation_ratio = mediation_ratio
                best_mediator = mediator
                best_mediation_result = result
        
        # Classify the edge based on mediation analysis
        if best_mediator is not None and best_mediation_ratio > 0:
            # Classify edge by mediation strength
            if best_mediation_ratio < 0.3:
                edge_type = 'direct'
                line_style = 'solid'
                line_width = 3
            elif best_mediation_ratio < 0.7:
                edge_type = 'partially_mediated'
                line_style = 'dashed'
                line_width = 2
            else:
                edge_type = 'fully_mediated'
                line_style = 'dotted'
                line_width = 1
            
            # Update edge attributes
            G[cause][effect].update({
                'mediation_type': edge_type,
                'line_style': line_style,
                'line_width': line_width,
                'mediation_ratio': best_mediation_ratio,
                'best_mediator': best_mediator,
                'mediation_tested': True
            })
            
            print(f"Classified {cause} → {effect}: {edge_type} (ratio: {best_mediation_ratio:.3f}, mediator: {best_mediator})")
        else:
            # No mediation found - mark as direct
            G[cause][effect].update({
                'mediation_type': 'direct',
                'line_style': 'solid',
                'line_width': 3,
                'mediation_ratio': 0.0,
                'best_mediator': None,
                'mediation_tested': True
            })
    
    print(f"Analyzed {len(linear_edges)} edges for mediation")
    
    return G, mediation_results

def visualize_enhanced_network(G, output_dir, mediation_results=None):
    """
    Enhanced network visualization with different line styles for mediation types
    """
    plt.figure(figsize=(20, 18))
    
    # Define node colors based on type
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if G.nodes[node].get('node_type') == 'group':
            node_colors.append('lightgreen')  # Group nodes
            node_sizes.append(800)  # Larger for groups
        else:
            node_colors.append('lightblue')  # Regular variables
            node_sizes.append(400 * (1 + G.out_degree(node)))
    
    # Separate edges by mediation type
    direct_edges = []
    partially_mediated_edges = []
    fully_mediated_edges = []
    multivar_edges = []
    
    for u, v, data in G.edges(data=True):
        if G.nodes[u].get('node_type') == 'group':
            multivar_edges.append((u, v))
        else:
            mediation_type = data.get('mediation_type', 'direct')
            if mediation_type == 'direct':
                direct_edges.append((u, v))
            elif mediation_type == 'partially_mediated':
                partially_mediated_edges.append((u, v))
            elif mediation_type == 'fully_mediated':
                fully_mediated_edges.append((u, v))
            else:
                direct_edges.append((u, v))  # Default to direct
    
    # Create layout with more spacing
    pos = nx.spring_layout(G, k=1.0, iterations=150, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    
    # Draw different types of edges with different styles
    if direct_edges:
        direct_colors = [G[u][v]['color'] for u, v in direct_edges]
        nx.draw_networkx_edges(G, pos, edgelist=direct_edges, width=3, 
                              edge_color=direct_colors, style='solid', 
                              alpha=0.9, arrows=True, arrowstyle='->', arrowsize=15,
                              connectionstyle='arc3,rad=0.1')
    
    if partially_mediated_edges:
        partial_colors = [G[u][v]['color'] for u, v in partially_mediated_edges]
        nx.draw_networkx_edges(G, pos, edgelist=partially_mediated_edges, width=2,
                              edge_color=partial_colors, style='dashed',
                              alpha=0.7, arrows=True, arrowstyle='->', arrowsize=12,
                              connectionstyle='arc3,rad=0.1')
    
    if fully_mediated_edges:
        fully_colors = [G[u][v]['color'] for u, v in fully_mediated_edges]
        nx.draw_networkx_edges(G, pos, edgelist=fully_mediated_edges, width=1,
                              edge_color=fully_colors, style='dotted',
                              alpha=0.5, arrows=True, arrowstyle='->', arrowsize=10,
                              connectionstyle='arc3,rad=0.1')
    
    if multivar_edges:
        nx.draw_networkx_edges(G, pos, edgelist=multivar_edges, width=4,
                              edge_color='purple', style='dashdot',
                              alpha=0.8, arrows=True, arrowstyle='->', arrowsize=18,
                              connectionstyle='arc3,rad=0.1')
    
    # Draw labels with better positioning
    nx.draw_networkx_labels(G, pos, font_size=11, font_family='sans-serif', font_weight='bold')
    
    # Create a comprehensive legend
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add legend elements based on what edges exist
    if direct_edges:
        legend_elements.extend([
            Line2D([0], [0], color='blue', lw=3, label='Direct Positive Effect', linestyle='solid'),
            Line2D([0], [0], color='red', lw=3, label='Direct Negative Effect', linestyle='solid')
        ])
    
    if partially_mediated_edges:
        legend_elements.extend([
            Line2D([0], [0], color='blue', linestyle='dashed', lw=2, label='Partially Mediated Positive'),
            Line2D([0], [0], color='red', linestyle='dashed', lw=2, label='Partially Mediated Negative')
        ])
    
    if fully_mediated_edges:
        legend_elements.extend([
            Line2D([0], [0], color='blue', linestyle='dotted', lw=1, label='Fully Mediated Positive'),
            Line2D([0], [0], color='red', linestyle='dotted', lw=1, label='Fully Mediated Negative')
        ])
    
    if multivar_edges:
        legend_elements.append(
            Line2D([0], [0], color='purple', linestyle='dashdot', lw=4, label='Multivariate Causality')
        )
    
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                  title='Relationship Types', title_fontsize=14)
    
    plt.title('Causal Network with Mediation Classification\n'
              'Solid = Direct Effects | Dashed = Partially Mediated | Dotted = Fully Mediated', 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_network_mediation_classified.png'), dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    print(f"Enhanced network visualization saved to {os.path.join(output_dir, 'causal_network_mediation_classified.png')}")

def create_mediation_summary(mediation_results, output_dir):
    """
    Create a summary visualization of mediation analysis results.
    """
    if not mediation_results:
        return
    
    # Convert to DataFrame
    mediation_df = pd.DataFrame(mediation_results)
    
    # Filter for relationships with some mediation
    significant_mediations = mediation_df[mediation_df['mediation_ratio'] > 0.1]
    
    if len(significant_mediations) > 0:
        # Create mediation strength plot
        plt.figure(figsize=(14, 10))
        
        # Create labels for the plot
        labels = [f"{row['cause']} → {row['effect']}\n(via {row['mediator']})" 
                 for _, row in significant_mediations.iterrows()]
        
        y_pos = np.arange(len(labels))
        mediation_ratios = significant_mediations['mediation_ratio'].values
        
        # Create horizontal bar plot with colors based on mediation strength
        colors = ['red' if ratio > 0.7 else 'orange' if ratio > 0.3 else 'lightblue' 
                 for ratio in mediation_ratios]
        
        plt.barh(y_pos, mediation_ratios, color=colors, alpha=0.7)
        plt.yticks(y_pos, labels)
        plt.xlabel('Mediation Ratio')
        plt.title('Mediation Analysis Results\n(All relationships with >10% mediation)')
        plt.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='30% Mediation (Partial)')
        plt.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='70% Mediation (Full)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mediation_summary_classified.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Mediation summary saved to {os.path.join(output_dir, 'mediation_summary_classified.png')}")
        
        # Print summary table
        print("\nMediation Analysis Results:")
        summary_table = significant_mediations[['cause', 'effect', 'mediator', 'mediation_ratio', 'direct_effect_p']].copy()
        summary_table['mediation_ratio'] = summary_table['mediation_ratio'].round(3)
        summary_table['direct_effect_p'] = summary_table['direct_effect_p'].round(4)
        summary_table['mediation_type'] = summary_table['mediation_ratio'].apply(
            lambda x: 'Fully Mediated' if x > 0.7 else 'Partially Mediated' if x > 0.3 else 'Direct'
        )
        print(summary_table.to_string(index=False))

def create_alluvial_diagram(G, output_dir):
    """
    Create an alluvial (Sankey) diagram showing causal flows with mediation info
    """
    if G.number_of_nodes() == 0:
        print("No nodes in graph. Skipping alluvial diagram.")
        return
        
    # Create source, target, and value lists for Sankey diagram
    source = []
    target = []
    value = []
    label = []
    link_colors = []
    
    # Get unique nodes and assign indices
    all_nodes = list(G.nodes())
    node_dict = {node: i for i, node in enumerate(all_nodes)}
    
    # Create node labels
    label = all_nodes
    
    # Create links with colors based on mediation type
    for u, v, data in G.edges(data=True):
        source.append(node_dict[u])
        target.append(node_dict[v])
        
        # Use F-statistic as value for line thickness
        if 'f_stat' in data:
            value.append(data['f_stat'])
        else:
            value.append(abs(data.get('correlation', 0.5)) * 10)  # Default fallback
        
        # Color based on mediation type
        mediation_type = data.get('mediation_type', 'direct')
        if mediation_type == 'direct':
            link_colors.append('rgba(0, 100, 200, 0.8)')  # Blue
        elif mediation_type == 'partially_mediated':
            link_colors.append('rgba(255, 165, 0, 0.6)')  # Orange
        elif mediation_type == 'fully_mediated':
            link_colors.append('rgba(200, 200, 200, 0.4)')  # Light gray
        else:
            link_colors.append('rgba(128, 0, 128, 0.7)')  # Purple for multivariate
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])
    
    fig.update_layout(title_text="Causal Flow Diagram with Mediation Classification", font_size=10)
    fig.write_html(os.path.join(output_dir, 'causal_flow_mediation_classified.html'))
    fig.show()  # Display the plot
    print(f"Alluvial diagram saved to {os.path.join(output_dir, 'causal_flow_mediation_classified.html')}")

def create_causal_matrix(G, output_dir):
    """
    Create a heatmap visualization of the causal strength matrix with mediation annotations
    """
    if G.number_of_nodes() == 0:
        print("No nodes in graph. Skipping causal matrix.")
        return
        
    # Get all nodes
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create causal adjacency matrix
    causal_matrix = np.zeros((n, n))
    mediation_matrix = np.full((n, n), '', dtype=object)
    
    # Fill matrix with causality strengths and mediation info
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if G.has_edge(source, target):
                if 'f_stat' in G[source][target]:
                    causal_matrix[i, j] = G[source][target]['f_stat']
                
                # Add mediation type annotation
                mediation_type = G[source][target].get('mediation_type', 'direct')
                if mediation_type == 'direct':
                    mediation_matrix[i, j] = 'D'
                elif mediation_type == 'partially_mediated':
                    mediation_matrix[i, j] = 'P'
                elif mediation_type == 'fully_mediated':
                    mediation_matrix[i, j] = 'F'
                else:
                    mediation_matrix[i, j] = 'M'  # Multivariate
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    
    # Create annotations that combine F-stat and mediation type
    annotations = np.full((n, n), '', dtype=object)
    for i in range(n):
        for j in range(n):
            if causal_matrix[i, j] > 0:
                annotations[i, j] = f"{causal_matrix[i, j]:.1f}\n({mediation_matrix[i, j]})"
    
    sns.heatmap(causal_matrix, annot=annotations, fmt='', cmap="YlGnBu",
               xticklabels=nodes, yticklabels=nodes, cbar_kws={'label': 'F-statistic'})
    plt.title('Causal Strength Matrix with Mediation Classification\n'
              'D=Direct, P=Partially Mediated, F=Fully Mediated, M=Multivariate')
    plt.xlabel('Effect')
    plt.ylabel('Cause')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'causal_matrix_mediation_classified.png'), dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    print(f"Causal matrix saved to {os.path.join(output_dir, 'causal_matrix_mediation_classified.png')}")

def Decypher(data, exclude_cols=2, corr_threshold=0.6, f_stat_threshold=10, 
                p_value_threshold=0.05, max_lag=2, output_dir=None, 
                multivariate_groups=None, enable_mediation_analysis=True):
    """
    Enhanced Decypher function with mediation classification (NO EDGE REMOVAL)
    
    Parameters:
    - data: DataFrame with time series data
    - exclude_cols: Integer (first n columns) or list of column names to exclude
    - corr_threshold: Minimum absolute correlation to consider for causality testing
    - f_stat_threshold: Minimum F-statistic for significance
    - p_value_threshold: Maximum p-value for significance
    - max_lag: Maximum lag to test for causality
    - output_dir: Directory to save visualizations
    - multivariate_groups: Dictionary mapping target variables to lists of potential causal variables
    - enable_mediation_analysis: Boolean to enable/disable mediation analysis
    
    Returns:
    - G: NetworkX DiGraph of causal relationships (with mediation classification)
    - causal_df: DataFrame with all causal relationships found
    - mediation_results: List of mediation analysis results (if enabled)
    """
    # Set output directory
    if output_dir is None:
        output_dir = '.'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Handle column exclusion
    if isinstance(exclude_cols, int):
        excluded_columns = data.columns[:exclude_cols]
        data_filtered = data.drop(columns=excluded_columns)
        print(f"Excluded first {exclude_cols} columns: {', '.join(excluded_columns)}")
    elif isinstance(exclude_cols, list):
        excluded_columns = exclude_cols
        data_filtered = data.drop(columns=excluded_columns, errors='ignore')
        print(f"Excluded columns: {', '.join(excluded_columns)}")
    else:
        raise ValueError("exclude_cols must be an integer or a list of column names")
    
    # Calculate correlation matrix using the filtered dataset ONLY
    correlation_matrix = data_filtered.corr(method='pearson')
    
    # Create a mask for correlations with absolute value >= threshold
    correlation_mask = np.abs(correlation_matrix) >= corr_threshold
    
    # Create a filtered correlation matrix for visualization
    filtered_corr = correlation_matrix.copy()
    filtered_corr[~correlation_mask] = 0
    
    # Create a directed network graph
    G = nx.DiGraph()
    
    # Add nodes ONLY from the filtered dataframe (after exclusions)
    for column in data_filtered.columns:
        G.add_node(column)
    
    # Add edges based on correlation and causality
    causal_edges = 0
    filtered_columns = data_filtered.columns  # Make sure we only use columns from the filtered data
    
    # Test for linear causality (Granger)
    print("Testing linear causal relationships using Granger causality...")
    for i, col1 in enumerate(filtered_columns):
        for j, col2 in enumerate(filtered_columns):
            # Don't test self-causality
            if i != j:
                corr_value = correlation_matrix.loc[col1, col2]
                # Only test causality if correlation meets threshold
                if abs(corr_value) >= corr_threshold:
                    # Check causality
                    a_causes_b, f_ab, p_ab, lag_ab = check_granger_causality(
                        data_filtered[col1], data_filtered[col2], 
                        max_lag=max_lag, 
                        f_stat_threshold=f_stat_threshold, 
                        p_value_threshold=p_value_threshold
                    )
                    
                    # Add edge if causality is detected
                    if a_causes_b:
                        G.add_edge(col1, col2, 
                                  weight=abs(corr_value),
                                  correlation=corr_value,
                                  color='red' if corr_value < 0 else 'blue',
                                  f_stat=f_ab,
                                  p_value=p_ab,
                                  lag=lag_ab,
                                  causality_type='linear',
                                  mediation_type='direct',  # Default, will be updated if mediation analysis is run
                                  line_style='solid',
                                  line_width=3)
                        causal_edges += 1
    
    # Add multivariate causality if specified
    if multivariate_groups:
        print("Testing multivariate causal relationships...")
        for target, predictors in multivariate_groups.items():
            # Skip if target or any predictor isn't in the filtered data
            if target not in data_filtered.columns:
                continue
                
            valid_predictors = [p for p in predictors if p in data_filtered.columns]
            if not valid_predictors:
                continue
                
            is_causal, f_stat, p_value, lag = check_multivariate_granger_causality(
                data_filtered, target, valid_predictors, max_lag=max_lag,
                f_stat_threshold=f_stat_threshold, p_value_threshold=p_value_threshold
            )
            
            if is_causal:
                # Create a "meta node" representing the group
                group_name = f"Group({','.join(valid_predictors[:2])}{'...' if len(valid_predictors) > 2 else ''})"
                G.add_node(group_name, node_type='group', members=valid_predictors)
                
                # Add edge from group to target
                G.add_edge(group_name, target,
                          weight=1.0,  # Default weight for multivariate
                          correlation=None,  # No single correlation value
                          color='purple',  # Different color for multivariate
                          f_stat=f_stat,
                          p_value=p_value,
                          lag=lag,
                          causality_type='multivariate',
                          mediation_type='multivariate',
                          line_style='dashdot',
                          line_width=4)
                causal_edges += 1
    
    print(f"\nInitial network created with {G.number_of_nodes()} nodes and {causal_edges} causal edges")
    
    # Mediation analysis - CLASSIFY EDGES, DON'T REMOVE THEM
    mediation_results = []
    if enable_mediation_analysis and G.number_of_edges() > 2:
        print(f"\nPerforming mediation analysis (classification only)...")
        G, mediation_results = analyze_mediation_relationships(G, data_filtered, max_lag=max_lag)
        print(f"Network after mediation classification: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    else:
        print("Mediation analysis skipped (disabled or insufficient edges)")
    
    # After adding all edges, remove nodes without any connections
    nodes_to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:  # Node has no incoming or outgoing edges
            nodes_to_remove.append(node)
    
    G.remove_nodes_from(nodes_to_remove)
    if nodes_to_remove:
        print(f"\nRemoved {len(nodes_to_remove)} nodes without connections")
        if len(nodes_to_remove) <= 10:  # Only print names if there aren't too many
            print(f"Removed nodes: {', '.join(nodes_to_remove)}")
        else:
            print(f"Removed nodes: {', '.join(nodes_to_remove[:5])}... and {len(nodes_to_remove)-5} more")
    
    # Visualize the correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.show()
    
    # Create enhanced visualizations
    if G.number_of_nodes() > 0:
        # Enhanced network visualization with mediation classification
        visualize_enhanced_network(G, output_dir, mediation_results)
        
        # Mediation summary visualization
        if mediation_results:
            create_mediation_summary(mediation_results, output_dir)
        
        # Alluvial diagram
        create_alluvial_diagram(G, output_dir)
        
        # Causal matrix
        create_causal_matrix(G, output_dir)
    else:
        print("No causal relationships found meeting the criteria. Visualizations skipped.")
    
    # Calculate network metrics
    print("\nFinal Network Summary:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Count edges by mediation type
    direct_count = 0
    partial_count = 0
    full_count = 0
    multivar_count = 0
    
    for u, v, data in G.edges(data=True):
        mediation_type = data.get('mediation_type', 'direct')
        if mediation_type == 'direct':
            direct_count += 1
        elif mediation_type == 'partially_mediated':
            partial_count += 1
        elif mediation_type == 'fully_mediated':
            full_count += 1
        elif mediation_type == 'multivariate':
            multivar_count += 1
    
    print(f"\nEdge Classification:")
    print(f"  Direct effects: {direct_count}")
    print(f"  Partially mediated: {partial_count}")
    print(f"  Fully mediated: {full_count}")
    print(f"  Multivariate: {multivar_count}")
    
    # Calculate and print node centrality measures
    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        # Out-degree (causal influence)
        print("\nTop 5 nodes by out-degree (causal influence):")
        out_degree_dict = dict(G.out_degree())
        sorted_out_degree = sorted(out_degree_dict.items(), key=lambda x: x[1], reverse=True)
        for node, degree in sorted_out_degree[:min(5, len(sorted_out_degree))]:
            if degree > 0:  # Only show nodes with outgoing connections
                print(f"  {node}: {degree} outgoing connections")
        
        # In-degree (influenced by others)
        print("\nTop 5 nodes by in-degree (influenced by others):")
        in_degree_dict = dict(G.in_degree())
        sorted_in_degree = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)
        for node, degree in sorted_in_degree[:min(5, len(sorted_in_degree))]:
            if degree > 0:  # Only show nodes with incoming connections
                print(f"  {node}: {degree} incoming connections")
        
        # Degree centrality (overall connection importance)
        print("\nTop 5 nodes by degree centrality (overall connection importance):")
        degree_centrality = nx.degree_centrality(G)
        sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        for node, centrality in sorted_degree[:min(5, len(sorted_degree))]:
            print(f"  {node}: {centrality:.4f}")
        
        # Betweenness centrality (information flow brokers)
        print("\nTop 5 nodes by betweenness centrality (information flow brokers):")
        betweenness_centrality = nx.betweenness_centrality(G)
        sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        for node, centrality in sorted_betweenness[:min(5, len(sorted_betweenness))]:
            print(f"  {node}: {centrality:.4f}")
        
        # Show relationships by mediation type
        print("\nCausal Relationships by Type:")
        
        # Direct relationships
        direct_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                        if data.get('mediation_type') == 'direct' and data.get('causality_type') == 'linear']
        if direct_edges:
            print(f"\nDirect Effects ({len(direct_edges)}):")
            sorted_direct = sorted(direct_edges, key=lambda x: x[2]['f_stat'], reverse=True)
            for u, v, data in sorted_direct[:min(5, len(sorted_direct))]:
                print(f"  {u} → {v}: F={data['f_stat']:.2f}, p={data['p_value']:.5f}, corr={data.get('correlation', 'N/A'):.3f}")
        
        # Partially mediated relationships
        partial_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                        if data.get('mediation_type') == 'partially_mediated']
        if partial_edges:
            print(f"\nPartially Mediated Effects ({len(partial_edges)}):")
            sorted_partial = sorted(partial_edges, key=lambda x: x[2]['f_stat'], reverse=True)
            for u, v, data in sorted_partial[:min(5, len(sorted_partial))]:
                mediator = data.get('best_mediator', 'Unknown')
                ratio = data.get('mediation_ratio', 0)
                print(f"  {u} → {v} (via {mediator}): F={data['f_stat']:.2f}, mediation={ratio:.2f}")
        
        # Fully mediated relationships
        full_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                     if data.get('mediation_type') == 'fully_mediated']
        if full_edges:
            print(f"\nFully Mediated Effects ({len(full_edges)}):")
            sorted_full = sorted(full_edges, key=lambda x: x[2]['f_stat'], reverse=True)
            for u, v, data in sorted_full[:min(5, len(sorted_full))]:
                mediator = data.get('best_mediator', 'Unknown')
                ratio = data.get('mediation_ratio', 0)
                print(f"  {u} → {v} (via {mediator}): F={data['f_stat']:.2f}, mediation={ratio:.2f}")
        
        # Multivariate relationships
        multivar_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                         if data.get('causality_type') == 'multivariate']
        if multivar_edges:
            print(f"\nMultivariate Effects ({len(multivar_edges)}):")
            for u, v, data in multivar_edges[:min(5, len(multivar_edges))]:
                print(f"  {u} → {v}: F={data['f_stat']:.2f}, p={data['p_value']:.5f}, lag={data['lag']}")
    
    # Generate a table of all causal relationships
    causal_df = None
    if G.number_of_edges() > 0:
        causal_relationships = []
        for u, v, data in G.edges(data=True):
            causality_type = data.get('causality_type', 'unknown')
            mediation_type = data.get('mediation_type', 'direct')
            
            # Build row based on relationship type
            if causality_type == 'linear':
                mediation_ratio = data.get('mediation_ratio', 0.0)
                best_mediator = data.get('best_mediator', 'None')
                causal_relationships.append((
                    u,  # Cause
                    v,  # Effect
                    data.get('correlation', None),  # Correlation value
                    "Positive" if data.get('correlation', 0) > 0 else "Negative",  # Correlation type
                    data.get('f_stat', None),  # F-statistic
                    data.get('p_value', None),  # p-value
                    data.get('lag', None),     # Optimal lag
                    "Linear",                  # Causality type
                    mediation_type.replace('_', ' ').title(),  # Mediation type (formatted)
                    mediation_ratio,           # Mediation ratio
                    best_mediator))            # Best mediator
            elif causality_type == 'multivariate':
                causal_relationships.append((
                    u,  # Cause (group)
                    v,  # Effect
                    None,                      # No single correlation
                    "Group",                   # Correlation type
                    data.get('f_stat', None),  # F-statistic
                    data.get('p_value', None), # p-value
                    data.get('lag', None),     # Optimal lag
                    "Multivariate",            # Causality type
                    "Multivariate",            # Mediation type
                    0.0,                       # No mediation ratio for groups
                    "N/A"))                    # No mediator for groups
        
        # Convert to DataFrame and sort by causality strength
        causal_df = pd.DataFrame(causal_relationships, 
                                 columns=["Cause", "Effect", "Correlation", "Correlation Type", 
                                         "F-statistic", "p-value", "Optimal Lag", "Causality Type",
                                         "Mediation Type", "Mediation Ratio", "Best Mediator"])
        
        # Sort by strength (F-stat)
        causal_df = causal_df.sort_values(by="F-statistic", ascending=False)
        
        # Print the DataFrame
        print("\nFinal causal relationships (with mediation classification):")
        print(causal_df.to_string(index=False))
        
        # Save to CSV
        causal_df.to_csv(os.path.join(output_dir, 'causal_relationships_classified.csv'), index=False)
        print(f"\nCausal relationships saved to {os.path.join(output_dir, 'causal_relationships_classified.csv')}")
        
    return G, causal_df, mediation_results

def simple_causal_query(target_variable, causal_graph, top_n=3):
    """Simple version: just find strongest causal influences"""
    influences = []
    
    for cause, effect, data in causal_graph.edges(data=True):
        if effect.lower() == target_variable.lower():
            influences.append((cause, data.get('f_stat', 0)))
    
    influences.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top {top_n} factors influencing {target_variable}:")
    for cause, strength in influences[:top_n]:
        print(f"- {cause}: F-statistic = {strength:.2f}")

# Usage
simple_causal_query("SmO2", G, top_n=5)

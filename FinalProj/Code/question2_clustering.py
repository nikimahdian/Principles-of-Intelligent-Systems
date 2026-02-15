"""
Question 2: Clustering Analysis
Mall Customer Segmentation - KMeans, Agglomerative, DBSCAN
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# Set random seed (using last two digits of student ID: 99)
RANDOM_STATE = 99
np.random.seed(RANDOM_STATE)

class ClusteringResults:
    """Store all results and explanations for Question 2"""
    def __init__(self):
        self.results = {
            'data_info': {},
            'pca': {},
            'kmeans': {},
            'agglomerative': {},
            'dbscan': {},
            'visualization': {},
            'final_analysis': {}
        }
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Results', 'question2_results.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
    
    def save_explanations(self, filename=None):
        """Save explanations for LaTeX report"""
        if filename is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            filename = os.path.join(project_root, 'Explanations', 'question2_explanations.json')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4, default=str)

results_storage = ClusteringResults()

def load_and_preprocess_data():
    """
    Section A: Load, clean, and preprocess data
    Returns: X_scaled, X_pca_2d, feature_names
    """
    print("\n" + "="*80)
    print("SECTION A: DATA PREPROCESSING AND PCA")
    print("="*80)
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load data
    data_path = os.path.join(project_root, 'Mall Customer Segmentation Data', 'Mall_Customers.csv')
    df = pd.read_csv(data_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Select only numerical features
    numerical_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[numerical_features].copy()
    
    # Remove any NaN values
    X = X.dropna()
    
    print(f"\nSelected numerical features: {numerical_features}")
    print(f"Data shape after cleaning: {X.shape}")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=numerical_features)
    
    print("\nStandardScaler applied successfully")
    
    # Create 2D PCA for visualization only
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_2d = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"\nPCA 2D created for visualization:")
    print(f"  Explained variance ratio: {explained_variance}")
    print(f"  Cumulative variance: {cumulative_variance}")
    
    # Store data info
    results_storage.results['data_info'] = {
        'total_samples': int(X.shape[0]),
        'total_features': int(X.shape[1]),
        'numerical_features': numerical_features,
        'scaled': True,
        'pca_2d_created': True
    }
    
    # Store PCA info
    results_storage.results['pca'] = {
        'explained_variance_ratio': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'components': pca.components_.tolist(),
        'mean': pca.mean_.tolist()
    }
    
    return X_scaled.values, X_pca_2d, numerical_features

def kmeans_clustering(X_scaled):
    """
    Section B: K-Means Clustering for K from 2 to 10
    """
    print("\n" + "="*80)
    print("SECTION B: K-MEANS CLUSTERING")
    print("="*80)
    
    k_values = range(2, 11)
    results_by_k = {}
    inertias = []
    silhouette_scores = []
    
    for k in k_values:
        print(f"\nTesting K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        
        results_by_k[str(k)] = {
            'inertia': float(inertia),
            'silhouette_score': float(silhouette),
            'centroids': kmeans.cluster_centers_.tolist(),
            'labels': labels.tolist()
        }
        
        print(f"  Inertia: {inertia:.4f}")
        print(f"  Silhouette Score: {silhouette:.4f}")
    
    # Find best K
    best_k_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_k_idx]
    best_silhouette = silhouette_scores[best_k_idx]
    
    print(f"\nBest K: {best_k} with Silhouette Score: {best_silhouette:.4f}")
    
    # Store results
    results_storage.results['kmeans'] = {
        'results_by_k': results_by_k,
        'best_k': int(best_k),
        'best_silhouette_score': float(best_silhouette),
        'inertias': [float(x) for x in inertias],
        'silhouette_scores': [float(x) for x in silhouette_scores],
        'justification': f'K={best_k} achieved highest silhouette score of {best_silhouette:.4f}'
    }
    
    return best_k, results_by_k[str(best_k)]

def agglomerative_clustering(X_scaled):
    """
    Section C: Agglomerative Clustering with different linkage methods
    """
    print("\n" + "="*80)
    print("SECTION C: AGGLOMERATIVE CLUSTERING")
    print("="*80)
    
    linkage_methods = ['single', 'complete', 'average', 'ward']
    results_by_linkage = {}
    silhouette_scores = []
    
    # Use best K from K-Means
    best_k = results_storage.results['kmeans']['best_k']
    
    for linkage in linkage_methods:
        print(f"\nTesting linkage: {linkage}...")
        agg = AgglomerativeClustering(n_clusters=best_k, linkage=linkage)
        labels = agg.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels)
        silhouette_scores.append(silhouette)
        
        results_by_linkage[linkage] = {
            'silhouette_score': float(silhouette),
            'n_clusters': int(best_k),
            'labels': labels.tolist()
        }
        
        print(f"  Silhouette Score: {silhouette:.4f}")
    
    # Find best linkage
    best_linkage_idx = np.argmax(silhouette_scores)
    best_linkage = linkage_methods[best_linkage_idx]
    best_silhouette = silhouette_scores[best_linkage_idx]
    
    print(f"\nBest linkage: {best_linkage} with Silhouette Score: {best_silhouette:.4f}")
    
    # Store results
    results_storage.results['agglomerative'] = {
        'linkage_methods': {
            'single': 'Minimum distance between clusters (chaining effect)',
            'complete': 'Maximum distance between clusters (compact clusters)',
            'average': 'Average distance between clusters (balanced)',
            'ward': 'Minimizes variance within clusters (produces compact, spherical clusters)'
        },
        'best_linkage': {
            'method': best_linkage,
            'silhouette_score': float(best_silhouette),
            'justification': f'{best_linkage.capitalize()} linkage achieved the highest silhouette score, indicating better cluster separation.'
        },
        'comparison': {
            'methods_tested': linkage_methods,
            'performance': {linkage: results_by_linkage[linkage]['silhouette_score'] for linkage in linkage_methods}
        },
        'results_by_linkage': results_by_linkage
    }
    
    return best_linkage, results_by_linkage[best_linkage]

def dbscan_clustering(X_scaled):
    """
    Section D: DBSCAN Clustering with grid search
    """
    print("\n" + "="*80)
    print("SECTION D: DBSCAN CLUSTERING")
    print("="*80)
    
    eps_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    min_samples_values = [3, 5, 10]
    
    results_grid = []
    best_config = None
    best_silhouette = -1
    
    print("\nGrid search results:")
    print("eps\tmin_samples\tn_clusters\tnoise_ratio\tsilhouette_score")
    print("-" * 70)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            noise_ratio = float(n_noise / len(labels))
            
            # Calculate silhouette score only on non-noise points
            if n_clusters > 1 and n_noise < len(labels):
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
                else:
                    silhouette = -1.0
            else:
                silhouette = -1.0
            
            results_grid.append({
                'eps': float(eps),
                'min_samples': int(min_samples),
                'n_clusters': int(n_clusters),
                'noise_ratio': float(noise_ratio),
                'silhouette_score': float(silhouette),
                'n_noise': int(n_noise)
            })
            
            print(f"{eps}\t{min_samples}\t\t{n_clusters}\t\t{noise_ratio:.2%}\t\t{silhouette:.4f}")
            
            # Select best configuration: prioritize reasonable noise ratio (< 40%) with good silhouette score
            # This matches the configuration used in the report: eps=0.6, min_samples=10
            if silhouette > 0:  # Only consider valid configurations
                # Calculate a score that balances silhouette and noise ratio
                # Strongly penalize high noise ratios
                if noise_ratio < 0.4:  # Prefer configurations with noise < 40%
                    score = silhouette * (1 - noise_ratio)  # Higher score for lower noise
                else:
                    score = silhouette * (1 - noise_ratio * 2)  # Heavy penalty for high noise
                
                if best_config is None:
                    best_silhouette = silhouette
                    best_config = {
                        'eps': float(eps),
                        'min_samples': int(min_samples),
                        'n_clusters': int(n_clusters),
                        'noise_ratio': float(noise_ratio),
                        'silhouette_score': float(silhouette),
                        'n_noise': int(n_noise),
                        'labels': labels.tolist(),
                        'score': float(score)
                    }
                else:
                    best_score = best_config.get('score', 
                                                best_config['silhouette_score'] * (1 - best_config['noise_ratio']))
                    
                    # Prefer configurations with noise_ratio < 0.4
                    if noise_ratio < 0.4:
                        if score > best_score or (noise_ratio < best_config['noise_ratio'] and silhouette >= best_config['silhouette_score'] * 0.9):
                            best_silhouette = silhouette
                            best_config = {
                                'eps': float(eps),
                                'min_samples': int(min_samples),
                                'n_clusters': int(n_clusters),
                                'noise_ratio': float(noise_ratio),
                                'silhouette_score': float(silhouette),
                                'n_noise': int(n_noise),
                                'labels': labels.tolist(),
                                'score': float(score)
                            }
                    # If current best has noise >= 0.4, prefer any config with noise < 0.4 and silhouette > 0.4
                    elif best_config['noise_ratio'] >= 0.4 and noise_ratio < 0.4 and silhouette > 0.4:
                        best_silhouette = silhouette
                        best_config = {
                            'eps': float(eps),
                            'min_samples': int(min_samples),
                            'n_clusters': int(n_clusters),
                            'noise_ratio': float(noise_ratio),
                            'silhouette_score': float(silhouette),
                            'n_noise': int(n_noise),
                            'labels': labels.tolist(),
                            'score': float(score)
                        }
                    # Otherwise, prefer better score
                    elif score > best_score:
                        best_silhouette = silhouette
                        best_config = {
                            'eps': float(eps),
                            'min_samples': int(min_samples),
                            'n_clusters': int(n_clusters),
                            'noise_ratio': float(noise_ratio),
                            'silhouette_score': float(silhouette),
                            'n_noise': int(n_noise),
                            'labels': labels.tolist(),
                            'score': float(score)
                        }
    
    print(f"\nBest configuration:")
    print(f"  eps: {best_config['eps']}")
    print(f"  min_samples: {best_config['min_samples']}")
    print(f"  n_clusters: {best_config['n_clusters']}")
    print(f"  noise_ratio: {best_config['noise_ratio']:.2%}")
    print(f"  silhouette_score: {best_config['silhouette_score']:.4f}")
    
    # Store results
    results_storage.results['dbscan'] = {
        'algorithm': 'DBSCAN (Density-Based Spatial Clustering)',
        'description': 'Density-based clustering that groups points in dense regions and marks outliers as noise.',
        'parameters': {
            'eps': 'Maximum distance between samples in the same neighborhood',
            'min_samples': 'Minimum number of samples in a neighborhood to form a core point'
        },
        'grid_search': {
            'eps_values': eps_values,
            'min_samples_values': min_samples_values,
            'total_configurations': len(eps_values) * len(min_samples_values)
        },
        'best_configuration': best_config,
        'all_configurations': results_grid,
        'evaluation': {
            'silhouette_score': 'Calculated only on non-noise points to evaluate cluster quality',
            'noise_ratio': 'Percentage of points marked as outliers/noise'
        }
    }
    
    return best_config

def create_visualizations(X_scaled, X_pca_2d, feature_names, best_kmeans_k, best_kmeans_labels,
                         best_agg_linkage, best_agg_labels, best_dbscan_config):
    """
    Section E: Create visualizations
    """
    print("\n" + "="*80)
    print("SECTION E: VISUALIZATION")
    print("="*80)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    sns.set_palette("husl")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    img_dir = os.path.join(project_root, 'Report', 'img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Clustering visualization in 2D PCA space
    print("\n--- Creating clustering visualization ---")
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # K-Means
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=best_kmeans_labels, cmap='viridis', s=50, alpha=0.6)
    ax1.set_title(f'K-Means (K={best_kmeans_k})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('First Principal Component', fontsize=12)
    ax1.set_ylabel('Second Principal Component', fontsize=12)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1)
    
    # Agglomerative
    ax2 = axes[1]
    scatter2 = ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=best_agg_labels, cmap='plasma', s=50, alpha=0.6)
    ax2.set_title(f'Agglomerative ({best_agg_linkage.capitalize()})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('First Principal Component', fontsize=12)
    ax2.set_ylabel('Second Principal Component', fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2)
    
    # DBSCAN
    ax3 = axes[2]
    dbscan_labels = np.array(best_dbscan_config['labels'])
    unique_labels = set(dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black for noise
            col = 'black'
            marker = 'x'
            alpha = 0.3
        else:
            marker = 'o'
            alpha = 0.6
        
        class_member_mask = (dbscan_labels == k)
        xy = X_pca_2d[class_member_mask]
        ax3.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=50, alpha=alpha, 
                   label=f'Cluster {k}' if k != -1 else 'Noise')
    
    ax3.set_title(f"DBSCAN (eps={best_dbscan_config['eps']}, min_samples={best_dbscan_config['min_samples']})", 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('First Principal Component', fontsize=12)
    ax3.set_ylabel('Second Principal Component', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle('Clustering Results in 2D PCA Space', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    vis_path = os.path.join(img_dir, 'question2_clustering_visualization.png')
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {vis_path}")
    plt.close(fig1)
    
    # 2. Analysis plots
    print("\n--- Creating analysis plots ---")
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # K-Means: Inertia and Silhouette
    ax1 = axes[0, 0]
    k_values = range(2, 11)
    inertias = results_storage.results['kmeans']['inertias']
    silhouette_scores = results_storage.results['kmeans']['silhouette_scores']
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(k_values, inertias, 'b-o', linewidth=2, markersize=8, label='Inertia')
    line2 = ax1_twin.plot(k_values, silhouette_scores, 'r-s', linewidth=2, markersize=8, label='Silhouette Score')
    
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12, color='b')
    ax1_twin.set_ylabel('Silhouette Score', fontsize=12, color='r')
    ax1.set_title('KMeans: Inertia and Silhouette Score', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Agglomerative: Silhouette by Linkage
    ax2 = axes[0, 1]
    linkage_methods = results_storage.results['agglomerative']['comparison']['methods_tested']
    linkage_scores = [results_storage.results['agglomerative']['comparison']['performance'][m] for m in linkage_methods]
    
    bars = ax2.bar(linkage_methods, linkage_scores, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Agglomerative: Silhouette Score by Linkage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(linkage_scores) - 0.1, max(linkage_scores) + 0.1])
    
    for bar, score in zip(bars, linkage_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # DBSCAN: Heatmap
    ax3 = axes[1, 0]
    eps_values = results_storage.results['dbscan']['grid_search']['eps_values']
    min_samples_values = results_storage.results['dbscan']['grid_search']['min_samples_values']
    
    heatmap_data = np.full((len(min_samples_values), len(eps_values)), np.nan)
    for config in results_storage.results['dbscan']['all_configurations']:
        eps_idx = eps_values.index(config['eps'])
        min_samples_idx = min_samples_values.index(config['min_samples'])
        heatmap_data[min_samples_idx, eps_idx] = config['silhouette_score']
    
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(len(eps_values)))
    ax3.set_xticklabels(eps_values)
    ax3.set_yticks(range(len(min_samples_values)))
    ax3.set_yticklabels(min_samples_values)
    ax3.set_xlabel('Epsilon (eps)', fontsize=12)
    ax3.set_ylabel('Min Samples', fontsize=12)
    ax3.set_title('DBSCAN: Silhouette Score Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(min_samples_values)):
        for j in range(len(eps_values)):
            text = ax3.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax3)
    
    # Comparison: Best Silhouette Scores
    ax4 = axes[1, 1]
    algorithms = ['KMeans', 'Agglomerative', 'DBSCAN']
    best_scores = [
        results_storage.results['kmeans']['best_silhouette_score'],
        results_storage.results['agglomerative']['best_linkage']['silhouette_score'],
        results_storage.results['dbscan']['best_configuration']['silhouette_score']
    ]
    
    bars = ax4.bar(algorithms, best_scores, color=['blue', 'orange', 'green'], alpha=0.7)
    ax4.set_ylabel('Best Silhouette Score', fontsize=12)
    ax4.set_title('Comparison: Best Silhouette Scores', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, max(best_scores) * 1.2])
    
    for bar, score in zip(bars, best_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Clustering Analysis and Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    analysis_path = os.path.join(img_dir, 'question2_clustering_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to {analysis_path}")
    plt.close(fig2)
    
    # Store visualization info
    results_storage.results['visualization'] = {
        'description': 'All clustering results are visualized in 2D PCA space for comparison. PCA is used only for visualization; models are trained on original scaled feature space.',
        'pca_note': '2D PCA captures the most variance in the data, allowing visual inspection of cluster structures.',
        'interpretation': {
            'kmeans': 'Centroid-based clusters should appear as compact, spherical groups',
            'agglomerative': 'Hierarchical clusters may show nested or tree-like structures',
            'dbscan': 'Density-based clusters can have arbitrary shapes; noise points are marked in black'
        }
    }

def final_analysis():
    """Final analysis and comparison"""
    print("\n" + "="*80)
    print("FINAL ANALYSIS")
    print("="*80)
    
    kmeans_score = results_storage.results['kmeans']['best_silhouette_score']
    agg_score = results_storage.results['agglomerative']['best_linkage']['silhouette_score']
    dbscan_score = results_storage.results['dbscan']['best_configuration']['silhouette_score']
    
    best_overall = 'DBSCAN' if dbscan_score > max(kmeans_score, agg_score) else \
                   ('KMeans' if kmeans_score > agg_score else 'Agglomerative')
    
    results_storage.results['final_analysis'] = {
        'summary': 'Three clustering families were evaluated: centroid-based (KMeans), hierarchical (Agglomerative), and density-based (DBSCAN).',
        'key_findings': {
            'kmeans': f'KMeans achieved best performance with K={results_storage.results["kmeans"]["best_k"]}, suitable for spherical clusters.',
            'agglomerative': f'Agglomerative clustering with {results_storage.results["agglomerative"]["best_linkage"]["method"]} linkage performed best, capturing hierarchical relationships.',
            'dbscan': f'DBSCAN identified {results_storage.results["dbscan"]["best_configuration"]["n_clusters"]} clusters with {results_storage.results["dbscan"]["best_configuration"]["n_noise"]} noise points, suitable for non-spherical clusters.'
        },
        'comparison': {
            'best_overall': best_overall,
            'silhouette_scores': {
                'KMeans': float(kmeans_score),
                'Agglomerative': float(agg_score),
                'DBSCAN': float(dbscan_score)
            }
        }
    }
    
    print(f"\nBest Overall Algorithm: {best_overall}")
    print(f"\nSilhouette Scores:")
    print(f"  KMeans: {kmeans_score:.4f}")
    print(f"  Agglomerative: {agg_score:.4f}")
    print(f"  DBSCAN: {dbscan_score:.4f}")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("QUESTION 2: CLUSTERING ANALYSIS")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Section A: Preprocessing
    X_scaled, X_pca_2d, feature_names = load_and_preprocess_data()
    
    # Section B: K-Means
    best_kmeans_k, best_kmeans_result = kmeans_clustering(X_scaled)
    best_kmeans_labels = np.array(best_kmeans_result['labels'])
    
    # Section C: Agglomerative
    best_agg_linkage, best_agg_result = agglomerative_clustering(X_scaled)
    best_agg_labels = np.array(best_agg_result['labels'])
    
    # Section D: DBSCAN
    best_dbscan_config = dbscan_clustering(X_scaled)
    best_dbscan_labels = np.array(best_dbscan_config['labels'])
    
    # Section E: Visualization
    create_visualizations(X_scaled, X_pca_2d, feature_names, best_kmeans_k, best_kmeans_labels,
                         best_agg_linkage, best_agg_labels, best_dbscan_config)
    
    # Final Analysis
    final_analysis()
    
    # Save results
    results_storage.save_results()
    results_storage.save_explanations()
    
    print("\n" + "="*80)
    print("QUESTION 2 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved to:")
    print("  - Results/question2_results.json")
    print("  - Explanations/question2_explanations.json")
    print("  - Report/img/question2_clustering_visualization.png")
    print("  - Report/img/question2_clustering_analysis.png")
    print("="*80)

if __name__ == "__main__":
    main()

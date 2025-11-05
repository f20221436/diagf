"""
Script to analyze and visualize data augmentation distribution.
Compares original training data with augmented data from fasttext.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import public_function as pf


def parse_fasttext_file(filepath, max_lines=None):
    """
    Parse a fasttext format file and extract label information.
    
    Args:
        filepath: Path to the fasttext file
        max_lines: Maximum number of lines to read (None for all)
    
    Returns:
        dict: Statistics about the data
    """
    label_counts = Counter()
    node_counts = Counter()
    anomaly_counts = Counter()
    sequence_lengths = []
    total_lines = 0
    
    print(f"Reading file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_lines and idx >= max_lines:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Split by tab to separate text and label
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                text = parts[0]
                label = parts[1]
                
                # Extract label information
                if label.startswith('__label__'):
                    label_value = label.replace('__label__', '')
                    label_counts[label_value] += 1
                    
                    # Try to parse node and anomaly indices
                    # Assuming format: __label__{node_idx}{anomaly_idx}
                    if len(label_value) >= 2:
                        try:
                            node_idx = label_value[0]
                            anomaly_idx = label_value[1:]
                            node_counts[node_idx] += 1
                            anomaly_counts[anomaly_idx] += 1
                        except:
                            pass
                
                # Count sequence length (number of events/tokens)
                tokens = text.split()
                sequence_lengths.append(len(tokens))
                
                total_lines += 1
                
                if (idx + 1) % 10000 == 0:
                    print(f"  Processed {idx + 1} lines...")
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"  Total lines read: {total_lines}")
    
    return {
        'total_lines': total_lines,
        'label_counts': dict(label_counts),
        'node_counts': dict(node_counts),
        'anomaly_counts': dict(anomaly_counts),
        'sequence_lengths': sequence_lengths,
        'avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0,
        'median_sequence_length': np.median(sequence_lengths) if sequence_lengths else 0
    }


def plot_anomaly_augmentation(original_stats, augmented_stats, output_dir='plots'):
    """
    Create separate plot for anomaly type augmentation analysis.
    
    Args:
        original_stats: Statistics from original training data
        augmented_stats: Statistics from augmented training data
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Anomaly Type Data Augmentation Analysis', fontsize=16, fontweight='bold')
    
    all_anomalies = sorted(set(list(original_stats['anomaly_counts'].keys()) + 
                              list(augmented_stats['anomaly_counts'].keys())))
    
    # 1. Anomaly type distribution comparison
    ax = axes[0]
    orig_anomaly_counts = [original_stats['anomaly_counts'].get(anom, 0) for anom in all_anomalies]
    aug_anomaly_counts = [augmented_stats['anomaly_counts'].get(anom, 0) for anom in all_anomalies]
    
    x = np.arange(len(all_anomalies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, orig_anomaly_counts, width, label='Original', 
                   alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, aug_anomaly_counts, width, label='Augmented', 
                   alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Anomaly Type Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=13, fontweight='bold')
    ax.set_title('Anomaly Type Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_anomalies, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Anomaly type augmentation increase
    ax = axes[1]
    
    increases = []
    increase_pcts = []
    anomaly_names = []
    
    for anom in all_anomalies:
        orig_count = original_stats['anomaly_counts'].get(anom, 0)
        aug_count = augmented_stats['anomaly_counts'].get(anom, 0)
        increase = aug_count - orig_count
        increase_pct = (increase / orig_count * 100) if orig_count > 0 else 0
        
        increases.append(increase)
        increase_pcts.append(increase_pct)
        anomaly_names.append(f"Type {anom}")
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in increases]
    bars = ax.barh(anomaly_names, increases, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, increase_pcts)):
        width = bar.get_width()
        label_x = width + max(increases) * 0.02 if width >= 0 else width - max(increases) * 0.02
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'+{int(width):,} ({pct:+.1f}%)',
                ha='left' if width >= 0 else 'right', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Increase in Sample Count', fontsize=13, fontweight='bold')
    ax.set_ylabel('Anomaly Type', fontsize=13, fontweight='bold')
    ax.set_title('Augmentation Increase by Anomaly Type', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'anomaly_augmentation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved anomaly augmentation plot to: {output_path}")
    
    plt.close()


def plot_service_augmentation(original_stats, augmented_stats, output_dir='plots'):
    """
    Create separate plot for service/node augmentation analysis.
    
    Args:
        original_stats: Statistics from original training data
        augmented_stats: Statistics from augmented training data
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Service/Node Data Augmentation Analysis', fontsize=16, fontweight='bold')
    
    all_nodes = sorted(set(list(original_stats['node_counts'].keys()) + 
                          list(augmented_stats['node_counts'].keys())))
    
    # 1. Node/Service distribution comparison
    ax = axes[0]
    orig_node_counts = [original_stats['node_counts'].get(node, 0) for node in all_nodes]
    aug_node_counts = [augmented_stats['node_counts'].get(node, 0) for node in all_nodes]
    
    x = np.arange(len(all_nodes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, orig_node_counts, width, label='Original', 
                   alpha=0.8, color='#9b59b6', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, aug_node_counts, width, label='Augmented', 
                   alpha=0.8, color='#f39c12', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Service/Node Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sample Count', fontsize=13, fontweight='bold')
    ax.set_title('Service/Node Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_nodes, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Service/Node augmentation increase
    ax = axes[1]
    
    increases = []
    increase_pcts = []
    node_names = []
    
    for node in all_nodes:
        orig_count = original_stats['node_counts'].get(node, 0)
        aug_count = augmented_stats['node_counts'].get(node, 0)
        increase = aug_count - orig_count
        increase_pct = (increase / orig_count * 100) if orig_count > 0 else 0
        
        increases.append(increase)
        increase_pcts.append(increase_pct)
        node_names.append(f"Service {node}")
    
    colors = ['#16a085' if x >= 0 else '#c0392b' for x in increases]
    bars = ax.barh(node_names, increases, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, increase_pcts)):
        width = bar.get_width()
        label_x = width + max(increases) * 0.02 if width >= 0 else width - max(increases) * 0.02
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'+{int(width):,} ({pct:+.1f}%)',
                ha='left' if width >= 0 else 'right', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Increase in Sample Count', fontsize=13, fontweight='bold')
    ax.set_ylabel('Service/Node', fontsize=13, fontweight='bold')
    ax.set_title('Augmentation Increase by Service/Node', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'service_augmentation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved service augmentation plot to: {output_path}")
    
    plt.close()


def compare_distributions(original_stats, augmented_stats, output_dir='plots'):
    """
    Create visualizations comparing original and augmented data distributions.
    
    Args:
        original_stats: Statistics from original training data
        augmented_stats: Statistics from augmented training data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create separate plots for anomaly and service augmentation
    plot_anomaly_augmentation(original_stats, augmented_stats, output_dir)
    plot_service_augmentation(original_stats, augmented_stats, output_dir)
    
    # Create a comprehensive comparison figure (overview)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Data Augmentation Overview: Original vs Augmented', fontsize=16, fontweight='bold')
    
    # 1. Overall sample count comparison
    ax = axes[0, 0]
    categories = ['Original', 'Augmented']
    counts = [original_stats['total_lines'], augmented_stats['total_lines']]
    bars = ax.bar(categories, counts, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Total Sample Count', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add augmentation ratio
    aug_ratio = augmented_stats['total_lines'] / original_stats['total_lines'] if original_stats['total_lines'] > 0 else 0
    ax.text(0.5, 0.95, f'Augmentation Ratio: {aug_ratio:.2f}x', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # 2. Label distribution comparison
    ax = axes[0, 1]
    all_labels = sorted(set(list(original_stats['label_counts'].keys()) + 
                           list(augmented_stats['label_counts'].keys())))
    
    orig_counts = [original_stats['label_counts'].get(label, 0) for label in all_labels]
    aug_counts = [augmented_stats['label_counts'].get(label, 0) for label in all_labels]
    
    x = np.arange(len(all_labels))
    width = 0.35
    
    ax.bar(x - width/2, orig_counts, width, label='Original', alpha=0.7, color='#3498db', edgecolor='black')
    ax.bar(x + width/2, aug_counts, width, label='Augmented', alpha=0.7, color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Label', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Label Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Sequence length distribution
    ax = axes[1, 0]
    
    if original_stats['sequence_lengths'] and augmented_stats['sequence_lengths']:
        # Sample if too many data points
        orig_lengths = original_stats['sequence_lengths']
        aug_lengths = augmented_stats['sequence_lengths']
        
        if len(orig_lengths) > 10000:
            orig_lengths = np.random.choice(orig_lengths, 10000, replace=False)
        if len(aug_lengths) > 10000:
            aug_lengths = np.random.choice(aug_lengths, 10000, replace=False)
        
        ax.hist(orig_lengths, bins=50, alpha=0.6, label='Original', color='#3498db', edgecolor='black')
        ax.hist(aug_lengths, bins=50, alpha=0.6, label='Augmented', color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel('Sequence Length (# of tokens)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Sequence Length Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Original: μ={original_stats['avg_sequence_length']:.1f}, median={original_stats['median_sequence_length']:.1f}\n"
        stats_text += f"Augmented: μ={augmented_stats['avg_sequence_length']:.1f}, median={augmented_stats['median_sequence_length']:.1f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Augmentation increase per label (heatmap style)
    ax = axes[1, 1]
    
    labels_sorted = sorted(all_labels, key=lambda x: augmented_stats['label_counts'].get(x, 0), reverse=True)[:15]
    
    increases = []
    label_names = []
    
    for label in labels_sorted:
        orig_count = original_stats['label_counts'].get(label, 0)
        aug_count = augmented_stats['label_counts'].get(label, 0)
        increase = aug_count - orig_count
        increases.append(increase)
        label_names.append(label)
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in increases]
    bars = ax.barh(label_names, increases, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Increase in Sample Count', fontsize=12)
    ax.set_ylabel('Label', fontsize=12)
    ax.set_title('Augmentation Increase (Top 15 Labels)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'data_augmentation_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview plot to: {output_path}")
    
    plt.close()
    
    # Create a detailed statistics comparison table
    create_statistics_table(original_stats, augmented_stats, output_dir)


def create_statistics_table(original_stats, augmented_stats, output_dir):
    """
    Create a detailed statistics comparison table and save as CSV.
    """
    # Overall statistics
    overall_data = {
        'Metric': [
            'Total Samples',
            'Unique Labels',
            'Unique Nodes',
            'Unique Anomaly Types',
            'Avg Sequence Length',
            'Median Sequence Length',
            'Min Sequence Length',
            'Max Sequence Length'
        ],
        'Original': [
            original_stats['total_lines'],
            len(original_stats['label_counts']),
            len(original_stats['node_counts']),
            len(original_stats['anomaly_counts']),
            f"{original_stats['avg_sequence_length']:.2f}",
            f"{original_stats['median_sequence_length']:.2f}",
            min(original_stats['sequence_lengths']) if original_stats['sequence_lengths'] else 0,
            max(original_stats['sequence_lengths']) if original_stats['sequence_lengths'] else 0
        ],
        'Augmented': [
            augmented_stats['total_lines'],
            len(augmented_stats['label_counts']),
            len(augmented_stats['node_counts']),
            len(augmented_stats['anomaly_counts']),
            f"{augmented_stats['avg_sequence_length']:.2f}",
            f"{augmented_stats['median_sequence_length']:.2f}",
            min(augmented_stats['sequence_lengths']) if augmented_stats['sequence_lengths'] else 0,
            max(augmented_stats['sequence_lengths']) if augmented_stats['sequence_lengths'] else 0
        ]
    }
    
    # Calculate differences
    differences = []
    for i, metric in enumerate(overall_data['Metric']):
        orig_val = overall_data['Original'][i]
        aug_val = overall_data['Augmented'][i]
        
        try:
            if isinstance(orig_val, str):
                orig_val = float(orig_val)
                aug_val = float(aug_val)
            diff = aug_val - orig_val
            if orig_val != 0:
                pct = (diff / orig_val) * 100
                differences.append(f"{diff:+.2f} ({pct:+.1f}%)")
            else:
                differences.append(f"{diff:+.2f}")
        except:
            differences.append('N/A')
    
    overall_data['Difference'] = differences
    
    df_overall = pd.DataFrame(overall_data)
    
    # Label-wise statistics
    all_labels = sorted(set(list(original_stats['label_counts'].keys()) + 
                           list(augmented_stats['label_counts'].keys())))
    
    label_data = {
        'Label': [],
        'Original_Count': [],
        'Augmented_Count': [],
        'Increase': [],
        'Increase_Percentage': []
    }
    
    for label in all_labels:
        orig_count = original_stats['label_counts'].get(label, 0)
        aug_count = augmented_stats['label_counts'].get(label, 0)
        increase = aug_count - orig_count
        increase_pct = (increase / orig_count * 100) if orig_count > 0 else 0
        
        label_data['Label'].append(label)
        label_data['Original_Count'].append(orig_count)
        label_data['Augmented_Count'].append(aug_count)
        label_data['Increase'].append(increase)
        label_data['Increase_Percentage'].append(f"{increase_pct:.2f}%")
    
    df_labels = pd.DataFrame(label_data)
    df_labels = df_labels.sort_values('Augmented_Count', ascending=False)
    
    # Save to CSV
    overall_path = os.path.join(output_dir, 'augmentation_overall_statistics.csv')
    labels_path = os.path.join(output_dir, 'augmentation_label_statistics.csv')
    
    df_overall.to_csv(overall_path, index=False)
    df_labels.to_csv(labels_path, index=False)
    
    print(f"Saved overall statistics to: {overall_path}")
    print(f"Saved label statistics to: {labels_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("DATA AUGMENTATION SUMMARY")
    print("="*80)
    print(df_overall.to_string(index=False))
    print("\n" + "="*80)
    print("TOP 10 LABELS BY AUGMENTED COUNT")
    print("="*80)
    print(df_labels.head(10).to_string(index=False))
    print("="*80 + "\n")


def main():
    """
    Main function to analyze data augmentation.
    """
    # Define paths (adjust these to your actual paths)
    fasttext_dir = os.path.join('fasttext', 'fasttext_temp')
    train_original_path = os.path.join(fasttext_dir, 'train.txt')
    train_augmented_path = os.path.join(fasttext_dir, 'train_da.txt')
    
    print("="*80)
    print("Data Augmentation Analysis Tool")
    print("="*80)
    print(f"Original data path: {train_original_path}")
    print(f"Augmented data path: {train_augmented_path}")
    print("="*80 + "\n")
    
    # Check if files exist
    if not os.path.exists(train_original_path):
        print(f"ERROR: Original training file not found: {train_original_path}")
        return
    
    if not os.path.exists(train_augmented_path):
        print(f"ERROR: Augmented training file not found: {train_augmented_path}")
        return
    
    # Parse both files
    print("\n[1/3] Parsing original training data...")
    original_stats = parse_fasttext_file(train_original_path)
    
    if original_stats is None:
        print("ERROR: Failed to parse original training data")
        return
    
    print("\n[2/3] Parsing augmented training data...")
    augmented_stats = parse_fasttext_file(train_augmented_path)
    
    if augmented_stats is None:
        print("ERROR: Failed to parse augmented training data")
        return
    
    # Compare and visualize
    print("\n[3/3] Creating visualizations and statistics...")
    compare_distributions(original_stats, augmented_stats)
    
    print("\n" + "="*80)
    print("Analysis complete! Generated outputs:")
    print("  1. anomaly_augmentation_analysis.png - Anomaly type augmentation details")
    print("  2. service_augmentation_analysis.png - Service/node augmentation details")
    print("  3. data_augmentation_overview.png - Overall augmentation summary")
    print("  4. augmentation_overall_statistics.csv - Summary statistics")
    print("  5. augmentation_label_statistics.csv - Label-wise statistics")
    print("Check the 'plots' directory for all outputs.")
    print("="*80)


if __name__ == "__main__":
    main()

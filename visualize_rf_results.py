"""
Visualize Random Forest model results:
- R² performance by anchor and group
- Feature importance comparisons across groups and anchors
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = '/Users/rebekahzhang/data/behavior_data'
EXP = "exp2"
RESULTS_DIR = os.path.join(DATA_DIR, f'{EXP}_modeling_by_group/results')
MODELS_DIR = os.path.join(DATA_DIR, f'{EXP}_modeling_by_group/models')
FIG_DIR = os.path.join(DATA_DIR, f'{EXP}_modeling_by_group/figures')

REWARD_FEATURES = [
    'previous_trial_reward', 'rewarded_streak', 'unrewarded_streak',
    'time_since_last_reward_in_block', 'cumulative_reward_in_block',
    'cumulative_reward', 'reward_rate_since_block_start',
    'reward_rate_past_1min_in_block', 'reward_rate_past_5min_in_block',
    'reward_rate_past_10min_in_block'
]

os.makedirs(FIG_DIR, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'

# Colors
COLORS = {
    'l': '#9080ff',  # Long - purple
    's': '#ffb400',  # Short - yellow/orange
    'Long_BG': '#9080ff',
    'Short_BG': '#ffb400'
}

ANCHOR_LABELS = {
    'cue_on': 'Cue On',
    'cue_off': 'Cue Off',
    'last_lick': 'Last Lick'
}

GROUP_LABELS = {
    'l': 'Long',
    's': 'Short',
    'Long_BG': 'Long',
    'Short_BG': 'Short'
}

# ============================================================================
# LOAD DATA
# ============================================================================

def load_results():
    """Load all result files."""
    results = {}
    
    # Main comparison file
    results['model_comparison'] = pd.read_csv(
        os.path.join(RESULTS_DIR, 'model_comparison_all.csv')
    )
    
    # R² summary
    results['r2_summary'] = pd.read_csv(
        os.path.join(RESULTS_DIR, 'r2_summary_by_anchor.csv'),
        index_col=0
    )
    
    # Top features
    results['top_features'] = pd.read_csv(
        os.path.join(RESULTS_DIR, 'top_15_features_all_conditions.csv')
    )
    
    # Feature importances per condition
    results['fi'] = {}
    for group in ['l', 's']:
        for anchor in ['cue_on', 'cue_off', 'last_lick']:
            for transform in ['raw', 'log']:
                key = f'{group}_{anchor}_{transform}'
                filepath = os.path.join(RESULTS_DIR, f'feature_importances_{key}.csv')
                if os.path.exists(filepath):
                    results['fi'][key] = pd.read_csv(filepath)
    
    # Group comparisons
    results['fi_comparison'] = {}
    for anchor in ['cue_on', 'cue_off', 'last_lick']:
        for transform in ['raw', 'log']:
            filepath = os.path.join(RESULTS_DIR, f'fi_group_comparison_{anchor}_{transform}.csv')
            if os.path.exists(filepath):
                results['fi_comparison'][f'{anchor}_{transform}'] = pd.read_csv(filepath)

    # no_wait_history feature importances — loaded from saved model pkl files
    # (only 'all' feature set FIs were saved as CSVs; others need model loading)
    results['fi_no_wait'] = {}
    for group in ['l', 's']:
        for anchor in ['cue_on', 'cue_off', 'last_lick']:
            for transform in ['raw', 'log']:
                key = f'{group}_{anchor}_{transform}'
                filepath = os.path.join(MODELS_DIR, f'rf_{group}_{anchor}_{transform}_no_wait_history.pkl')
                if os.path.exists(filepath):
                    saved = joblib.load(filepath)
                    fi_df = pd.DataFrame({
                        'feature': saved['feature_names'],
                        'importance': saved['model'].feature_importances_
                    }).sort_values('importance', ascending=False).reset_index(drop=True)
                    results['fi_no_wait'][key] = fi_df

    return results


# ============================================================================
# FIGURE 1: R² BY ANCHOR AND GROUP
# ============================================================================

def plot_r2_by_anchor(results, save=True):
    """Bar plot of R² for each anchor, comparing groups."""
    
    df = results['model_comparison']
    df_all = df[df['feature_set'] == 'all'].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, transform in enumerate(['raw', 'log']):
        ax = axes[idx]
        subset = df_all[df_all['transform'] == transform]
        
        # Pivot for plotting
        pivot = subset.pivot(index='anchor', columns='group', values='r2')
        pivot = pivot.reindex(['cue_on', 'cue_off', 'last_lick'])
        
        x = np.arange(len(pivot))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pivot['l'], width, 
                       label='Long BG', color=COLORS['l'], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, pivot['s'], width, 
                       label='Short BG', color=COLORS['s'], edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Timing Anchor')
        ax.set_ylabel('R²')
        ax.set_title(f'{"Raw" if transform == "raw" else "Log-Transformed"} Target')
        ax.set_xticks(x)
        ax.set_xticklabels([ANCHOR_LABELS[a] for a in pivot.index])
        ax.legend(loc='upper left')
        ax.set_ylim(0, 1.0)
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    plt.suptitle('Model Performance (R²) by Timing Anchor and Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, 'r2_by_anchor_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, 'r2_by_anchor_comparison.png'), bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================================
# FIGURE 2: R² DIFFERENCE BETWEEN GROUPS
# ============================================================================

def plot_r2_difference(results, save=True):
    """Plot R² difference (Long BG - Short BG) by anchor."""
    
    df = results['model_comparison']
    df_all = df[df['feature_set'] == 'all'].copy()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    anchors = ['cue_on', 'cue_off', 'last_lick']
    x = np.arange(len(anchors))
    width = 0.35
    
    diffs_raw = []
    diffs_log = []
    
    for anchor in anchors:
        for transform in ['raw', 'log']:
            subset = df_all[(df_all['anchor'] == anchor) & (df_all['transform'] == transform)]
            r2_long = subset[subset['group'] == 'l']['r2'].values[0]
            r2_short = subset[subset['group'] == 's']['r2'].values[0]
            diff = r2_long - r2_short
            
            if transform == 'raw':
                diffs_raw.append(diff)
            else:
                diffs_log.append(diff)
    
    bars1 = ax.bar(x - width/2, diffs_raw, width, label='Raw', color='#5D9CEC', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, diffs_log, width, label='Log', color='#AC92EB', edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{height:+.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, offset), textcoords="offset points",
                       ha='center', va=va, fontsize=9)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Timing Anchor')
    ax.set_ylabel('R² Difference (Long BG − Short BG)')
    ax.set_title('Group Difference in Predictability by Anchor', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([ANCHOR_LABELS[a] for a in anchors])
    ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add annotation
    ax.text(0.02, 0.98, 'Positive = Long BG more predictable\nNegative = Short BG more predictable',
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, 'r2_difference_by_anchor.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, 'r2_difference_by_anchor.png'), bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================================
# FIGURE 3: TOP FEATURE IMPORTANCES BY GROUP AND ANCHOR
# ============================================================================

def plot_feature_importance_grid(results, transform='raw', top_n=10, save=True):
    """Grid of feature importance plots for each group × anchor combination."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    groups = ['l', 's']
    anchors = ['cue_on', 'cue_off', 'last_lick']
    
    for i, group in enumerate(groups):
        for j, anchor in enumerate(anchors):
            ax = axes[i, j]
            key = f'{group}_{anchor}_{transform}'
            
            if key in results['fi']:
                fi = results['fi'][key].head(top_n).copy()
                fi = fi.iloc[::-1]  # Reverse for horizontal bar plot
                
                # Shorten feature names for display
                fi['feature_short'] = fi['feature'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
                
                bars = ax.barh(fi['feature_short'], fi['importance'], 
                              color=COLORS[group], edgecolor='black', linewidth=0.5)
                
                ax.set_xlabel('Importance')
                ax.set_title(f'{GROUP_LABELS[group]} | {ANCHOR_LABELS[anchor]}')
                ax.set_xlim(0, max(fi['importance']) * 1.15)
                
                # Add value labels
                for bar, val in zip(bars, fi['importance']):
                    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                           f'{val:.3f}', va='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{GROUP_LABELS[group]} | {ANCHOR_LABELS[anchor]}')
    
    plt.suptitle(f'Top {top_n} Feature Importances ({transform.capitalize()} Target)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_grid_{transform}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_grid_{transform}.png'), bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================================
# FIGURE 4: FEATURE IMPORTANCE COMPARISON BETWEEN GROUPS
# ============================================================================

def plot_feature_importance_comparison(results, anchor='cue_off', transform='raw', top_n=15, save=True):
    """Side-by-side comparison of feature importances between groups."""
    
    key = f'{anchor}_{transform}'
    if key not in results['fi_comparison']:
        print(f"No comparison data for {key}")
        return None
    
    df = results['fi_comparison'][key].head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y = np.arange(len(df))
    height = 0.35
    
    # Sort by absolute difference
    df = df.sort_values('abs_diff', ascending=True)
    
    # Shorten feature names
    df['feature_short'] = df['feature'].apply(lambda x: x[:35] + '...' if len(x) > 35 else x)
    
    bars1 = ax.barh(y - height/2, df['importance_long'], height, 
                    label='Long BG', color=COLORS['l'], edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y + height/2, df['importance_short'], height, 
                    label='Short BG', color=COLORS['s'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'Feature Importance Comparison: {ANCHOR_LABELS[anchor]} ({transform.capitalize()})\n'
                 f'(Sorted by largest group difference)', fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(df['feature_short'])
    ax.legend(loc='lower right')
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_comparison_{anchor}_{transform}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_comparison_{anchor}_{transform}.png'), 
                   bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================================
# FIGURE 5: FEATURE IMPORTANCE ACROSS ANCHORS (WITHIN GROUP)
# ============================================================================

def plot_feature_importance_across_anchors(results, group='l', transform='raw', top_n=10, save=True):
    """Show how feature importance changes across anchors for one group."""
    
    anchors = ['cue_on', 'cue_off', 'last_lick']
    
    # Get top features from 'all' anchor (or union of top features)
    all_features = set()
    for anchor in anchors:
        key = f'{group}_{anchor}_{transform}'
        if key in results['fi']:
            all_features.update(results['fi'][key].head(top_n)['feature'].tolist())
    
    # Build dataframe
    data = []
    for anchor in anchors:
        key = f'{group}_{anchor}_{transform}'
        if key in results['fi']:
            fi = results['fi'][key].set_index('feature')
            for feat in all_features:
                imp = fi.loc[feat, 'importance'] if feat in fi.index else 0
                data.append({'feature': feat, 'anchor': anchor, 'importance': imp})
    
    df = pd.DataFrame(data)
    
    # Get mean importance for sorting
    mean_imp = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    top_features = mean_imp.head(top_n).index.tolist()
    df = df[df['feature'].isin(top_features)]
    
    # Pivot for heatmap
    pivot = df.pivot(index='feature', columns='anchor', values='importance')
    pivot = pivot.reindex(top_features)
    pivot = pivot[anchors]  # Ensure order
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Importance'}, linewidths=0.5)
    
    ax.set_xlabel('Timing Anchor')
    ax.set_ylabel('Feature')
    ax.set_xticklabels([ANCHOR_LABELS[a] for a in anchors])
    ax.set_title(f'{GROUP_LABELS[group]}: Feature Importance Across Anchors ({transform.capitalize()})',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        group_name = 'long' if group == 'l' else 'short'
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_heatmap_{group_name}_{transform}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_heatmap_{group_name}_{transform}.png'), 
                   bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================================
# FIGURE 6: ABLATION ANALYSIS
# ============================================================================

def plot_ablation(results, anchor='cue_off', transform='raw', save=True):
    """Show R² when each feature set is removed."""
    
    df = results['model_comparison']
    subset = df[(df['anchor'] == anchor) & (df['transform'] == transform)].copy()
    
    # Pivot
    pivot = subset.pivot(index='feature_set', columns='group', values='r2')
    pivot.columns = ['Long BG', 'Short BG']
    
    # Sort by Long BG R²
    pivot = pivot.sort_values('Long BG', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y = np.arange(len(pivot))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, pivot['Long BG'], height, 
                    label='Long BG', color=COLORS['l'], edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y + height/2, pivot['Short BG'], height, 
                    label='Short BG', color=COLORS['s'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('R²')
    ax.set_ylabel('Feature Set')
    ax.set_title(f'Feature Set Ablation: {ANCHOR_LABELS[anchor]} ({transform.capitalize()})',
                fontsize=12, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.legend(loc='lower right')
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1.0)
    
    # Add baseline reference
    baseline = pivot.loc['all', 'Long BG']
    ax.axvline(x=baseline, color=COLORS['l'], linestyle='--', alpha=0.5, label='_nolegend_')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, f'ablation_{anchor}_{transform}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, f'ablation_{anchor}_{transform}.png'), 
                   bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================================
# FIGURE 7: SUMMARY PANEL
# ============================================================================

def plot_summary_panel(results, save=True):
    """Create a comprehensive summary figure."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    df = results['model_comparison']
    df_all = df[df['feature_set'] == 'all'].copy()
    
    # Panel A: R² by anchor (raw)
    ax1 = fig.add_subplot(gs[0, 0])
    subset = df_all[df_all['transform'] == 'raw']
    pivot = subset.pivot(index='anchor', columns='group', values='r2')
    pivot = pivot.reindex(['cue_on', 'cue_off', 'last_lick'])
    
    x = np.arange(len(pivot))
    width = 0.35
    ax1.bar(x - width/2, pivot['l'], width, label='Long BG', color=COLORS['l'], edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, pivot['s'], width, label='Short BG', color=COLORS['s'], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('R²')
    ax1.set_title('A. R² by Anchor (Raw)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([ANCHOR_LABELS[a] for a in pivot.index], rotation=15)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1.0)
    
    # Panel B: R² by anchor (log)
    ax2 = fig.add_subplot(gs[0, 1])
    subset = df_all[df_all['transform'] == 'log']
    pivot = subset.pivot(index='anchor', columns='group', values='r2')
    pivot = pivot.reindex(['cue_on', 'cue_off', 'last_lick'])
    
    ax2.bar(x - width/2, pivot['l'], width, label='Long BG', color=COLORS['l'], edgecolor='black', linewidth=0.5)
    ax2.bar(x + width/2, pivot['s'], width, label='Short BG', color=COLORS['s'], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('R²')
    ax2.set_title('B. R² by Anchor (Log)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([ANCHOR_LABELS[a] for a in pivot.index], rotation=15)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.0)
    
    # Panel C: R² difference
    ax3 = fig.add_subplot(gs[0, 2])
    anchors = ['cue_on', 'cue_off', 'last_lick']
    diffs = []
    for anchor in anchors:
        subset = df_all[(df_all['anchor'] == anchor) & (df_all['transform'] == 'raw')]
        r2_long = subset[subset['group'] == 'l']['r2'].values[0]
        r2_short = subset[subset['group'] == 's']['r2'].values[0]
        diffs.append(r2_long - r2_short)
    
    colors = ['#E74C3C' if d < 0 else '#27AE60' for d in diffs]
    ax3.bar(range(len(anchors)), diffs, color=colors, edgecolor='black', linewidth=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_ylabel('ΔR² (Long − Short)')
    ax3.set_title('C. Group Difference', fontweight='bold')
    ax3.set_xticks(range(len(anchors)))
    ax3.set_xticklabels([ANCHOR_LABELS[a] for a in anchors], rotation=15)
    
    # Panels D-F: Top 5 features for each anchor (raw, combined groups)
    for idx, anchor in enumerate(['cue_on', 'cue_off', 'last_lick']):
        ax = fig.add_subplot(gs[1, idx])
        
        # Get Long BG features
        key_l = f'l_{anchor}_raw'
        key_s = f's_{anchor}_raw'
        
        if key_l in results['fi'] and key_s in results['fi']:
            fi_l = results['fi'][key_l].head(5).copy()
            fi_s = results['fi'][key_s].head(5).copy()
            
            # Combine and get union of top features
            all_feats = list(dict.fromkeys(fi_l['feature'].tolist() + fi_s['feature'].tolist()))[:7]
            
            fi_l_dict = dict(zip(fi_l['feature'], fi_l['importance']))
            fi_s_dict = dict(zip(fi_s['feature'], fi_s['importance']))
            
            y = np.arange(len(all_feats))
            height = 0.35
            
            vals_l = [fi_l_dict.get(f, 0) for f in all_feats]
            vals_s = [fi_s_dict.get(f, 0) for f in all_feats]
            
            ax.barh(y - height/2, vals_l, height, label='Long BG', color=COLORS['l'], edgecolor='black', linewidth=0.5)
            ax.barh(y + height/2, vals_s, height, label='Short BG', color=COLORS['s'], edgecolor='black', linewidth=0.5)
            
            # Shorten labels
            short_labels = [f[:25] + '..' if len(f) > 25 else f for f in all_feats]
            ax.set_yticks(y)
            ax.set_yticklabels(short_labels, fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f'D{idx+1}. Top Features: {ANCHOR_LABELS[anchor]}', fontweight='bold')
            if idx == 0:
                ax.legend(fontsize=8)
    
    # Panel G: Ablation for cue_off
    ax7 = fig.add_subplot(gs[2, :2])
    subset = df[(df['anchor'] == 'cue_off') & (df['transform'] == 'raw')].copy()
    pivot = subset.pivot(index='feature_set', columns='group', values='r2')
    pivot.columns = ['Long BG', 'Short BG']
    pivot = pivot.sort_values('Long BG', ascending=True)
    
    y = np.arange(len(pivot))
    height = 0.35
    ax7.barh(y - height/2, pivot['Long BG'], height, label='Long BG', color=COLORS['l'], edgecolor='black', linewidth=0.5)
    ax7.barh(y + height/2, pivot['Short BG'], height, label='Short BG', color=COLORS['s'], edgecolor='black', linewidth=0.5)
    ax7.set_xlabel('R²')
    ax7.set_title('G. Feature Set Ablation (Cue Offset, Raw)', fontweight='bold')
    ax7.set_yticks(y)
    ax7.set_yticklabels(pivot.index)
    ax7.legend(loc='lower right', fontsize=9)
    ax7.set_xlim(0, 0.5)
    
    # Panel H: Text summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = """Key Findings:
    
1. Both groups best predicted 
   from last_lick anchor
   
2. Long BG more predictable at
   last_lick (R²=0.84 vs 0.70)
   
3. previous_trial_miss_trial
   dominates at last_lick anchor
   
4. time_waited_rolling_mean
   dominates at cue_on/cue_off
   
5. Removing wait_history or
   reward features has largest
   impact on R²"""
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax8.set_title('H. Summary', fontweight='bold')
    
    plt.suptitle('Random Forest Model Results Summary', fontsize=16, fontweight='bold', y=1.02)
    
    if save:
        plt.savefig(os.path.join(FIG_DIR, 'summary_panel.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIG_DIR, 'summary_panel.png'), bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================================
# FIGURE 8: TOP FEATURE IMPORTANCES — NO WAIT HISTORY FEATURE SET
# ============================================================================

def plot_feature_importance_grid_no_wait(results, transform='raw', top_n=10, save=True):
    """Top feature importances from models trained without wait history features."""

    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    groups = ['l', 's']
    anchors = ['cue_on', 'cue_off', 'last_lick']

    for i, group in enumerate(groups):
        for j, anchor in enumerate(anchors):
            ax = axes[i, j]
            key = f'{group}_{anchor}_{transform}'

            if key in results['fi_no_wait']:
                fi = results['fi_no_wait'][key].head(top_n).copy()
                fi = fi.iloc[::-1]
                fi['feature_short'] = fi['feature'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

                bars = ax.barh(fi['feature_short'], fi['importance'],
                               color=COLORS[group], edgecolor='black', linewidth=0.5)

                ax.set_xlabel('Importance')
                ax.set_title(f'{GROUP_LABELS[group]} | {ANCHOR_LABELS[anchor]}')
                ax.set_xlim(0, max(fi['importance']) * 1.15)

                for bar, val in zip(bars, fi['importance']):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                            f'{val:.3f}', va='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{GROUP_LABELS[group]} | {ANCHOR_LABELS[anchor]}')

    plt.suptitle(f'Top {top_n} Feature Importances — No Wait History ({transform.capitalize()} Target)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_grid_no_wait_{transform}.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# FIGURE 9: FEATURE IMPORTANCE HEATMAP — NO WAIT HISTORY FEATURE SET
# ============================================================================

def plot_feature_importance_heatmap_no_wait(results, group='l', transform='raw', top_n=10, save=True):
    """Heatmap of feature importance across anchors, for no_wait_history models."""

    anchors = ['cue_on', 'cue_off', 'last_lick']

    all_features = set()
    for anchor in anchors:
        key = f'{group}_{anchor}_{transform}'
        if key in results['fi_no_wait']:
            all_features.update(results['fi_no_wait'][key].head(top_n)['feature'].tolist())

    data = []
    for anchor in anchors:
        key = f'{group}_{anchor}_{transform}'
        if key in results['fi_no_wait']:
            fi = results['fi_no_wait'][key].set_index('feature')
            for feat in all_features:
                imp = fi.loc[feat, 'importance'] if feat in fi.index else 0
                data.append({'feature': feat, 'anchor': anchor, 'importance': imp})

    df = pd.DataFrame(data)
    mean_imp = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    top_features = mean_imp.head(top_n).index.tolist()
    df = df[df['feature'].isin(top_features)]

    pivot = df.pivot(index='feature', columns='anchor', values='importance')
    pivot = pivot.reindex(top_features)[anchors]

    _, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Importance'}, linewidths=0.5)

    ax.set_xlabel('Timing Anchor')
    ax.set_ylabel('Feature')
    ax.set_xticklabels([ANCHOR_LABELS[a] for a in anchors])
    ax.set_title(f'{GROUP_LABELS[group]}: Feature Importance Across Anchors\n'
                 f'No Wait History ({transform.capitalize()})',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save:
        group_name = 'long' if group == 'l' else 'short'
        plt.savefig(os.path.join(FIG_DIR, f'feature_importance_heatmap_no_wait_{group_name}_{transform}.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# FIGURE 10: REWARD FEATURES EXPLORATION
# ============================================================================

def plot_reward_features(results, transform='raw', save=True):
    """
    Two panels:
      A. Heatmap of each reward feature's importance across all 6 conditions
         (2 groups × 3 anchors), from the 'all' feature set models.
      B. R² comparison: 'all' vs 'only_reward' vs 'no_reward' feature sets,
         showing how much reward features contribute.
    """

    anchors = ['cue_on', 'cue_off', 'last_lick']
    groups = ['l', 's']

    # ---- Panel A: reward feature importance heatmap ----
    # Build columns as "Long BG / Cue On" etc.
    col_labels = []
    heatmap_data = {f: [] for f in REWARD_FEATURES}

    for group in groups:
        for anchor in anchors:
            col_labels.append(f'{GROUP_LABELS[group]}\n{ANCHOR_LABELS[anchor]}')
            key = f'{group}_{anchor}_{transform}'
            fi = results['fi'].get(key)
            fi_dict = dict(zip(fi['feature'], fi['importance'])) if fi is not None else {}
            for feat in REWARD_FEATURES:
                heatmap_data[feat].append(fi_dict.get(feat, 0))

    heatmap_df = pd.DataFrame(heatmap_data, index=col_labels).T
    # Sort rows by mean importance
    heatmap_df = heatmap_df.loc[heatmap_df.mean(axis=1).sort_values(ascending=False).index]

    # ---- Panel B: R² for all / no_reward / only_reward ----
    df = results['model_comparison']
    df_sub = df[df['transform'] == transform].copy()

    _, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel A
    ax = axes[0]
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Importance'}, linewidths=0.5)
    ax.set_title(f'A. Reward Feature Importances Across Conditions\n({transform.capitalize()} target, all-features model)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Reward Feature')
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Panel B: grouped bars — one group of bars per anchor, three bars per group
    ax = axes[1]
    feature_sets_to_compare = ['all', 'only_reward', 'no_reward']
    fs_colors = {'all': '#555555', 'only_reward': '#E67E22', 'no_reward': '#3498DB'}
    fs_labels = {'all': 'All features', 'only_reward': 'Only reward', 'no_reward': 'No reward'}

    x = np.arange(len(anchors))
    n_fs = len(feature_sets_to_compare)
    width = 0.22
    offsets = np.linspace(-(n_fs-1)/2 * width, (n_fs-1)/2 * width, n_fs)

    for group in groups:
        for fi, (fs, offset) in enumerate(zip(feature_sets_to_compare, offsets)):
            r2_vals = []
            for anchor in anchors:
                row = df_sub[(df_sub['group'] == group) & (df_sub['anchor'] == anchor) &
                             (df_sub['feature_set'] == fs)]
                r2_vals.append(row['r2'].values[0] if len(row) > 0 else 0)

            label = f'{GROUP_LABELS[group]} — {fs_labels[fs]}'
            hatch = '' if group == 'l' else '//'
            ax.bar(x + offset + (0.01 if group == 's' else -0.01),
                   r2_vals, width * 0.9,
                   label=label, color=fs_colors[fs],
                   alpha=0.85 if group == 'l' else 0.5,
                   edgecolor='black', linewidth=0.5, hatch=hatch)

    ax.set_xlabel('Timing Anchor')
    ax.set_ylabel('R²')
    ax.set_title(f'B. R²: All vs Only Reward vs No Reward Features\n({transform.capitalize()} target)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([ANCHOR_LABELS[a] for a in anchors])
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.suptitle('Role of Reward Features in Waiting Behavior Prediction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(FIG_DIR, f'reward_features_{transform}.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading results...")
    results = load_results()
    
    print(f"\nGenerating figures in: {FIG_DIR}")
    print("-" * 50)
    
    print("1. R² by anchor comparison...")
    plot_r2_by_anchor(results)
    
    print("2. R² difference between groups...")
    plot_r2_difference(results)
    
    print("3. Feature importance grid (raw)...")
    plot_feature_importance_grid(results, transform='raw')
    
    print("4. Feature importance grid (log)...")
    plot_feature_importance_grid(results, transform='log')
    
    print("5. Feature importance comparison (cue_on)...")
    plot_feature_importance_comparison(results, anchor='cue_on', transform='raw')

    print("6. Feature importance comparison (cue_off)...")
    plot_feature_importance_comparison(results, anchor='cue_off', transform='raw')

    print("7. Feature importance comparison (last_lick)...")
    plot_feature_importance_comparison(results, anchor='last_lick', transform='raw')

    print("8. Feature importance heatmap (Long BG)...")
    plot_feature_importance_across_anchors(results, group='l', transform='raw')

    print("9. Feature importance heatmap (Short BG)...")
    plot_feature_importance_across_anchors(results, group='s', transform='raw')

    print("10. Ablation (cue_on)...")
    plot_ablation(results, anchor='cue_on', transform='raw')

    print("11. Ablation (cue_off)...")
    plot_ablation(results, anchor='cue_off', transform='raw')

    print("12. Ablation (last_lick)...")
    plot_ablation(results, anchor='last_lick', transform='raw')

    print("13. Feature importance grid — no wait history (raw)...")
    plot_feature_importance_grid_no_wait(results, transform='raw')

    print("14. Feature importance grid — no wait history (log)...")
    plot_feature_importance_grid_no_wait(results, transform='log')

    print("15. Feature importance heatmap — no wait history (Long BG)...")
    plot_feature_importance_heatmap_no_wait(results, group='l', transform='raw')

    print("16. Feature importance heatmap — no wait history (Short BG)...")
    plot_feature_importance_heatmap_no_wait(results, group='s', transform='raw')

    print("17. Reward features exploration (raw)...")
    plot_reward_features(results, transform='raw')

    print("18. Reward features exploration (log)...")
    plot_reward_features(results, transform='log')

    print("19. Summary panel...")
    plot_summary_panel(results)
    
    print("-" * 50)
    print(f"All figures saved to: {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()

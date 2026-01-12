import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cycler, font_manager
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font to Times New Roman (same method as shap_explain.py)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Create a FontProperties pointing to the system Times New Roman TTF (if available)
try:
    tn_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))
    font_prop = font_manager.FontProperties(fname=tn_path)
except Exception:
    font_prop = font_manager.FontProperties(family='Times New Roman')

# Modern clean color palette
# 现代简洁配色方案
colors = {
    'primary': '#2E8B8B',      # 柔和的青绿色
    'secondary': '#F5F5F5',    # 浅灰色背景
    'accent': '#4682B4',       # 钢蓝色
    'text': '#2F4F4F',         # 深灰绿色
    'light': '#E6F3F3',        # 浅青绿色
    'warning': '#CD853F'       # 秘鲁色
}

# Set Matplotlib style and color cycle (replace Seaborn palette)
plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = cycler('color', [
    colors['primary'], colors['accent'], colors['warning'], '#87CEEB', '#DDA0DD'
])

# Style-like settings similar to seaborn whitegrid
plt.rcParams.update({
    'axes.edgecolor': colors['text'],
    'axes.labelcolor': colors['text'],
    'text.color': colors['text'],
    'grid.color': '#DDDDDD',
    'grid.linestyle': '-',
    'grid.alpha': 0.3
})

print("=== Generating visualization charts ===")

# 读取数据
final_results = pd.read_csv('feature_selection_results.csv')
f_results = pd.read_csv('f_score_results.csv')
rf_results = pd.read_csv('rf_importance_results.csv')
lasso_results = pd.read_csv('lasso_results.csv')

# Merge p-values for visualization (for adding stars)
if 'p_value' not in rf_results.columns:
    rf_results = pd.merge(rf_results, f_results[['Feature', 'p_value']], on='Feature', how='left')
if 'p_value' not in lasso_results.columns:
    lasso_results = pd.merge(lasso_results, f_results[['Feature', 'p_value']], on='Feature', how='left')

X_scaled = pd.read_csv('X_scaled.csv')
y_os = pd.read_csv('y_os.csv', header=None, names=['OS'])['OS']

print("Data loaded successfully.")

# Figure 1: Horizontal bar chart of top 15 features by composite score
plt.figure(figsize=(12, 8))
top_15 = final_results.head(15)
bars = plt.barh(range(len(top_15)), top_15['Composite_score'], 
                color=colors['primary'], alpha=0.8, edgecolor=colors['text'], linewidth=0.5)

plt.yticks(range(len(top_15)), top_15['Feature'], fontsize=11, fontproperties=font_prop)
plt.xlabel('Composite Score', fontsize=12, color=colors['text'], fontproperties=font_prop)
plt.title('Top 15 Features: Composite Importance', fontsize=14, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color(colors['text'])
plt.gca().spines['bottom'].set_color(colors['text'])
plt.grid(axis='x', alpha=0.3, color=colors['text'])

# Figure 2: Radar chart comparing three feature selection methods (top 8 features)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Select top 8 features
top_8 = final_results.head(8)
features = top_8['Feature'].tolist()

# Normalized scores for radar
f_scores_norm = top_8['F_score_norm'].tolist()
rf_importance_norm = top_8['RF_importance_norm'].tolist()
lasso_coef_norm = top_8['LASSO_coef_norm'].tolist()

# Angles for radar
angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

# Close the loops
f_scores_norm += f_scores_norm[:1]
rf_importance_norm += rf_importance_norm[:1]
lasso_coef_norm += lasso_coef_norm[:1]

# Plot radar
ax.plot(angles, f_scores_norm, 'o-', linewidth=2, label='F-score Analysis', color=colors['primary'])
ax.fill(angles, f_scores_norm, alpha=0.25, color=colors['primary'])

ax.plot(angles, rf_importance_norm, 's-', linewidth=2, label='Random Forest', color=colors['accent'])
ax.fill(angles, rf_importance_norm, alpha=0.25, color=colors['accent'])

ax.plot(angles, lasso_coef_norm, '^-', linewidth=2, label='LASSO Regression', color=colors['warning'])
ax.fill(angles, lasso_coef_norm, alpha=0.25, color=colors['warning'])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=10, fontproperties=font_prop)
ax.set_ylim(0, 1)
ax.set_title('Comparison of Three Feature Selection Methods (Top 8 Features)', fontsize=14, fontweight='bold', 
             color=colors['text'], pad=20, fontproperties=font_prop)

plt.legend(prop=font_prop, loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('output/radar_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()




# Figure 3: Correlation heatmap
print("Creating correlation heatmap...")

# 选择前15个最重要的特征进行相关性分析
top_15_features = final_results.head(15)['Feature'].tolist()
X_top15 = X_scaled[top_15_features]

# 计算相关性矩阵
correlation_matrix = X_top15.corr()

# Create correlation heatmap using Matplotlib (mask upper triangle and annotate)
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Use a diverging colormap from Matplotlib
cmap = plt.cm.RdBu_r

# Convert to numpy array and mask
corr_vals = correlation_matrix.values
masked_corr = np.ma.array(corr_vals, mask=mask)

im = ax.imshow(masked_corr, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')

# Colorbar for heatmap (use separate variable to avoid overwriting)
cbar_heatmap = fig.colorbar(im, ax=ax, shrink=0.8)
# set font for colorbar label and ticks
cbar_heatmap.set_label('Correlation', fontproperties=font_prop)
for tl in cbar_heatmap.ax.get_yticklabels():
    tl.set_fontproperties(font_prop)

# Set ticks and labels
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.index)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right', fontsize=10, fontproperties=font_prop)
ax.set_yticklabels(correlation_matrix.index, fontsize=10, fontproperties=font_prop)

# Annotate the visible cells (lower triangle + diagonal)
for i in range(correlation_matrix.shape[0]):
    for j in range(correlation_matrix.shape[1]):
        if not mask[i, j]:
            text = f"{correlation_matrix.iloc[i, j]:.2f}"
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black', fontproperties=font_prop)

ax.set_title('Correlation Heatmap of Top 15 Important Features', fontsize=14, fontweight='bold', 
             color=colors['text'], pad=20)

# Styling
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Figure 4: F-score distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
bars1 = plt.bar(range(len(f_results)), f_results['F_score'], 
                color=[colors['primary'] if p < 0.05 else colors['secondary'] for p in f_results['p_value']],
                edgecolor=colors['text'], linewidth=0.5)

plt.xlabel('Feature Index', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.ylabel('F-score', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.title('F-score Distribution of All Features', fontsize=12, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.3, color=colors['text'])

# 添加显著性标记
for i, (bar, p_val) in enumerate(zip(bars1, f_results['p_value'])):
    if p_val < 0.05:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 '*', ha='center', va='bottom', fontsize=12, color=colors['warning'], fontweight='bold', fontproperties=font_prop)

plt.subplot(1, 2, 2)
# Detailed chart of top 10 F-score features
top_10_f = f_results.head(10)
bars2 = plt.bar(range(len(top_10_f)), top_10_f['F_score'], 
                color=colors['primary'], alpha=0.8, edgecolor=colors['text'], linewidth=0.5)

plt.xticks(range(len(top_10_f)), top_10_f['Feature'], rotation=45, ha='right', fontsize=10, fontproperties=font_prop)
plt.ylabel('F-score', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.title('Top 10 Features by F-score', fontsize=12, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.3, color=colors['text'])

# Set font for all ticks in current axis
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font_prop)

# 添加数值标签
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.2, 
             f'{height:.1f}', ha='center', va='bottom', fontsize=9, color=colors['text'], fontproperties=font_prop)

plt.tight_layout()
plt.savefig('output/f_score_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Figure 4.1: Random Forest Importance Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# Color logic: Accent if p < 0.05, else Secondary (Gray)
rf_colors = [colors['accent'] if p < 0.05 else colors['secondary'] for p in rf_results['p_value']]
bars_rf = plt.bar(range(len(rf_results)), rf_results['RF_importance'], 
                  color=rf_colors, edgecolor=colors['text'], linewidth=0.5)

plt.xlabel('Feature Index', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.ylabel('Importance', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.title('Random Forest Importance Distribution', fontsize=12, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.3, color=colors['text'])

# Add stars for significant features
rf_max_val = rf_results['RF_importance'].max()
rf_offset = rf_max_val * 0.02
for i, (bar, p_val) in enumerate(zip(bars_rf, rf_results['p_value'])):
    if p_val < 0.05:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rf_offset, 
                 '*', ha='center', va='bottom', fontsize=12, color=colors['accent'], fontweight='bold', fontproperties=font_prop)

# Set font for all ticks in current axis
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font_prop)

plt.subplot(1, 2, 2)
# Detailed chart of top 10 RF features
top_10_rf = rf_results.sort_values(by='RF_importance', ascending=False).head(10)
bars_rf2 = plt.bar(range(len(top_10_rf)), top_10_rf['RF_importance'], 
                   color=colors['accent'], alpha=0.8, edgecolor=colors['text'], linewidth=0.5)

plt.xticks(range(len(top_10_rf)), top_10_rf['Feature'], rotation=45, ha='right', fontsize=10, fontproperties=font_prop)
plt.ylabel('Importance', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.title('Top 10 Features by RF Importance', fontsize=12, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.3, color=colors['text'])

# Set font for all ticks in current axis
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font_prop)

# Add value labels
for bar in bars_rf2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, 
             f'{height:.3f}', ha='center', va='bottom', fontsize=9, color=colors['text'], fontproperties=font_prop)

plt.tight_layout()
plt.savefig('output/rf_importance_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Figure 4.2: LASSO Coefficient Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# Keep sign colors but maybe dim non-significant ones? 
# For consistency with F-score plot, let's keep the sign colors but add stars.
# If we want to dim non-significant, we would need 4 colors (Pos-Sig, Pos-NonSig, Neg-Sig, Neg-NonSig).
# Let's stick to just adding stars to avoid overcomplicating the color scheme which already encodes sign.
bars_lasso = plt.bar(range(len(lasso_results)), lasso_results['LASSO_coef'], 
                     color=[colors['warning'] if c >= 0 else colors['primary'] for c in lasso_results['LASSO_coef']],
                     edgecolor=colors['text'], linewidth=0.5)

plt.xlabel('Feature Index', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.ylabel('Coefficient', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.title('LASSO Coefficient Distribution', fontsize=12, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.3, color=colors['text'])
plt.axhline(0, color=colors['text'], linewidth=0.5)

# Add stars for significant features
lasso_max_abs = lasso_results['LASSO_abs_coef'].max()
lasso_offset = lasso_max_abs * 0.05
for i, (bar, p_val) in enumerate(zip(bars_lasso, lasso_results['p_value'])):
    if p_val < 0.05:
        height = bar.get_height()
        y_pos = height + lasso_offset if height >= 0 else height - lasso_offset
        va = 'bottom' if height >= 0 else 'top'
        # Star color matches the bar color
        star_color = colors['warning'] if height >= 0 else colors['primary']
        plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                 '*', ha='center', va=va, fontsize=12, color=star_color, fontweight='bold', fontproperties=font_prop)

# Set font for all ticks in current axis
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font_prop)

plt.subplot(1, 2, 2)
# Detailed chart of top 10 LASSO features (by absolute value)
top_10_lasso = lasso_results.sort_values(by='LASSO_abs_coef', ascending=False).head(10)
bars_lasso2 = plt.bar(range(len(top_10_lasso)), top_10_lasso['LASSO_coef'], 
                      color=[colors['warning'] if c >= 0 else colors['primary'] for c in top_10_lasso['LASSO_coef']],
                      alpha=0.8, edgecolor=colors['text'], linewidth=0.5)

plt.xticks(range(len(top_10_lasso)), top_10_lasso['Feature'], rotation=45, ha='right', fontsize=10, fontproperties=font_prop)
plt.ylabel('Coefficient', fontsize=11, color=colors['text'], fontproperties=font_prop)
plt.title('Top 10 Features by LASSO (Abs)', fontsize=12, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.grid(axis='y', alpha=0.3, color=colors['text'])
plt.axhline(0, color=colors['text'], linewidth=0.5)

# Set font for all ticks in current axis
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font_prop)

# Add value labels
for bar in bars_lasso2:
    height = bar.get_height()
    y_pos = height + (0.01 if height >= 0 else -0.01)
    va = 'bottom' if height >= 0 else 'top'
    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
             f'{height:.3f}', ha='center', va=va, fontsize=9, color=colors['text'], fontproperties=font_prop)

plt.tight_layout()
plt.savefig('output/lasso_analysis.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Figure 5: Random Forest vs LASSO scatter plot
plt.figure(figsize=(12, 8))

# 散点图
scatter = plt.scatter(final_results['RF_importance'], final_results['LASSO_abs_coef'], 
                     s=final_results['F_score']*5, alpha=0.6, 
                     c=final_results['Composite_score'], cmap='viridis',
                     edgecolors=colors['text'], linewidth=0.5)

# 添加颜色条
cbar = plt.colorbar(scatter)
# set composite score label and ticks to Times New Roman
cbar.set_label('Composite Score')
cbar.ax.yaxis.label.set_fontproperties(font_prop)
for tl in cbar.ax.get_yticklabels():
    tl.set_fontproperties(font_prop)
for tl in cbar.ax.get_yticklabels():
    tl.set_color(colors['text'])

# 标注前5个最重要的特征
top_5 = final_results.head(5)
for i, row in top_5.iterrows():
    plt.annotate(row['Feature'], 
                (row['RF_importance'], row['LASSO_abs_coef']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, color=colors['text'], fontweight='bold', fontproperties=font_prop)

plt.xlabel('Random Forest Feature Importance', fontsize=12, color=colors['text'], fontproperties=font_prop)
plt.ylabel('Absolute LASSO Coefficient', fontsize=12, color=colors['text'], fontproperties=font_prop)
plt.title('Random Forest vs LASSO Feature Importance\n(Bubble size = F-score)', 
          fontsize=14, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.grid(True, alpha=0.3, color=colors['text'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Set font for all ticks in current axis
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontproperties(font_prop)

plt.tight_layout()
plt.savefig('output/scatter_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# Figure 6: Violin plot comparing feature importance distributions
plt.figure(figsize=(14, 8))

# 准备数据
methods = ['F_score_norm', 'RF_importance_norm', 'LASSO_coef_norm']
method_names = ['F-score Analysis', 'Random Forest', 'LASSO Regression']

# 创建小提琴图数据
violin_data = []
for method in methods:
    violin_data.append(final_results[method].values)

parts = plt.violinplot(violin_data, positions=range(1, 4), showmeans=True, showmedians=True)

# 自定义小提琴图颜色
for pc in parts['bodies']:
    pc.set_facecolor(colors['primary'])
    pc.set_alpha(0.7)

parts['cmeans'].set_color(colors['warning'])
parts['cmedians'].set_color(colors['text'])

plt.xticks(range(1, 4), method_names, fontsize=12, fontproperties=font_prop)
plt.ylabel('Normalized Score', fontsize=12, color=colors['text'], fontproperties=font_prop)
plt.title('Distribution of Feature Importance (Three Methods)', fontsize=14, fontweight='bold', color=colors['text'], fontproperties=font_prop)
plt.grid(axis='y', alpha=0.3, color=colors['text'])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('output/violin_distribution.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("All visualization charts have been generated!")
print("Saved in: output/")
print("- composite_scores.png: Top 15 Features Composite Importance")
print("- radar_comparison.png: Radar Comparison of Three Methods")
print("- correlation_heatmap.png: Correlation Heatmap (Top 15 Features)")
print("- f_score_analysis.png: F-score Analysis")
print("- rf_importance_analysis.png: Random Forest Importance Analysis")
print("- lasso_analysis.png: LASSO Coefficient Analysis")
print("- scatter_comparison.png: Random Forest vs LASSO Scatter Plot")
print("- violin_distribution.png: Importance Distribution (Violin Plot)")
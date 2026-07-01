import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data preparation (including fictitious data for '10')
data = {
    "ViT-B-32 TextSpan": [35.84, 52.62, 60.00, 62.83, 64.62, 64.01, 25.00],
    "ViT-B-32 PCLens": [51.85, 60.90, 63.69, 64.40, 65.28, 64.01, 30.00],
    "ViT-B-16 TextSpan": [35.03, 56.09, 63.84, 67.00, 68.55, 69.64, 28.00],
    "ViT-B-16 PCLens": [54.78, 63.92, 67.06, 68.70, 69.36, 69.64, 33.00],
    "ViT-L-14 TextSpan": [40.73, 61.98, 69.18, 71.80, 73.21, 73.88, 35.00],
    "ViT-L-14 PCLens": [58.79, 68.74, 71.79, 73.11, 73.77, 73.88, 38.00],
    "ViT-H-14 TextSpan": [29.14, 47.45, 60.54, 67.40, 70.95, 73.02, 20.00],
    "ViT-H-14 PCLens": [45.82, 59.80, 66.93, 70.42, 71.97, 73.02, 22.00],
}

num_embeddings = ['10', '20', '30', '40', '50', '60', 'Baseline (Mean Abl.)']

# Creating the DataFrame
df = pd.DataFrame(data, index=num_embeddings[::-1])

# Create a custom colormap plot with larger gaps between certain rows and columns
plt.figure(figsize=(14, 9))
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5, linecolor='white', cbar_kws={'label': 'Accuracy'})

# Increase column spacing by adjusting tick positions manually
ax.set_xticks(np.arange(len(df.columns)) + 0.5)
ax.set_xticklabels(df.columns, rotation=45, ha='right')

# Add extra space between specific rows (between '60' and 'Baseline')
ax.set_yticks(np.arange(len(df.index)) + 0.5)
ax.set_yticklabels(df.index)

# Set the custom spacing
for ytick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
    if label.get_text() == '60':
        ax.axhline(ytick + 0.5, color='white', linewidth=2)

# Set space between column groups (manually draw lines for visual spacing)
for i in [1, 3, 5]:  # after each pair of TextSpan/PCLens
    ax.axvline(i * 1.0, color='white', linewidth=4)

plt.title('Zero-Shot Accuracy of TextSpan vs SVDLens')
plt.xlabel('Models and Algorithms')
plt.ylabel('Number of Text Embeddings used')

plt.tight_layout()
plt.show()

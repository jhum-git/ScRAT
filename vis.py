import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap.umap_ import UMAP

# Load your data
pca_results = np.load('data/Haniffa/Haniffa_X_pca.npy')
labels = pd.read_pickle('data/Haniffa/labels.pkl')


labels = labels.str.strip()  
labels = labels.replace('nan', np.nan)

selected_labels = ['Mild', 'Moderate', 'Severe', 'Critical']

# mask for labels
mask = labels.isin(selected_labels)

# # Apply filtering
filtered_pca = pca_results[mask.values]
filtered_labels = labels[mask]

# # Run UMAP
# umap_2d = UMAP(n_components=2, n_jobs=-1)
# umap_results = umap_2d.fit_transform(filtered_pca)

# # Plot with only the selected labels
# plt.figure(figsize=(10, 6))
# ax = sns.scatterplot(
#     x=umap_results[:, 0],
#     y=umap_results[:, 1],
#     hue=filtered_labels,
#     palette='viridis',
#     s=50,
#     hue_order=selected_labels  # This ensures only these appear in legend
# )

# plt.title('UMAP Applied to 50D PCA')
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.legend(title='Clinical Status')
# plt.show()
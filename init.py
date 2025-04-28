import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

dataframe = pd.read_csv('data.csv')

data = dataframe[['X', 'Y']].values
labels = dataframe['Label'].values

linkage_matrix = linkage(data, method='ward')

print("Linkage Matrix:")
print(linkage_matrix)

plt.figure(figsize=(8, 4))
dendrogram(linkage_matrix, labels=labels)
plt.title('Dendrogram')
plt.xlabel('Titik')
plt.ylabel('Jarak')
plt.show()

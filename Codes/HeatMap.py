import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_excel('steel-data.xlsx')


plt.figure(figsize=(18, 16), dpi=300)

corr_matrix = df.corr()


sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='RdBu_r',       
    annot_kws={'size': 12, 'weight': 'bold'},  
    cbar=True,
    fmt=".2f",
    square=True,         
    linewidths=0.5,      
    linecolor='gray'
)


plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12, rotation=0)


plt.title("Correlation Heatmap (PCC)", fontsize=16, weight='bold')

plt.tight_layout()
plt.show()

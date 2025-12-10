import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("./data/creditcard.csv.bz2")

sns.heatmap(df.corr())
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns
from scipy.stats import pearsonr

baselines = ["baseline_500_20", "baseline_500", "baseline_500_60"]
tests = ["tabql3_500_20", "tabql3_500", "tabql3_500_60"]

labels = ["500 cars/20% AVs", "500 cars/40% AVs", "500 cars/60% AVs"]

col_pre = "t_HDV_pre"
col_test = "t_HDV_test" 

total_data = []

for test in tests:
    path = f"results_common/{test}/metrics/BenchmarkMetrics.csv"
    df = pd.read_csv(path)
    df["name"] = test
    total_data.append(df)

combined_df = pd.concat(total_data, ignore_index=True)
combined_df = combined_df.sort_values(by=['name']).reset_index(drop=True)

baseline_data = []
for baseline in baselines:
    path = f"results_common/{baseline}/metrics/BenchmarkMetrics.csv"
    df = pd.read_csv(path)
    df["name"] = baseline
    baseline_data.append(df)

baseline_df = pd.concat(baseline_data, ignore_index=True)
baseline_df = baseline_df.sort_values(by=["name"]).reset_index(drop=True)

values1 = baseline_df[col_test].tolist()
values2 = combined_df[col_test].tolist()

fig, ax = plt.subplots(figsize=(16, 9)) 

x = np.arange(len(tests)) 
width = 0.35  

rects1 = ax.bar(x - width/2, values1, width, label="random AVs", color='skyblue')
rects2 = ax.bar(x + width/2, values2, width, label="malicious AVs", color='lightcoral')

ax.set_xlabel(f'Comparison of human travel time after adding malicious AVs vs random AVs')
ax.set_ylabel('Seconds')
ax.set_title(f'Human travel time before and after AVs are added to the network ')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha='right') 

ax.set_ylim(bottom=0, top=350)

ax.legend() 
ax.grid(axis='y', linestyle='--', alpha=0.7) 

plt.tight_layout() 
plt.show() 
plt.show()

av_ratios = [0.2, 0.4, 0.6]


delta = np.array(values2) - np.array(values1)

df = pd.DataFrame({
    'AV_ratio': av_ratios,
    'Time_Increase': delta
})

# Calculate correlation
r, p = pearsonr(df['AV_ratio'], df['Time_Increase'])

# Plot
plt.figure(figsize=(10, 6))
sns.regplot(x='AV_ratio', y='Time_Increase', data=df,
            color='mediumseagreen', scatter_kws={'s': 100}, line_kws={'color': 'darkgreen'})
plt.title(f"Correlation Between AV Ratio and Travel Time Increase after intruducing random AVs vs Malicious AVs:\n"
          f"r = {r:.2f}, p = {p:.3g}")
plt.xlabel("AV Ratio")
plt.ylabel("Travel Time Increase (Random AVs - Malicious AVs, in seconds)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
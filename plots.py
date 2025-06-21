import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr


# tests = ["tabql_500", "tabql_700", "tabql_1034", "tabql_500_adapt", "tabql_700_adapt", "tabql_1034_adapt"]
# tests = ['tabql_20', 'tabql_40', 'tabql_60'] for 1035
tests = ['tabql_700_20', 'tabql_700_40'] # for 700
# tests = ['tabql3_500_20', 'tabql3_500', 'tabql3_500_60'] # for 500
col_pre = "t_HDV_pre"
col_test = "t_HDV_test" 

total_data = []

for test in tests:
    # path = f"results/{test}/metrics/BenchmarkMetrics.csv"
    path = f"./results_jk/{test}/metrics/BenchmarkMetrics.csv"
    df = pd.read_csv(path)
    df["test"] = test


    total_data.append(df)

combined_df = pd.concat(total_data, ignore_index=True)

combined_df = combined_df.sort_values(by=['test']).reset_index(drop=True)

values1 = combined_df[col_pre].tolist()
values2 = combined_df[col_test].tolist()
print(values2)
print(np.array(values2) - np.array(values1))
fig, ax = plt.subplots(figsize=(16, 9))

x = np.arange(len(tests))
width = 0.35

rects1 = ax.bar(x - width/2, values1, width, label=col_pre, color='skyblue')
rects2 = ax.bar(x + width/2, values2, width, label=col_test, color='lightcoral')

ax.set_xlabel(f'Comparison of {col_pre} and {col_test} across all tests')
ax.set_ylabel('Seconds')
ax.set_title(f'Human travel time before and after AVs are added to the network ')
ax.set_xticks(x)
ax.set_xticklabels(['20% AV', '40% AV', '60% AV'], rotation=0, ha='right')

ax.set_ylim(bottom=0, top=350)

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
plt.show()


# AV ratios
x = [0.2, 0.4, 0.6]


# Calculate improvement (difference)
delta = np.array(values1) - np.array(values2)

# Build DataFrame
df = pd.DataFrame({
    'AV_ratio': x,
    'Time_Improvement': delta
})

# Calculate correlation
r, p = pearsonr(df['AV_ratio'], df['Time_Improvement'])

# Plot
plt.figure(figsize=(10, 6))
sns.regplot(x='AV_ratio', y='Time_Improvement', data=df,
            color='mediumseagreen', scatter_kws={'s': 100}, line_kws={'color': 'darkgreen'})
plt.title(f"Correlation Between AV Ratio and Travel Time Improvement:\n"
          f"r = {r:.2f}, p = {p:.3g}")
plt.xlabel("AV Ratio")
plt.ylabel("Travel Time Improvement (Before - After, in seconds)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
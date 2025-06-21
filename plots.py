import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tests = ["tabql_500", "tabql_700", "tabql_1034", "tabql_500_adapt", "tabql_700_adapt", "tabql_1034_adapt"]

col_pre = "t_HDV_pre"
col_test = "t_HDV_test" 

total_data = []

for test in tests[3:]:
    path = f"results/{test}/metrics/BenchmarkMetrics.csv"
    df = pd.read_csv(path)
    df["test"] = test


    total_data.append(df)

combined_df = pd.concat(total_data, ignore_index=True)

combined_df = combined_df.sort_values(by=['test']).reset_index(drop=True)

values1 = combined_df[col_pre].tolist()
values2 = combined_df[col_test].tolist()

fig, ax = plt.subplots(figsize=(16, 9)) 

x = np.arange(len(tests[3:])) 
width = 0.35  

rects1 = ax.bar(x - width/2, values1, width, label=col_pre, color='skyblue')
rects2 = ax.bar(x + width/2, values2, width, label=col_test, color='lightcoral')

ax.set_xlabel(f'Comparison of {col_pre} and {col_test} across all tests')
ax.set_ylabel('Seconds')
ax.set_title(f'Human travel time before and after AVs are added to the network ')
ax.set_xticks(x)
ax.set_xticklabels(tests[3:], rotation=0, ha='right') 

ax.set_ylim(bottom=0, top=350)

ax.legend() 
ax.grid(axis='y', linestyle='--', alpha=0.7) 

plt.tight_layout() 
plt.show() 
plt.show()
import pandas as pd

v100_df = pd.read_csv('energy_df_v100_total.csv')
a100_df = pd.read_csv('energy_df_a100_total.csv')
a100_idle_df = pd.read_csv('energy_df_a100_idle_total.csv')

print(v100_df.groupby(['M']).sum())
print(a100_df.groupby(['M']).sum())
print(a100_idle_df.groupby(['M']).sum())

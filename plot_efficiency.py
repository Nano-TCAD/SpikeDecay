import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


if __name__ == "__main__":

    tags = ['Total Weight', 'Activation', 'Decay', 'STP']
    dev_id = 0
    reps_per_op = 10000

    float16 = pd.DataFrame()
    float32 = pd.DataFrame()

    M = 64
    while M <= 4096:
        df_16 = pd.read_csv(f'full_energy_a100_{M}_float16.csv')
        df_16 = df_16[df_16['tag'].isin(tags)]
        df_16 = df_16.drop(columns=['timestamp', 'duration'])
        df_16.reset_index(inplace=True)
        # NOTE: FP16 ran with twice as many reps per op
        df_16[f'nvidia_gpu_{dev_id}'] *= (64 * 6826) / (2 * reps_per_op * M)
        for tag in tags:
            if tag != 'Activation':
                df_16.loc[df_16['tag'] == tag, f'nvidia_gpu_{dev_id}'] /= 10
        df_16 = df_16.groupby(df_16.index // 4).sum()
        df_16['M'] = M
        df_16['dtype'] = 'fp16'
        float16 = pd.concat([float16, df_16])

        df_32 = pd.read_csv(f'full_energy_a100_{M}_float32.csv')
        df_32 = df_32[df_32['tag'].isin(tags)]
        df_32 = df_32.drop(columns=['timestamp', 'duration'])
        df_32.reset_index(inplace=True)
        df_32[f'nvidia_gpu_{dev_id}'] *= (64 * 6826) / (reps_per_op * M)
        for tag in tags:
            if tag != 'Activation':
                df_32.loc[df_32['tag'] == tag, f'nvidia_gpu_{dev_id}'] /= 10
        df_32 = df_32.groupby(df_32.index // 4).sum()
        df_32['M'] = M
        df_32['dtype'] = 'fp32'
        float32 = pd.concat([float32, df_32])

        M *= 2
    
    data = pd.concat([float16, float32])
    data.rename(columns={'M': 'x', f'nvidia_gpu_{dev_id}': 'Energy [mJ]', 'dtype': 'Precision'}, inplace=True)
    g = sns.lineplot(data=data, x='x', y=f'Energy [mJ]', hue='Precision', estimator='median', errorbar=('ci', 99), n_boot=1000, seed=42)
    g.set_xticks((64, 512, 1024, 2048, 4096), labels=('64', '512', '1024', '2048', '4096'))
    plt.savefig('full_energy_a100.pdf')


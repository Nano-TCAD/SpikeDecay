import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch


from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.pandas_handler import PandasHandler

dev_id = 1


def forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, ctx, rep_per_op=1000):
    
    if ctx:
        ctx.record(tag=f'Total Weight')
    # Compute total weights
    for i in range(rep_per_op):
        total_weights = weights + plastic_weights
    torch.cuda.default_stream(device=dev_id).synchronize()

    if ctx:
        ctx.record(tag=f'Activation')
    # Compute activation
    for i in range(rep_per_op):
        h_tp1 = torch.einsum('bf,bhf->bh', total_input, total_weights)
    torch.cuda.default_stream(device=dev_id).synchronize()

    if ctx:
        ctx.record(tag=f'Decay')
    # Compute weight decay
    for i in range(rep_per_op):
        plastic_weights *= weight_lambda
    torch.cuda.default_stream(device=dev_id).synchronize()
    
    if ctx:
        ctx.record(tag=f'STP')
    # Compute short term plasticity
    for i in range(rep_per_op):
        plastic_weights += assoc
    torch.cuda.default_stream(device=dev_id).synchronize()

    return plastic_weights




#config = {'gpu': -1}
config = {'gpu': dev_id}
device = torch.device(f"cuda:{config['gpu']}") if config['gpu'] > -1 else torch.device('cpu')

# domains = [RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)]
domains = [NvidiaGPUDomain(dev_id)]
#domains = [RaplPackageDomain(0)]

idle_consumption = 55.0  # Watt


num_steps = 10
max_reps_per_op = 32768 * 4

weight_lambda_max = 0.9

# size_total_input = [1, 2656]
# size_weights = [64, 2656]
# size_plastic_weights = [1, 64, 2656]
# size_assoc = [1, 64, 2656]

total_df = pd.DataFrame()

tags = ['Total Weight', 'Activation', 'Decay', 'STP']

for i in range(10):

    M = 2**(6+i)
    reps_per_op = max_reps_per_op // 2**i

    size_total_input = [1, 2656]
    size_weights = [M, 2656]
    size_plastic_weights = [1, M, 2656]
    size_assoc = [1, M, 2656]

    # create input (always use the same input for each step)
    total_input = torch.rand(size_total_input, dtype=torch.float32, device=device)-0.5
    weights = torch.rand(size_weights, dtype=torch.float32, device=device)
    plastic_weights = torch.rand(size_plastic_weights, dtype=torch.float32, device=device) - 0.5
    weight_lambda = torch.rand(size_weights, dtype=torch.float32, device=device)*weight_lambda_max
    assoc = torch.rand(size_assoc, dtype=torch.float32, device=device)-0.5

    forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, None)

    pandas_handler = PandasHandler()
    with EnergyContext(handler=pandas_handler, domains=domains, start_tag='start_forward_pass_loop') as ctx:
        for s in range(num_steps):
            ctx.record(tag=f'forward pass')
            forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, ctx, reps_per_op)
        torch.cuda.default_stream(device=dev_id).synchronize()

    energy_df = pandas_handler.get_dataframe()

    print(f"M = {M}")

    df = energy_df[energy_df['tag'].isin(tags)]
    df[f'nvidia_gpu_{dev_id}'] = df[f'nvidia_gpu_{dev_id}'] - idle_consumption * df['duration'] * 1000  # mJ
    df = df.drop(columns=['timestamp', 'duration'])
    df = df.groupby(['tag']).median()
    df['M'] = M
    df[f'nvidia_gpu_{dev_id}'] *= 64 / (reps_per_op * M)
    print(df)
    print()

    total_df = pd.concat([total_df, df])

    # print('Energy consumption')
    # print(energy_df)

    energy_df.to_csv(f'energy_df_a100_idle_{M}.csv')

# print(total_df)
total_df.to_csv(f'energy_df_a100_idle_total.csv')

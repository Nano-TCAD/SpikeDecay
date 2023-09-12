import numpy as np
import matplotlib.pyplot as plt
import time
import torch


from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.pandas_handler import PandasHandler


def forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, ctx, rep_per_op=1000):
    
    ctx.record(tag=f'Total Weight')
    # Compute total weights
    for i in range(rep_per_op):
        total_weights = weights + plastic_weights

    # print('total_weights')
    # print(total_weights.size())
    # print('total_input')
    # print(total_input.size())
    torch.cuda.default_stream(device=0).synchronize()

    ctx.record(tag=f'Activation')
    # Compute activation
    for i in range(rep_per_op):
        h_tp1 = torch.einsum('bf,bhf->bh', total_input, total_weights)
    torch.cuda.default_stream(device=0).synchronize()

    # Plastic weight update
    ctx.record(tag=f'Decay')
    # Compute weight decay
    for i in range(rep_per_op):
        plastic_weights *= weight_lambda
    torch.cuda.default_stream(device=0).synchronize()
    
    # Compute short term plasticity
    ctx.record(tag=f'STP')
    for i in range(rep_per_op):
        plastic_weights += assoc
    torch.cuda.default_stream(device=0).synchronize()

    return plastic_weights




config = {'gpu': -1}
#config = {'gpu': 0}
device = torch.device(f"cuda:{config['gpu']}") if config['gpu'] > -1 else torch.device('cpu')

pandas_handler = PandasHandler()
#domains = [RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)]
domains = [RaplPackageDomain(0)]


num_steps = 10

weight_lambda_max = 0.9

size_total_input = [1, 2656]
size_weights = [64, 2656]
size_plastic_weights = [1, 64, 2656]
size_assoc = [1, 64, 2656]

# create input (always use the same input for each step)
total_input = torch.rand(size_total_input, dtype=torch.float32, device=device)-0.5
weights = torch.rand(size_weights, dtype=torch.float32, device=device)
plastic_weights = torch.rand(size_plastic_weights, dtype=torch.float32, device=device) - 0.5
weight_lambda = torch.rand(size_weights, dtype=torch.float32, device=device)*weight_lambda_max
assoc = torch.rand(size_assoc, dtype=torch.float32, device=device)-0.5

with EnergyContext(handler=pandas_handler, domains=domains, start_tag='start_forward_pass_loop') as ctx:
    for s in range(num_steps):
        ctx.record(tag=f'forward pass')
        forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, ctx)
    torch.cuda.default_stream(device=0).synchronize()

energy_df = pandas_handler.get_dataframe()

print('Energy consmumption')
print(energy_df)

energy_df.to_csv('energy_df.csv')

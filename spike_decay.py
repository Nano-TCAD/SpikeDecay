import numpy as np
import matplotlib.pyplot as plt
import time
import torch


from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.pandas_handler import PandasHandler


def spike_decay(input_seq, weight_lambda, t_step, batch_size, size_F, device):
    full_size = (batch_size,) + size_F
    F = torch.zeros(full_size, dtype=torch.float32, device=device)
    lambda_tensor = torch.ones(size_F, dtype=torch.float32, device=device) * weight_lambda
    F_result = []
    timestamps_result = []
    pandas_handler = PandasHandler()
    domains = [RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)]
    # domains = [RaplPackageDomain(0)]
    # domains = [NvidiaGPUDomain(0)]
    with EnergyContext(handler=pandas_handler, domains=domains, start_tag='start_decay_loop') as ctx:
        for i, inp in enumerate(input_seq):
            timestamps_result.append(time.time())
            # ctx.record(tag=f'inp_el {i}: scale F')
            # for j in range(10):
            F *= lambda_tensor
            # torch.cuda.default_stream(device=0).synchronize()
            # ctx.record(tag=f'inp_el {i}: add input')
            # for j in range(10):
            F += inp
            # torch.cuda.default_stream(device=0).synchronize()
            # ctx.record(tag=f'inp_el {i}: sleep')
            F_result.append(F[0,0,0])
            time.sleep(t_step)
            # torch.cuda.default_stream(device=0).synchronize()
        torch.cuda.default_stream(device=0).synchronize()
    
    return F_result, timestamps_result, pandas_handler.get_dataframe()


config = {'gpu': 0}
device = torch.device(f"cuda:{config['gpu']}") if config['gpu'] > -1 else torch.device('cpu')
input_seq = [1, 2, 3, 4, 5, 6, 7, 8]
weight_lambda = 0.8
t_step = 0
batch_size = 128
size_F = (37, 48)
size_F = (4096, 4096)

spike_decay(input_seq, weight_lambda, t_step, batch_size, size_F, device)
F_result, timestamps_result, energy_df = spike_decay(input_seq, weight_lambda, t_step, batch_size, size_F, device)

F_result = [F.cpu() for F in F_result]
rel_time = (np.array(timestamps_result)- timestamps_result[0])*1000


plt.figure()
plt.plot(rel_time, F_result)
plt.xlabel('Time [ms]')
plt.ylabel('weight F')
plt.savefig('figA100.pdf')

print(energy_df)

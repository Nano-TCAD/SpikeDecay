import numpy as np
import numpy.typing as npt
import pandas as pd
import torch


from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.pandas_handler import PandasHandler


def bootstrap_ci(data: npt.ArrayLike, confidence: float = 0.01, num_samples: int = 1000, seed: int = 42):
    """Computes the confidence interval of a sample using the (empirical) bootstrap method.

    Parameters
    ----------
    data : array_like
        Input data.
    confidence : float, optional
        Confidence level. Default is 0.01 (99% CI).
    num_samples : int, optional
        Number of bootstrap samples. Default is 1000.
    seed : int, optional
        Seed for the random number generator. Default is 42.

    Returns
    -------
    tuple
        Confidence interval.
    """

    rng = np.random.default_rng(42)

    n = len(data)
    idx = rng.integers(low=0, high=n, size=(num_samples, n))
    samples = data[idx]
    medians = np.median(samples, axis=1)
    medians.sort()
    return (medians[int((confidence / 2) * num_samples)], medians[int((1 - confidence / 2) * num_samples)])



def forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, dev_id, ctx, reps_per_op=1000):
    
    if ctx:
        ctx.record(tag=f'Total Weight')
    # Compute total weights
    for i in range(10 * reps_per_op):
        total_weights = weights + plastic_weights
    torch.cuda.default_stream(device=dev_id).synchronize()

    if ctx:
        ctx.record(tag=f'Activation')
    # Compute activation
    for i in range(reps_per_op):
        h_tp1 = torch.einsum('bf,bhf->bh', total_input, total_weights)
    torch.cuda.default_stream(device=dev_id).synchronize()

    if ctx:
        ctx.record(tag=f'Decay')
    # Compute weight decay
    for i in range(10 * reps_per_op):
        plastic_weights *= weight_lambda
    torch.cuda.default_stream(device=dev_id).synchronize()
    
    if ctx:
        ctx.record(tag=f'STP')
    # Compute short term plasticity
    for i in range(10 * reps_per_op):
        plastic_weights += assoc
    torch.cuda.default_stream(device=dev_id).synchronize()

    return plastic_weights


if __name__ == '__main__':

    dev_id = 0
    config = {'gpu': dev_id}
    device = torch.device(f"cuda:{config['gpu']}") if config['gpu'] > -1 else torch.device('cpu')
    domains = [NvidiaGPUDomain(dev_id)]

    num_steps = 100
    max_reps_per_op = 32768 * 4
    weight_lambda_max = 0.9

    tags = ['Total Weight', 'Activation', 'Decay', 'STP']

    for dt, tdt  in ((np.float16, torch.float16), (np.float32, torch.float32)):
    # for dt, tdt  in ((np.float32, torch.float32), ):

            for i in range(10):

                M = 2**(6+i)
                # reps_per_op = max_reps_per_op // 2**i
                reps_per_op = 10000
                if dt == np.float16:
                    reps_per_op *= 2

                size_total_input = [1, 2656]
                size_weights = [M, 2656]
                size_plastic_weights = [1, M, 2656]
                size_assoc = [1, M, 2656]

                total_input = torch.rand(size_total_input, dtype=tdt, device=device)-0.5
                weights = torch.rand(size_weights, dtype=tdt, device=device)
                plastic_weights = torch.rand(size_plastic_weights, dtype=tdt, device=device) - 0.5
                weight_lambda = torch.rand(size_weights, dtype=tdt, device=device)*weight_lambda_max
                assoc = torch.rand(size_assoc, dtype=tdt, device=device)-0.5

                forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, dev_id, None)

                pandas_handler = PandasHandler()
                with EnergyContext(handler=pandas_handler, domains=domains, start_tag='start_forward_pass_loop') as ctx:
                    for s in range(num_steps):
                        total_input = torch.rand(size_total_input, dtype=tdt, device=device)-0.5
                        weights = torch.rand(size_weights, dtype=tdt, device=device)
                        plastic_weights = torch.rand(size_plastic_weights, dtype=tdt, device=device) - 0.5
                        weight_lambda = torch.rand(size_weights, dtype=tdt, device=device)*weight_lambda_max
                        assoc = torch.rand(size_assoc, dtype=tdt, device=device)-0.5
                        torch.cuda.default_stream(device=dev_id).synchronize()
                        if s % 10 == 0:
                            print(f"Step {s}")
                        forward_pass_memristor_ops(total_input, weights, plastic_weights, weight_lambda, assoc, dev_id, ctx, reps_per_op)
                    torch.cuda.default_stream(device=dev_id).synchronize()

                energy_df = pandas_handler.get_dataframe()

                print(f"M = {M}")

                df = energy_df[energy_df['tag'].isin(tags)]
                df = df.drop(columns=['timestamp', 'duration'])
                df['M'] = M
                df[f'nvidia_gpu_{dev_id}'] *= (64 * 6826) / (reps_per_op * M)

                center = 0
                left_ci = 0
                right_ci = 0
                
                for tag in tags:
                    median = df[df['tag'] == tag][f'nvidia_gpu_{dev_id}'].median()
                    values = df[df['tag'] == tag][f'nvidia_gpu_{dev_id}'].values
                    if tag != 'Activation':
                        median /= 10
                        values /= 10
                    ci = bootstrap_ci(values)
                    print(f'{tag} float{np.dtype(dt).itemsize * 8} energy: {median} {ci} mJ')
                    center += median
                    left_ci += np.square(median - ci[0])
                    right_ci += np.square(ci[1] - median)
                left_ci = np.sqrt(left_ci)
                right_ci = np.sqrt(right_ci)
                print(f'Total energy: {center} ({center - left_ci}, {center + right_ci}) mJ')

                energy_df.to_csv(f'full_energy_a100_{M}_float{np.dtype(dt).itemsize * 8}.csv')

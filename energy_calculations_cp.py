import cupy as cp
import cupyx as cpx
import numpy as np
import matplotlib.pyplot as plt
import time


from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.pandas_handler import PandasHandler


@cpx.jit.rawkernel()
def simple_add(a, b, c, size):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if tid < size:
        c[tid] = a[tid] + b[tid]


@cpx.jit.rawkernel()
def multi_add(a, b, c, size):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if tid < size:
        tmp1 = a[tid]
        tmp2 = b[tid]
        for i in range(1000):
            tmp1 += tmp2
        c[tid] = tmp1


@cpx.jit.rawkernel()
def simple_mul(a, b, c, size):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if tid < size:
        c[tid] = a[tid] * b[tid]


@cpx.jit.rawkernel()
def multi_mul(a, b, c, size):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if tid < size:
        tmp1 = a[tid]
        tmp2 = b[tid]
        for i in range(1000):
            tmp1 *= tmp2
        c[tid] = tmp1


@cpx.jit.rawkernel()
def simple_fma(a, b, c, size):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if tid < size:
        tmp = a[tid]
        c[tid] = tmp + tmp * b[tid]


@cpx.jit.rawkernel()
def multi_fma(a, b, c, size):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    if tid < size:
        tmp1 = a[tid]
        tmp2 = b[tid]
        for i in range(1000):
            tmp1 += tmp1 * tmp2
        c[tid] = tmp1


def flop_energy_test(a, b, c, ctx, rep_per_op=1000):

    nthreads = 1024
    nblocks = (a.size + (nthreads - 1)) // nthreads
    print(f"nblocks: {nblocks}, nthreads: {nthreads}")

    print('Simple Add')
    if ctx:
        ctx.record(tag='Simple_Add')
    for i in range(rep_per_op):
        simple_add[nblocks, nthreads](a, b, c, a.size)
    cp.cuda.get_current_stream().synchronize()

    print('Multi Add')
    if ctx:
        ctx.record(tag='Multi_Add')
    for i in range(rep_per_op):
        multi_add[nblocks, nthreads](a, b, c, a.size)
    cp.cuda.get_current_stream().synchronize()

    print('Simple Mul')
    if ctx:
        ctx.record(tag='Simple_Mul')
    for i in range(rep_per_op):
        simple_mul[nblocks, nthreads](a, b, c, a.size)
    cp.cuda.get_current_stream().synchronize()

    print('Multi Mul')
    if ctx:
        ctx.record(tag='Multi_Mul')
    for i in range(rep_per_op):
        multi_mul[nblocks, nthreads](a, b, c, a.size)
    cp.cuda.get_current_stream().synchronize()

    print('Simple FMA')
    if ctx:
        ctx.record(tag='Simple_FMA')
    for i in range(rep_per_op):
        simple_fma[nblocks, nthreads](a, b, c, a.size)
    cp.cuda.get_current_stream().synchronize()

    print('Multi FMA')
    if ctx:
        ctx.record(tag='Multi_FMA')
    for i in range(rep_per_op):
        multi_fma[nblocks, nthreads](a, b, c, a.size)
    cp.cuda.get_current_stream().synchronize()


pandas_handler = PandasHandler()
# domains = [RaplPackageDomain(0), RaplDramDomain(0), NvidiaGPUDomain(0)]
domains = [NvidiaGPUDomain(0)]
#domains = [RaplPackageDomain(0)]


num_steps = 10
size = (2048*2048,)
reps_per_op = 10000

# create input (always use the same input for each step)
rng = np.random.default_rng(42)
A = cp.asarray(rng.random(size=size, dtype=np.float32))
B = cp.asarray(rng.random(size=size, dtype=np.float32))
C = cp.empty(size, dtype=np.float32)

flop_energy_test(A, B, C, None)

with EnergyContext(handler=pandas_handler, domains=domains, start_tag='start_flop_energy_test') as ctx:
    for s in range(num_steps):
        flop_energy_test(A, B, C, ctx, rep_per_op=reps_per_op)
    cp.cuda.get_current_stream().synchronize()

energy_df = pandas_handler.get_dataframe()

print(energy_df)

for op in ('Add', 'Mul', 'FMA'):
    mem_bounded = energy_df[energy_df['tag'] == f'Simple_{op}']['nvidia_gpu_0'].median()
    comp_bounded = energy_df[energy_df['tag'] == f'Multi_{op}']['nvidia_gpu_0'].median()
    est_flop_energy = (comp_bounded - mem_bounded) * 1e9 / (999 * 2048 * 2048 * reps_per_op)
    print(f'{op} energy: {est_flop_energy} pJ/flop')

energy_df.to_csv('flop_energy_a100.csv')

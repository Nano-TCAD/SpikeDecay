import cupy as cp
import cupyx as cpx
import numpy as np
import numpy.typing as npt


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


multi_add_float16 = cp.RawKernel(r'''
#include "cuda_fp16.h"
extern "C" __global__
void multi_add_float16(const half2* a, const half2* b, half2* c, int iterations) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    half2 tmp0 = a[tid];
    half2 tmp1 = b[tid];
    for (int i = 0; i < iterations; i++) {
        tmp1 = __hadd2(tmp0, tmp1);
    }                                                        
    c[tid] = tmp1;
}
''', 'multi_add_float16')


multi_mul_float16 = cp.RawKernel(r'''
#include "cuda_fp16.h"
extern "C" __global__
void multi_mul_float16(const half2* a, const half2* b, half2* c, int iterations) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    half2 tmp0 = a[tid];
    half2 tmp1 = b[tid];
    for (int i = 0; i < iterations; i++) {
        tmp1 = __hmul2(tmp0, tmp1);
    }                                                        
    c[tid] = tmp1;
}
''', 'multi_mul_float16')


multi_fma_float16 = cp.RawKernel(r'''
#include "cuda_fp16.h"
extern "C" __global__
void multi_fma_float16(const half2* a, const half2* b, half2* c, int iterations) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    half2 tmp0 = a[tid];
    half2 tmp1 = b[tid];
    for (int i = 0; i < iterations; i++) {
        tmp1 = __hfma2(tmp0, tmp1, tmp1);
    }                                                        
    c[tid] = tmp1;
}
''', 'multi_fma_float16')


@cpx.jit.rawkernel()
def multi_add(a, b, c, iterations):
    tid = cpx.jit.blockIdx.x * cpx.jit.blockDim.x + cpx.jit.threadIdx.x
    tmp0 = a[tid]
    tmp1 = b[tid]
    for i in range(iterations):
        tmp1 = tmp0 + tmp1
    c[tid] = tmp1


@cpx.jit.rawkernel()
def multi_mul(a, b, c, iterations):
    tmp0 = a[0]
    tmp1 = b[0]
    for i in range(iterations):
        tmp1 = tmp0 * tmp1
    c[0] = tmp1


@cpx.jit.rawkernel()
def multi_fma(a, b, c, iterations):
    tmp0 = a[0]
    tmp1 = b[0]
    for i in range(iterations):
        tmp1 += tmp0 * tmp1
    c[0] = tmp1


def flop_energy_test_float16(a, b, c, iterations, ctx, reps_per_op=1000):

    nthreads = 1024
    nblocks = 216

    if ctx:
        ctx.record(tag='Add')
    for _ in range(reps_per_op):
        multi_add_float16((nblocks, ), (nthreads, ), (a, b, c, iterations))
    cp.cuda.get_current_stream().synchronize()

    if ctx:
        ctx.record(tag='Mul')
    for _ in range(reps_per_op):
        multi_mul_float16((nblocks, ), (nthreads, ), (a, b, c, iterations))
    cp.cuda.get_current_stream().synchronize()

    if ctx:
        ctx.record(tag='FMA')
    for _ in range(reps_per_op):
        multi_fma_float16((nblocks, ), (nthreads, ), (a, b, c, iterations))
    cp.cuda.get_current_stream().synchronize()


def flop_energy_test(a, b, c, iterations, ctx, reps_per_op=1000):

    nthreads = 1024
    nblocks = 216

    if ctx:
        ctx.record(tag='Add')
    for _ in range(reps_per_op):
        multi_add[nblocks, nthreads](a, b, c, iterations)
    cp.cuda.get_current_stream().synchronize()

    if ctx:
        ctx.record(tag='Mul')
    for _ in range(reps_per_op):
        multi_mul[nblocks, nthreads](a, b, c, iterations)
    cp.cuda.get_current_stream().synchronize()

    if ctx:
        ctx.record(tag='FMA')
    for _ in range(reps_per_op):
        multi_fma[nblocks, nthreads](a, b, c, iterations)
    cp.cuda.get_current_stream().synchronize()


if __name__ == '__main__':

    pandas_handler = PandasHandler()
    domains = [NvidiaGPUDomain(0)]

    nblocks = 216
    nthreads = 1024

    num_steps = 100
    iterations = int((64 * 2656 * 6826) / (nblocks * nthreads))
    reps_per_op = 10000

    for dt, cdt  in ((np.float16, cp.float16), (np.float32, cp.float32)):

        size = (nblocks * nthreads, )
        if dt == np.float16:
            size = (nblocks * nthreads * 2, )
            iterations = int((64 * 2656 * 6826) / (nblocks * nthreads * 2))
        rng = np.random.default_rng(42)
        A = cp.asarray(rng.random(size=size, dtype=np.float32).astype(dt), dtype=cdt)
        B = cp.asarray(rng.random(size=size, dtype=np.float32).astype(dt), dtype=cdt)
        C = cp.empty(size, dtype=cdt)

        func = flop_energy_test_float16 if dt == np.float16 else flop_energy_test

        func(A, B, C, 1000, None)

        with EnergyContext(handler=pandas_handler, domains=domains, start_tag='start_flop_energy_test') as ctx:
            for s in range(num_steps):
                if s % 10 == 0:
                    print(f"Step {s}")
                func(A, B, C, iterations, ctx, reps_per_op=reps_per_op)
            cp.cuda.get_current_stream().synchronize()

        energy_df = pandas_handler.get_dataframe()

        energy_df['nvidia_gpu_0'] = energy_df['nvidia_gpu_0'] / reps_per_op

        for op in ('Add', 'Mul', 'FMA'):
            median = energy_df[energy_df['tag'] == op]['nvidia_gpu_0'].median()
            ci = bootstrap_ci(energy_df[energy_df['tag'] == op]['nvidia_gpu_0'].values)
            print(f'{op} float{np.dtype(dt).itemsize * 8} energy: {median} {ci} mJ')
            mul = 2 if op == 'FMA' else 1
            ops = mul * ((iterations * reps_per_op * nblocks * nthreads) / energy_df[energy_df['tag'] == op]['duration']).median() / 1e9
            print(f'{op} float{np.dtype(dt).itemsize * 8}: {ops} GFLOP/s')

        energy_df.to_csv(f'compute_energy_a100_float{np.dtype(dt).itemsize * 8}.csv')

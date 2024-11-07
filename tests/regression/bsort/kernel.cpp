#include <vx_spawn.h>
#include "common.h"
#include <vx_print.h>

void kernel_body(kernel_arg_t *__UNIFORM__ arg)
{
	// uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t idx = blockIdx.x;
	uint32_t j = arg->j;
	uint32_t k = arg->k;

	auto src_ptr = (TYPE *)arg->src_addr;
	auto dst_ptr = (TYPE *)arg->dst_addr;

	uint32_t ij = idx ^ j;
	if (ij > idx)
	{
		if ((idx & k) == 0 && src_ptr[idx] > src_ptr[ij])
		{
			TYPE temp = src_ptr[idx];
			src_ptr[idx] = src_ptr[ij];
			src_ptr[ij] = temp;
			// std::swap(src_ptr[idx], src_ptr[ij]);
		}
		if ((idx & k) != 0 && src_ptr[idx] < src_ptr[ij])
		{
			TYPE temp = src_ptr[idx];
			src_ptr[idx] = src_ptr[ij];
			src_ptr[ij] = temp;
			// std::swap(src_ptr[idx], src_ptr[ij]);
		}
	}

	__syncthreads();
	// dst_ptr[idx] = src_ptr[idx];
}

int main()
{
	kernel_arg_t *arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
	uint32_t N = arg->num_points;
	auto num_cores = vx_num_cores();

	uint32_t j, k;

	for (k = 2; k <= N; k = 2 * k)
	{
		for (j = k >> 1; j > 0; j = j >> 1)
		{
			arg->j = j;
			arg->k = k;
			vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
			vx_barrier(0x80000000, num_cores);
		}
	}

	auto src_ptr = (TYPE *)arg->src_addr;
	auto dst_ptr = (TYPE *)arg->dst_addr;

	for (uint32_t i = 0; i < N; i++)
	{
		dst_ptr[i] = src_ptr[i];
	}
}

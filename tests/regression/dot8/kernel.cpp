#include <vx_spawn.h>
#include "common.h"
#include <vx_print.h>

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto A = reinterpret_cast<TYPE*>(arg->A_addr);
    auto B = reinterpret_cast<TYPE*>(arg->B_addr);
    auto C = reinterpret_cast<int*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;
    int packedA, packedB;
    int sum(0);

    for (int e = 0; e < size / 4; ++e) {
        packedA = (A[row * size + 0 + e * 4]) \
                | (A[row * size + 1 + e * 4] << 8) \
                | (A[row * size + 2 + e * 4] << 16) \
                | (A[row * size + 3 + e * 4] << 24);
        packedB = (B[(0 + e * 4) * size + col]) \
                | (B[(1 + e * 4) * size + col] << 8) \
                | (B[(2 + e * 4) * size + col] << 16) \
                | (B[(3 + e * 4) * size + col] << 24);
        //int res = vx_dot8(packedA, packedB);
        //vx_printf("%d %d %d %d %d %d %d %d %x %x %d\n", A[row * size + 0 + e * 4], A[row * size + 1 + e * 4], A[row * size + 2 + e * 4], A[row * size + 3 + e * 4], B[(0 + e * 4) * size + col], B[(1 + e * 4) * size + col], B[(2 + e * 4) * size + col], B[(3 + e * 4) * size + col], packedA, packedB, res);
        //sum += res;
        sum+=vx_dot8(packedA, packedB);
    }

    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}

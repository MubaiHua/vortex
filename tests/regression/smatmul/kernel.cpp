#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"
#include <vx_print.h>

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
        uint32_t task_id = blockIdx.x;
        INTYPE* src0_ptr = (INTYPE*)arg->A_addr;
        INTYPE* src1_ptr = (INTYPE*)arg->B_addr;
        INTYPE* src2_ptr = (INTYPE*)arg->C_addr;
        OUTTYPE* dst_ptr  = (OUTTYPE*)arg->D_addr;

        uint32_t A_base = reinterpret_cast<uint32_t>(src0_ptr);
        uint32_t B_base = reinterpret_cast<uint32_t>(src1_ptr);
        uint32_t C_base = reinterpret_cast<uint32_t>(src2_ptr);
        uint32_t D_base = reinterpret_cast<uint32_t>(dst_ptr);

        int offset = task_id;

        uint32_t A_base_addr = A_base + offset * sizeof(INTYPE);
        vx_printf("src0_ptr:%d, A_base:%d, A_base_addr=%d \n", src0_ptr, A_base, A_base_addr);
     
        auto size = arg->size;
        csr_write(VX_TC_SIZE, size);

        vx_matrix_load(0, A_base_addr);
        // vx_matrix_load(1, B_base);
        // vx_matrix_load(2, C_base); 
        // vx_fence();
        
        // vx_matrix_mul();
        // vx_fence();

        // vx_matrix_store(D_base);
        // vx_fence();
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}

#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef INTYPE
#define INTYPE uint8_t   // Modify the data type here
#endif

#ifndef OUTTYPE
#define OUTTYPE uint32_t // Modify the data type here
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t num_tasks;
  uint32_t size;    // size of the matrix is 4 in this case
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t D_addr;
} kernel_arg_t;

#endif

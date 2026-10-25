// mmq tiles for the V4.1 Flash weight types that V4 Flash does not use. One
// translation unit per type keeps the instantiations building in parallel;
// DS41_MMQ_TYPE selects it.
#include "common.hip.hpp"
#include "mmq.hip.hpp"

template void mul_mat_q_case<DS41_MMQ_TYPE>(
    ds41_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);

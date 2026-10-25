// HIP fallback for the NVIDIA-specific D2R launchers in ds41_mmq_d2r.cu.
// Generic MMQ remains available; callers will take its raw-layout path.

#include "ds41_mmq_d2r.hip.hpp"

bool ds41_mmq_q2_K_moe_d2r_available(int) {
    return false;
}

bool ds41_mmq_iq2_xxs_moe_d2r_available(int) {
    return false;
}

bool ds41_mmq_q8_0_dense_d2r_available(int) {
    return false;
}

size_t ds41_mmq_q2_K_moe_d2r_scratch_bytes(int64_t, int) {
    return 0;
}

size_t ds41_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t, int) {
    return 0;
}

size_t ds41_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(int64_t, int) {
    return 0;
}

int ds41_mmq_q2_K_moe_d2r_launch(
        const void *, int64_t, const void *, const int32_t *, const int32_t *,
        float *, int, int, int64_t, int, void *, size_t, hipStream_t) {
    return -1;
}

int ds41_mmq_q8_0_dense_d2r_launch(
        const void *, const void *, float *, int, int, int, hipStream_t) {
    return -1;
}

int ds41_mmq_iq2_xxs_moe_d2r_pair_launch(
        const void *, const void *, int64_t, const void *, const int32_t *,
        const int32_t *, float *, float *, int, int, int64_t, int, void *,
        size_t, hipStream_t) {
    return -1;
}

int ds41_mmq_iq2_xxs_moe_d2r_fused_launch(
        const void *, const void *, int64_t, const void *, const int32_t *, int,
        const int32_t *, const int32_t *, const float *, void *, int, int,
        int64_t, int, float, void *, size_t, hipStream_t) {
    return -1;
}

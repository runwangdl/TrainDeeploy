/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Ported verbatim from Deeploy zo-support (shipped RandomNoise.c). -- QW

#pragma clang optimize off

#include "DeeployPULPMath.h"
#include "pmsis.h"
#include <math.h>
#include "perf_utils.h"

// TODO: 1) loop unrolling for ILP perf
// TODO: 2) Perturbation directly integrated in GEMM or Conv kernels.
/* --------------------------- RNG ---------------------------------- */

uint32_t Xorshift32(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

/* --------------------------- Samplers ---------------------------------- */

float32_t TriangularSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    // mutate state to avoid same seed for u2.
    *state = Xorshift32(*state);
    float32_t u2 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return u1 - u2;
}

float32_t UniformSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return u1-0.5f; // centered around 0
}

float32_t GaussianSample(uint32_t *state) {
    // Box-Muller transform
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in (0,1]
    // mutate state to avoid same seed for u2.
    *state = Xorshift32(*state);
    float32_t u2 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float32_t)PI_F * u2);
}

/* ---------------- Ziggurat method for Gaussian sampling ---------------- */
// This implementation is adapted from the public domain Ziggurat algorithm
// by Marsaglia and Tsang.

void build_ziggurat_tables() {
    if (ziggurat_tables_initialized) return;

    float32_t dn = ZIGGURAT_R;
    float32_t tn = dn;
    float32_t vn = ZIGGURAT_V;

    float32_t q = vn / expf(-0.5f * dn * dn);
    kn[0] = (uint32_t)((dn / q) * ZIGGURAT_INT32_MAX);
    kn[1] = 0;

    wn[0] = (float32_t)(q / ZIGGURAT_INT32_MAX);
    wn[ZIGGURAT_TABLE_SIZE - 1] = (float32_t)(dn / ZIGGURAT_INT32_MAX);

    fn[0] = 1.0f;
    fn[ZIGGURAT_TABLE_SIZE - 1] = (float32_t)expf(-0.5f * dn * dn);

    for (int i = ZIGGURAT_TABLE_SIZE - 2; i > 0; i--) {
        dn = sqrtf(-2.0 * logf(vn / dn + expf(-0.5f * dn * dn)));
        kn[i + 1] = (uint32_t)((dn / tn) * ZIGGURAT_INT32_MAX);
        tn = dn;
        fn[i] = (float32_t)expf(-0.5f * dn * dn);
        wn[i] = (float32_t)(dn / ZIGGURAT_INT32_MAX);
    }
    ziggurat_tables_initialized = 1;
}


// Convert uint32 -> float in (0,1], never 0
static inline float u32_to_u01_open(uint32_t u) {
    return ((float32_t)u + 1.0f) * 2.3283064365386963e-10f; // (u+1)/2^32
}


float32_t GaussianZigguratSample(uint32_t *state) {
    if (!ziggurat_tables_initialized) {
        build_ziggurat_tables();
    }

    int32_t hz;
    uint32_t iz;
    float32_t x, y;

    for (;;) {
        *state = Xorshift32(*state);
        hz = (int32_t)(*state);
        iz = hz & (ZIGGURAT_TABLE_SIZE - 1);

        // Quick acceptance path
        if ((uint32_t)((hz < 0) ? -hz : hz) < kn[iz])  {
            return (float32_t)hz * wn[iz];
        }

        // Handle the tail
        if (iz == 0) {
            do {
                *state = Xorshift32(*state);
                x = -logf(u32_to_u01_open(*state)) / ZIGGURAT_R;
                *state = Xorshift32(*state);
                y = -logf(u32_to_u01_open(*state));
            } while (y + y < x * x);
            return (hz > 0) ? ZIGGURAT_R + x : -ZIGGURAT_R - x;
        }

        // Slow path: use a fresh uniform independent of hz
        float32_t x = (float32_t)hz * wn[iz];
        *state = Xorshift32(*state);
        float32_t u2 = (float32_t)(*state) * ZIGGURAT_INV_UINT32_MAX; // in [0,1]
        if (fn[iz] + u2 * (fn[iz - 1] - fn[iz]) < expf(-0.5f * x * x)) {
            return x;
        }
    }
}

void RademacherRNG_init(RademacherRNG *rng, uint32_t seed) {
    rng->state = seed;
    rng->bits = 0;
    rng->bitpos = 32; // force refill on first use
}

float32_t RademacherSample(RademacherRNG *rng) {
    if (rng->bitpos >= 32) {
        rng->state = Xorshift32(rng->state);
        rng->bits = rng->state;
        rng->bitpos = 0;
    }
    float32_t val = (rng->bits & 1) ? 1.0f : -1.0f;
    rng->bits >>= 1;
    rng->bitpos++;
    return val;
}

/* ------------------------- Perturbation Functions -------------------------------- */

void ApplyTrianglePerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon)
{
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) {epsilon *= -1.0f;}
    uint32_t n_full = size & ~3u;
    uint32_t i = 0;
    for (; i < n_full; i+=4) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i] = pweights[i] + (u1-u2) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i+1] = pweights[i+1] + (u1-u2) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i+2] = pweights[i+2] + (u1-u2) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i+3] = pweights[i+3] + (u1-u2) * epsilon;
    }
    // Remainder
    for (; i < size; i++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i] = pweights[i] + (u1-u2) * epsilon;
    }
}

void ApplyUniformPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon)
{
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    // sqrt(3)*2 factor already included in epsilon to match Gaussian(0, 1) l2 norm.
    if (dir == 0) {epsilon *= -1.0f;}
    uint32_t n_full = size & ~3u;
    uint32_t i = 0;
    for (; i < n_full; i+=4) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i] = pweights[i] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i+1] = pweights[i+1] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i+2] = pweights[i+2] + (u1-0.5f) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i+3] = pweights[i+3] + (u1-0.5f) * epsilon;
    }
    // Remainder
    for (; i < size; i++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        pweights_dest[i] = pweights[i] + (u1-0.5f) * epsilon;
    }
}

void ApplyGaussianPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) {

    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) {epsilon *= -1.0f;}
    uint32_t n_full = size & ~3u;
    uint32_t i = 0;
    for (; i < n_full; i+=4) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        float32_t mag = sqrtf(-2.0f * logf(u1));
        float32_t angle = 2.0f * PI_F * u2;
        pweights_dest[i]     = pweights[i]     + mag * cosf(angle) * epsilon;
        pweights_dest[i + 1] = pweights[i + 1] + mag * sinf(angle) * epsilon;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        mag = sqrtf(-2.0f * logf(u1));
        angle = 2.0f * PI_F * u2;
        pweights_dest[i + 2] = pweights[i + 2] + mag * cosf(angle) * epsilon;
        pweights_dest[i + 3] = pweights[i + 3] + mag * sinf(angle) * epsilon;
    }
    // Remainder: Box-Muller produces pairs, handle 1-3 leftover elements
    if (i < size) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        float32_t mag = sqrtf(-2.0f * logf(u1));
        float32_t angle = 2.0f * PI_F * u2;
        pweights_dest[i] = pweights[i] + mag * cosf(angle) * epsilon;
        i++;
        if (i < size) {
            pweights_dest[i] = pweights[i] + mag * sinf(angle) * epsilon;
            i++;
        }
        if (i < size) {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            u2 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
            mag = sqrtf(-2.0f * logf(u1));
            angle = 2.0f * PI_F * u2;
            pweights_dest[i] = pweights[i] + mag * cosf(angle) * epsilon;
        }
    }
}

void ApplyRademacherPerturbation(const float32_t *pweights,
                            float32_t *pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) {

    // QW: data_in (pweights) and data_out (pweights_dest) may partially OVERLAP in L1 when the tiler
    // under-allocates a tiny tile (e.g. fc_bias, 9 elems: data_out is placed only +4 B into data_in). If
    // data_out sits just AHEAD of data_in, writing dest[i] clobbers src[i+1] — on this core or the next —
    // before it is read, corrupting elements (this was the odd-index fc_bias bug). Detect that small forward
    // overlap (decision is identical on every core, since dest-src is constant, so no barrier divergence),
    // have each core copy its chunk to a PRIVATE temp, then a team barrier guarantees all reads complete
    // before any write. Such overlap only occurs for the tiny under-allocated tiles (offset < 128 B ⇒ tile
    // < 32 elems ⇒ ≤ a few elems per core), so the bounded temp always suffices. z-stream/order is unchanged
    // ⇒ bit-identical to the non-overlapping fast path. (dest behind src, or a normal gap, is already safe.)
    // -- QW
    const float32_t *src = pweights;                                          // -- QW
    long off = (long)((const char *)pweights_dest - (const char *)pweights);  // -- QW
    // RW 2026-10-13: was tmp[128] (512 B of every slave stack, the deepest frame of the ZO graph). The overlap
    // only occurs for tiles < 32 elements (offset < 128 B), so a core's chunk never exceeds 32 elements.
    float32_t tmp[32];                                                        // -- QW (RW: 128 -> 32)
    if (off > 0 && off < 128 && size <= 32u) {                               // -- QW (RW: + size bound)
        uint32_t n = size;                                                   // -- QW (RW: size <= 32)
        for (uint32_t k = 0; k < n; k++) tmp[k] = pweights[k];               // -- QW  read chunk into private temp
        pi_cl_team_barrier(0);                                               // -- QW  all reads done before any write
        src = tmp;                                                           // -- QW  perturb from the temp copy
    }

    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) epsilon *= -1.0f;

    uint32_t n_full_batches = size / 32;
    uint32_t leftover = size % 32;
    uint32_t i = 0;

    // Process full batches
    for (uint32_t batch = 0; batch < n_full_batches; batch++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < 32; b+=2, i+=2) {
            float32_t r = (bits & 1) ? 1.0f : -1.0f;
            pweights_dest[i]   = src[i] + r * epsilon;
            bits >>= 1;
            r = (bits & 1) ? 1.0f : -1.0f;
            pweights_dest[i+1] = src[i+1] + r * epsilon;
            bits >>= 1;
        }
    }

    // Process leftover elements
    if (leftover > 0) {
        rng_state = Xorshift32(rng_state);
        uint32_t bits = rng_state;
        for (uint32_t b = 0; b < leftover; b++, i++) {
            float32_t r = (bits & 1) ? 1.0f : -1.0f;
            pweights_dest[i] = src[i] + r * epsilon;
            bits >>= 1;
        }
    }


}

void ApplySequentialRademacherPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            uint32_t dir,
                            uint32_t size,
                            float32_t epsilon) {

    // Rademacher (+/-1) perturbation that, unlike ApplyRademacherPerturbation,
    // consumes only the LSB of the Xorshift state and advances the PRNG once per
    // element -- exactly the stepping scheme used by ApplyUniformPerturbation.
    int8_t core_id = pi_core_id();
    int8_t log2Core = LOG2(NUM_CORES);

#ifdef ZO_PERTURB_PERF  // RW 2026-10-13: 3 perf_stats_t + a 100 B label lived on every slave stack
    perf_stats_t perf_start, perf_end, perf_total;

    // Initialize and start performance counters (only core 0)
    if (core_id == 0) {
        perf_bench_init();
        perf_bench_start();
        perf_bench_read(&perf_start);

    }
#endif
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    if (dir == 0) epsilon *= -1.0f;
    uint32_t n_full = size & ~3u;
    uint32_t i = 0;
    for (; i < n_full; i+=4) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        pweights_dest[i]   = pweights[i]   + ((rng_state & 1u) ? epsilon : -epsilon);

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        pweights_dest[i+1] = pweights[i+1] + ((rng_state & 1u) ? epsilon : -epsilon);

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        pweights_dest[i+2] = pweights[i+2] + ((rng_state & 1u) ? epsilon : -epsilon);

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        pweights_dest[i+3] = pweights[i+3] + ((rng_state & 1u) ? epsilon : -epsilon);
    }
    // Remainder
    for (; i < size; i++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        pweights_dest[i] = pweights[i] + ((rng_state & 1u) ? epsilon : -epsilon);
    }
    
#ifdef ZO_PERTURB_PERF
    if (core_id == 0) {
        perf_bench_stop();
        perf_bench_read(&perf_end);
        perf_bench_diff(&perf_total, &perf_end, &perf_start);

        char label[100];
        snprintf(label, sizeof(label), "Perturb Rad seq seed=%u N=%u",
                seed, size);
        perf_bench_print(label, &perf_total);
    }
#endif
}

void GenEggrollPerturbation(float32_t *__restrict__ p_dest,
                            uint32_t seed,
                            uint32_t size)
{
    // For compatibility with existing codegen templates. Currently maps to Rademacher noise.
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    uint32_t n_full = size & ~3u;
    uint32_t i = 0;
    for (; i < n_full; i+=4) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        p_dest[i] = u1-0.5f;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        p_dest[i+1] = u1-0.5f;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        p_dest[i+2] = u1-0.5f;

        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        p_dest[i+3] = u1-0.5f;
    }
    // Remainder
    for (; i < size; i++) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        float32_t u1 = (float32_t)(rng_state) / (float32_t)0xFFFFFFFF;
        p_dest[i] = u1-0.5f;
    }
}


/* --------------------------- Update functions ---------------------------------- */

// void UpdateWeightsTriangle(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     uint32_t rng_state = (seed * 1664525u) + 1013904223u;
//     float32_t sqrt6 = 2.44948974278f;
//     const float32_t scale = sqrt6; // sqrt(6): => Gaussian(0, 1) l2 norm.
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t tr = TriangularSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * tr * scale;
//     }
// }

// void UpdateWeightsUniform(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     uint32_t rng_state = (seed * 1664525u) + 1013904223u;
//     float32_t sqrt3 = 1.73205080757f;
//     const float32_t scale = sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => variance 1
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t u = UniformSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u * scale;
//     }
// }

// void UpdateWeightsGaussian(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     uint32_t rng_state = (seed * 1664525u) + 1013904223u;
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t u = GaussianSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u;
//     }
// }

// void UpdateWeightsRademacher(float32_t *__restrict__ pweights,
//                             float32_t loss,
//                             uint32_t seed,
//                             float32_t epsilon,
//                             float32_t lr,
//                             uint32_t size) {
//     RademacherRNG rng_state = { (seed * 1664525u) + 1013904223u, 0, 32 };
//     for (uint32_t i = 0; i < size; i++) {
//         float32_t u = RademacherSample(&rng_state);
//         pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u;
//     }
// }
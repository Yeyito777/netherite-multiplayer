#include <stdlib.h>
#include <string.h>
#include "pvp_env.h"

typedef struct {
    int n;
    PvpMatch *matches;
    McSinTable *st;
} PvpVec;

void *pvp_create(int device, int n) {
    PvpVec *v;
    (void)device;
    if (n <= 0) return NULL;
    v = (PvpVec *)calloc(1, sizeof *v);
    if (!v) return NULL;
    v->n = n;
    v->matches = (PvpMatch *)calloc((size_t)n, sizeof *v->matches);
    v->st = (McSinTable *)malloc(sizeof *v->st);
    if (!v->matches || !v->st) {
        free(v->matches); free(v->st); free(v); return NULL;
    }
    mc_sin_table_init(v->st);
    return v;
}

void pvp_destroy(void *vh) {
    PvpVec *v = (PvpVec *)vh;
    if (!v) return;
    free(v->matches); free(v->st); free(v);
}

int pvp_reset(void *vh, const uint8_t *mask, const uint64_t *seeds,
              float *obs) {
    PvpVec *v = (PvpVec *)vh;
    int i;
    if (!v || !seeds) return -1;
    #pragma omp parallel for
    for (i = 0; i < v->n; ++i) {
        if (!mask || mask[i]) pvp_match_init_seed(&v->matches[i], seeds[i]);
        if (obs) pvp_observe_match(&v->matches[i], v->st,
                                   obs + (size_t)i * PVP_PLAYERS * PVP_OBS);
    }
    return 0;
}

int pvp_step(void *vh, const double *actions, int repeat, float *obs,
             float *reward, uint8_t *done, int32_t *hits, float *damage) {
    PvpVec *v = (PvpVec *)vh;
    int i;
    if (!v || !actions || repeat <= 0) return -1;
    #pragma omp parallel for
    for (i = 0; i < v->n; ++i)
        pvp_step_match(&v->matches[i], v->st,
                       actions + (size_t)i * PVP_PLAYERS * PVP_ACT, repeat,
                       obs ? obs + (size_t)i * PVP_PLAYERS * PVP_OBS : NULL,
                       reward ? reward + (size_t)i * PVP_PLAYERS : NULL,
                       done ? done + i : NULL,
                       hits ? hits + (size_t)i * PVP_PLAYERS : NULL,
                       damage ? damage + (size_t)i * PVP_PLAYERS : NULL);
    return 0;
}

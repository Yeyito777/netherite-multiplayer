#include <stdio.h>
#include <stdlib.h>
#include "../core/pvp_arena.h"

static int selftest(const McSinTable *st) {
    PvpMatch m;
    PvpAction a[2];
    PvpStepResult r;
    pvp_match_init(&m);
    m.player[0].x = -1.0; m.player[0].yaw = -90.0F;
    m.player[1].x =  1.0; m.player[1].yaw =  90.0F;
    memset(a, 0, sizeof a);
    a[0].attack = a[1].attack = 1;
    r = pvp_match_step(&m, st, a);
    if (!r.hit[0] || !r.hit[1]) return 10;
    if (m.player[0].health != 19.0F || m.player[1].health != 19.0F) return 11;
    if (m.player[0].x != -m.player[1].x ||
        m.player[0].mx != -m.player[1].mx) return 12;
    r = pvp_match_step(&m, st, a);
    if (r.hit[0] || r.hit[1]) return 13; /* hurt resistance rejects spam */
    if (m.player[0].health != 19.0F || m.player[1].health != 19.0F) return 14;

    pvp_match_init(&m);
    m.player[0].x = -1.0; m.player[0].yaw = 90.0F; /* faces away */
    m.player[1].x =  1.0; m.player[1].yaw = 90.0F;
    memset(a, 0, sizeof a);
    a[0].attack = 1;
    r = pvp_match_step(&m, st, a);
    if (r.hit[0] || m.player[1].health != PVP_MAX_HEALTH) return 15;
    return 0;
}

int main(int argc, char **argv) {
    int ticks = argc > 1 ? atoi(argv[1]) : 96;
    size_t n = (size_t)ticks * PVP_PLAYERS * PVP_STATE_FIELDS;
    McSinTable *st = (McSinTable *)malloc(sizeof *st);
    u64 *out = (u64 *)calloc(n, sizeof *out);
    size_t i;
    if (!st || !out || ticks <= 0) return 2;
    mc_sin_table_init(st);
    {
        int rc = selftest(st);
        if (rc) { fprintf(stderr, "pvp_arena selftest failed: %d\n", rc); return rc; }
    }
    pvp_run_scenario(st, out, ticks);
    for (i = 0; i < n; ++i) printf("%016llx\n", (unsigned long long)out[i]);
    free(out);
    free(st);
    return 0;
}

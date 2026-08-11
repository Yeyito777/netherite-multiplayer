#ifndef NETHERITE_PVP_ENV_H
#define NETHERITE_PVP_ENV_H

#include <stddef.h>
#include <stdint.h>
#include "../core/pvp_arena.h"

#define PVP_ACT 7
#define PVP_OBS 24

/* Raw action row: forward, strafe, dyaw, dpitch, jump, sprint, attack. */
MC_HD static inline PvpAction pvp_decode_action(const double *a, int rep) {
    PvpAction out;
    out.forward = (float)a[0];
    out.strafe = (float)a[1];
    out.dyaw = rep == 0 ? (float)a[2] : 0.0F;
    out.dpitch = rep == 0 ? (float)a[3] : 0.0F;
    out.jump = (int)a[4];
    out.sprint = (int)a[5];
    out.attack = (int)a[6];
    return out;
}

MC_HD static inline void pvp_match_init_seed(PvpMatch *m, uint64_t seed) {
    int axis = (int)(seed & 1ULL);
    int lane0 = (int)((seed >> 1) % 9ULL) - 4;
    int lane1 = (int)((seed >> 5) % 9ULL) - 4;
    int turn0 = (int)((seed >> 9) % 7ULL) - 3;
    int turn1 = (int)((seed >> 12) % 7ULL) - 3;
    double off0 = (double)lane0 * 0.75;
    double off1 = (double)lane1 * 0.75;
    float yaw0 = (float)turn0 * 15.0F;
    float yaw1 = (float)turn1 * 15.0F;
    pvp_match_init(m);
    /* Independent lateral offsets and +/-45 degree yaw errors force policies
     * to learn turn/re-engagement instead of memorizing one straight charge. */
    if (!axis) {
        pvp_player_init(&m->player[0], -4.0, off0, -90.0F + yaw0);
        pvp_player_init(&m->player[1],  4.0, off1,  90.0F + yaw1);
    } else {
        pvp_player_init(&m->player[0], off0, -4.0, yaw0);
        pvp_player_init(&m->player[1], off1,  4.0, 180.0F + yaw1);
    }
}

/* Privileged egocentric observation used by the first MVP. Every field must
 * later have a matching Java server/mod producer before transfer is claimed. */
MC_HD static inline void pvp_observe_one(const PvpMatch *m,
                                         const McSinTable *st, int role,
                                         float *o) {
    const PvpPlayer *p = &m->player[role];
    const PvpPlayer *q = &m->player[1 - role];
    float rad = p->yaw * 0.017453292F;
    float sy = mc_sin(st, rad), cy = mc_cos(st, rad);
    double dx = q->x - p->x, dy = q->y - p->y, dz = q->z - p->z;
    double d = sqrt(dx * dx + dy * dy + dz * dz);
    double dmx = q->mx - p->mx, dmz = q->mz - p->mz;
    o[0] = p->health / PVP_MAX_HEALTH;
    o[1] = q->health / PVP_MAX_HEALTH;
    /* Entity.moveRelative basis: right=(cos(yaw),sin(yaw)),
     * forward=(-sin(yaw),cos(yaw)). */
    o[2] = (float)(p->mx * (double)cy + p->mz * (double)sy);
    o[3] = (float)(-p->mx * (double)sy + p->mz * (double)cy);
    o[4] = (float)p->my;
    o[5] = (float)(dx * (double)cy + dz * (double)sy) / 32.0F;
    o[6] = (float)(-dx * (double)sy + dz * (double)cy) / 32.0F;
    o[7] = (float)dy / 8.0F;
    o[8] = (float)(dmx * (double)cy + dmz * (double)sy);
    o[9] = (float)(-dmx * (double)sy + dmz * (double)cy);
    o[10] = (float)(d / 32.0);
    o[11] = mc_sin(st, (q->yaw - p->yaw) * 0.017453292F);
    o[12] = mc_cos(st, (q->yaw - p->yaw) * 0.017453292F);
    o[13] = sy;
    o[14] = cy;
    o[15] = pvp_cooled_attack(p);
    o[16] = (float)p->hurt_resistant_time / PVP_MAX_HURT_RESISTANT;
    o[17] = (float)q->hurt_resistant_time / PVP_MAX_HURT_RESISTANT;
    o[18] = (float)p->on_ground;
    o[19] = (float)p->sprinting;
    o[20] = (float)((p->x - PVP_ARENA_MIN) / 32.0);
    o[21] = (float)((PVP_ARENA_MAX - p->x) / 32.0);
    o[22] = (float)((p->z - PVP_ARENA_MIN) / 32.0);
    o[23] = (float)((PVP_ARENA_MAX - p->z) / 32.0);
}

MC_HD static inline void pvp_observe_match(const PvpMatch *m,
                                           const McSinTable *st, float *obs) {
    pvp_observe_one(m, st, 0, obs);
    pvp_observe_one(m, st, 1, obs + PVP_OBS);
}

MC_HD static inline void pvp_step_match(PvpMatch *m, const McSinTable *st,
                                        const double *actions, int repeat,
                                        float *obs, float *reward,
                                        uint8_t *done, int32_t *hits,
                                        float *damage) {
    float rew[2] = {0.0F, 0.0F};
    int hs[2] = {0, 0};
    float dmg[2] = {0.0F, 0.0F};
    int rep;
    for (rep = 0; rep < repeat && !m->done; ++rep) {
        PvpAction a[2];
        PvpStepResult r;
        a[0] = pvp_decode_action(actions, rep);
        a[1] = pvp_decode_action(actions + PVP_ACT, rep);
        r = pvp_match_step(m, st, a);
        dmg[0] += r.damage[0]; dmg[1] += r.damage[1];
        hs[0] += r.hit[0]; hs[1] += r.hit[1];
        rew[0] += r.damage[0] - r.damage[1] - 0.001F;
        rew[1] += r.damage[1] - r.damage[0] - 0.001F;
        if (r.done && r.winner >= 0) {
            rew[r.winner] += 10.0F;
            rew[1 - r.winner] -= 10.0F;
        }
    }
    if (obs) pvp_observe_match(m, st, obs);
    if (reward) { reward[0] = rew[0]; reward[1] = rew[1]; }
    if (done) *done = (uint8_t)m->done;
    if (hits) { hits[0] = hs[0]; hits[1] = hs[1]; }
    if (damage) { damage[0] = dmg[0]; damage[1] = dmg[1]; }
}

#endif

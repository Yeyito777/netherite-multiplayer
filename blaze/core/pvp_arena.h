/* pvp_arena: fixed two-player shared-world Minecraft 1.11.2 PvP scaffold.
 *
 * This is the first multiplayer kernel, deliberately separate from Blaze's
 * stable single-player ABI. One PvpMatch is one shared world with exactly two
 * actors. Both attack intents are evaluated before either damage result is
 * committed. CPU and CUDA compile this header from the same source.
 *
 * The dry 32x32 arena movement and combat constants are an executable draft.
 * CPU==CUDA and symmetry are hard gates now; Java 1.11.2 server traces become
 * the behavioral gate before a vanilla-fidelity claim. */
#ifndef MC_PVP_ARENA_H
#define MC_PVP_ARENA_H

#include <math.h>
#include <string.h>
#include "mc.h"
#include "mc_math.h"

#define PVP_PLAYERS 2
#define PVP_ARENA_MIN (-16.0)
#define PVP_ARENA_MAX ( 16.0)
#define PVP_FLOOR_Y 1.0
#define PVP_PLAYER_HALF ((double)(0.6F / 2.0F))
#define PVP_PLAYER_HEIGHT ((double)1.8F)
#define PVP_EYE_HEIGHT ((double)1.62F)
#define PVP_MAX_HEALTH 20.0F
#define PVP_ENTITY_REACH 3.0
#define PVP_ATTACK_PERIOD 5.0F
#define PVP_MAX_HURT_RESISTANT 20
#define PVP_MATCH_TICKS 1200
#define PVP_STATE_FIELDS 20

typedef struct {
    float forward, strafe;
    float dyaw, dpitch;
    int jump, sprint, attack;
} PvpAction;

typedef struct {
    double x, y, z;
    double mx, my, mz;
    float yaw, pitch;
    float health;
    float last_damage;
    int on_ground;
    int sprinting;
    int jump_ticks;
    int ticks_since_attack;
    int hurt_resistant_time;
    int hurt_time;
    int dead;
} PvpPlayer;

typedef struct {
    float damage[PVP_PLAYERS];       /* damage dealt by each player */
    int hit[PVP_PLAYERS];            /* accepted hit by each player */
    int attempted[PVP_PLAYERS];
    int done;
    int winner;                      /* 0/1, -1 draw, -2 live */
} PvpStepResult;

typedef struct {
    PvpPlayer player[PVP_PLAYERS];
    int tick;
    int done;
    int winner;
} PvpMatch;

typedef struct {
    int target;
    int attempted;
    int in_reach;
    int sprint_knockback;
    float damage;
} PvpAttackIntent;

MC_HD static inline float pvp_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

MC_HD static inline double pvp_clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

MC_HD static inline void pvp_player_init(PvpPlayer *p, double x, double z,
                                          float yaw) {
    memset(p, 0, sizeof *p);
    p->x = x;
    p->y = PVP_FLOOR_Y;
    p->z = z;
    p->yaw = yaw;
    p->health = PVP_MAX_HEALTH;
    p->on_ground = 1;
    p->ticks_since_attack = 1000;
}

MC_HD static inline void pvp_match_init(PvpMatch *m) {
    memset(m, 0, sizeof *m);
    pvp_player_init(&m->player[0], -2.0, 0.0, -90.0F);
    pvp_player_init(&m->player[1],  2.0, 0.0,  90.0F);
    m->winner = -2;
}

/* Entity.moveRelative / EntityLivingBase.moveFlying operator order. */
MC_HD static inline void pvp_move_relative(const McSinTable *st,
                                            PvpPlayer *p, float strafe,
                                            float forward, float accel) {
    float f = strafe * strafe + forward * forward;
    if (f >= 1.0e-4F) {
        f = (float)sqrt((double)f);
        if (f < 1.0F) f = 1.0F;
        f = accel / f;
        strafe *= f;
        forward *= f;
        {
            float rad = p->yaw * 0.017453292F;
            float sy = mc_sin(st, rad);
            float cy = mc_cos(st, rad);
            p->mx += (double)(strafe * cy - forward * sy);
            p->mz += (double)(forward * cy + strafe * sy);
        }
    }
}

/* Dry, full-cube stone floor plus an explicit square match boundary. The
 * boundary is a match rule, not a claim that vanilla has invisible walls. */
MC_HD static inline void pvp_move_tick(const McSinTable *st, PvpPlayer *p,
                                       const PvpAction *a) {
    float f2 = 0.91F;
    float strafe = a->strafe * 0.98F;
    float forward = a->forward * 0.98F;

    p->yaw += a->dyaw;
    p->pitch = pvp_clampf(p->pitch + a->dpitch, -90.0F, 90.0F);
    p->sprinting = a->sprint && forward >= 0.8F;

    if (fabs(p->mx) < 0.003) p->mx = 0.0;
    if (fabs(p->my) < 0.003) p->my = 0.0;
    if (fabs(p->mz) < 0.003) p->mz = 0.0;
    if (p->jump_ticks > 0) --p->jump_ticks;

    if (a->jump && p->on_ground && p->jump_ticks == 0) {
        p->jump_ticks = 10;
        p->my = 0.41999998688697815;
        if (p->sprinting) {
            float rad = p->yaw * 0.017453292F;
            p->mx -= (double)(mc_sin(st, rad) * 0.2F);
            p->mz += (double)(mc_cos(st, rad) * 0.2F);
        }
    }

    if (p->on_ground) f2 = 0.6F * 0.91F;
    {
        float f3 = 0.16277136F / (f2 * f2 * f2);
        float speed = p->sprinting
            ? (float)(0.10000000149011612 *
                      (1.0 + 0.30000001192092896))
            : 0.1F;
        float accel = p->on_ground ? speed * f3 : 0.02F;
        pvp_move_relative(st, p, strafe, forward, accel);
    }

    p->x += p->mx;
    p->y += p->my;
    p->z += p->mz;

    {
        double lo = PVP_ARENA_MIN + PVP_PLAYER_HALF;
        double hi = PVP_ARENA_MAX - PVP_PLAYER_HALF;
        double nx = pvp_clampd(p->x, lo, hi);
        double nz = pvp_clampd(p->z, lo, hi);
        if (nx != p->x) p->mx = 0.0;
        if (nz != p->z) p->mz = 0.0;
        p->x = nx;
        p->z = nz;
    }
    if (p->y <= PVP_FLOOR_Y) {
        p->y = PVP_FLOOR_Y;
        p->my = 0.0;
        p->on_ground = 1;
    } else {
        p->on_ground = 0;
    }

    p->my -= 0.08;
    p->my *= 0.9800000190734863;
    p->mx *= (double)f2;
    p->mz *= (double)f2;
}

MC_HD static inline int pvp_vertical_overlap(const PvpPlayer *a,
                                              const PvpPlayer *b) {
    return a->y < b->y + PVP_PLAYER_HEIGHT &&
           a->y + PVP_PLAYER_HEIGHT > b->y;
}

/* Entity.applyEntityCollision's symmetric horizontal impulse, restricted to
 * overlapping player boxes. */
MC_HD static inline void pvp_player_collision(PvpPlayer *a, PvpPlayer *b) {
    double dx, dz, m, n;
    if (!pvp_vertical_overlap(a, b)) return;
    if (fabs(a->x - b->x) >= 2.0 * PVP_PLAYER_HALF ||
        fabs(a->z - b->z) >= 2.0 * PVP_PLAYER_HALF) return;
    dx = b->x - a->x;
    dz = b->z - a->z;
    m = fabs(dx) > fabs(dz) ? fabs(dx) : fabs(dz);
    if (m < 0.01) return;
    m = sqrt(m);
    dx /= m;
    dz /= m;
    n = 1.0 / m;
    if (n > 1.0) n = 1.0;
    dx *= n * 0.05;
    dz *= n * 0.05;
    a->mx -= dx;
    a->mz -= dz;
    b->mx += dx;
    b->mz += dz;
}

/* Segment versus expanded target AABB. Returns nearest ray parameter in
 * blocks, or -1. The slab test avoids a fixed-step entity hit approximation. */
MC_HD static inline double pvp_ray_target(const McSinTable *st,
                                          const PvpPlayer *a,
                                          const PvpPlayer *b) {
    float f = mc_cos(st, -a->yaw * 0.017453292F - 3.1415927F);
    float f1 = mc_sin(st, -a->yaw * 0.017453292F - 3.1415927F);
    float f2 = -mc_cos(st, -a->pitch * 0.017453292F);
    float f3 = mc_sin(st, -a->pitch * 0.017453292F);
    double d[3] = {(double)(f1 * f2), (double)f3, (double)(f * f2)};
    double o[3] = {a->x, a->y + PVP_EYE_HEIGHT, a->z};
    const double border = 0.1;
    double lo[3] = {b->x - PVP_PLAYER_HALF - border, b->y - border,
                    b->z - PVP_PLAYER_HALF - border};
    double hi[3] = {b->x + PVP_PLAYER_HALF + border,
                    b->y + PVP_PLAYER_HEIGHT + border,
                    b->z + PVP_PLAYER_HALF + border};
    double tmin = 0.0, tmax = PVP_ENTITY_REACH;
    int k;
    for (k = 0; k < 3; ++k) {
        if (fabs(d[k]) < 1.0e-12) {
            if (o[k] < lo[k] || o[k] > hi[k]) return -1.0;
        } else {
            double inv = 1.0 / d[k];
            double t0 = (lo[k] - o[k]) * inv;
            double t1 = (hi[k] - o[k]) * inv;
            if (t0 > t1) { double q = t0; t0 = t1; t1 = q; }
            if (t0 > tmin) tmin = t0;
            if (t1 < tmax) tmax = t1;
            if (tmax < tmin) return -1.0;
        }
    }
    return tmin <= PVP_ENTITY_REACH ? tmin : -1.0;
}

MC_HD static inline float pvp_cooled_attack(const PvpPlayer *p) {
    return pvp_clampf(((float)p->ticks_since_attack + 0.5F) /
                      PVP_ATTACK_PERIOD, 0.0F, 1.0F);
}

MC_HD static inline PvpAttackIntent pvp_attack_intent(
        const McSinTable *st, const PvpPlayer *a, const PvpPlayer *b,
        const PvpAction *act, int target) {
    PvpAttackIntent out;
    float cooled;
    memset(&out, 0, sizeof out);
    out.target = target;
    if (!act->attack || a->dead) return out;
    out.attempted = 1;
    out.in_reach = !b->dead && pvp_ray_target(st, a, b) >= 0.0;
    cooled = pvp_cooled_attack(a);
    out.damage = 1.0F * (0.2F + cooled * cooled * 0.8F);
    out.sprint_knockback = a->sprinting && cooled > 0.9F;
    return out;
}

/* EntityLivingBase hurt-resistance rule. Returns the health actually removed. */
MC_HD static inline float pvp_apply_damage(PvpPlayer *p, float amount) {
    float before = p->health;
    if (p->dead || amount <= 0.0F) return 0.0F;
    if (p->hurt_resistant_time > PVP_MAX_HURT_RESISTANT / 2) {
        if (amount <= p->last_damage) return 0.0F;
        p->health -= amount - p->last_damage;
        p->last_damage = amount;
    } else {
        p->last_damage = amount;
        p->hurt_resistant_time = PVP_MAX_HURT_RESISTANT;
        p->hurt_time = 10;
        p->health -= amount;
    }
    if (p->health <= 0.0F) {
        p->health = 0.0F;
        p->dead = 1;
    }
    return before - p->health;
}

MC_HD static inline void pvp_knockback(PvpPlayer *target,
                                       const PvpPlayer *attacker,
                                       float strength) {
    double dx = attacker->x - target->x;
    double dz = attacker->z - target->z;
    double n = sqrt(dx * dx + dz * dz);
    if (n < 1.0e-6) return;
    target->mx /= 2.0;
    target->my /= 2.0;
    target->mz /= 2.0;
    target->mx -= dx / n * (double)strength;
    target->my += (double)strength;
    if (target->my > 0.4000000059604645)
        target->my = 0.4000000059604645;
    target->mz -= dz / n * (double)strength;
    target->on_ground = 0;
}

MC_HD static inline void pvp_tick_timers(PvpPlayer *p) {
    if (p->hurt_resistant_time > 0) --p->hurt_resistant_time;
    if (p->hurt_time > 0) --p->hurt_time;
    if (p->ticks_since_attack < 1000000) ++p->ticks_since_attack;
}

MC_HD static inline PvpStepResult pvp_match_step(PvpMatch *m,
                                                  const McSinTable *st,
                                                  const PvpAction action[2]) {
    PvpStepResult r;
    PvpAttackIntent intent[2];
    int i;
    memset(&r, 0, sizeof r);
    r.winner = -2;
    if (m->done) { r.done = 1; r.winner = m->winner; return r; }

    for (i = 0; i < 2; ++i) pvp_tick_timers(&m->player[i]);
    for (i = 0; i < 2; ++i)
        if (!m->player[i].dead)
            pvp_move_tick(st, &m->player[i], &action[i]);
    pvp_player_collision(&m->player[0], &m->player[1]);

    intent[0] = pvp_attack_intent(st, &m->player[0], &m->player[1],
                                  &action[0], 1);
    intent[1] = pvp_attack_intent(st, &m->player[1], &m->player[0],
                                  &action[1], 0);
    for (i = 0; i < 2; ++i) {
        r.attempted[i] = intent[i].attempted;
        if (intent[i].attempted) m->player[i].ticks_since_attack = 0;
    }

    /* Both intents were captured from the same pre-damage state. */
    for (i = 0; i < 2; ++i) {
        if (intent[i].in_reach) {
            int t = intent[i].target;
            float dealt = pvp_apply_damage(&m->player[t], intent[i].damage);
            if (dealt > 0.0F) {
                float kb = intent[i].sprint_knockback ? 0.9F : 0.4F;
                r.hit[i] = 1;
                r.damage[i] = dealt;
                pvp_knockback(&m->player[t], &m->player[i], kb);
                if (intent[i].sprint_knockback) {
                    m->player[i].mx *= 0.6;
                    m->player[i].mz *= 0.6;
                    m->player[i].sprinting = 0;
                }
            }
        }
    }

    ++m->tick;
    if (m->player[0].dead || m->player[1].dead || m->tick >= PVP_MATCH_TICKS) {
        m->done = 1;
        if (m->player[0].dead == m->player[1].dead) m->winner = -1;
        else m->winner = m->player[0].dead ? 1 : 0;
    }
    r.done = m->done;
    r.winner = m->winner;
    return r;
}

MC_HD static inline u64 pvp_f2u(float v) {
    union { float f; u32 u; } x; x.f = v; return (u64)x.u;
}

MC_HD static inline u64 pvp_d2u(double v) {
    union { double d; u64 u; } x; x.d = v; return x.u;
}

MC_HD static inline void pvp_emit_player(const PvpPlayer *p, u64 *out) {
    out[0] = pvp_d2u(p->x); out[1] = pvp_d2u(p->y); out[2] = pvp_d2u(p->z);
    out[3] = pvp_d2u(p->mx); out[4] = pvp_d2u(p->my); out[5] = pvp_d2u(p->mz);
    out[6] = pvp_f2u(p->yaw); out[7] = pvp_f2u(p->pitch);
    out[8] = pvp_f2u(p->health); out[9] = pvp_f2u(p->last_damage);
    out[10] = (u64)(u32)p->on_ground;
    out[11] = (u64)(u32)p->sprinting;
    out[12] = (u64)(u32)p->jump_ticks;
    out[13] = (u64)(u32)p->ticks_since_attack;
    out[14] = (u64)(u32)p->hurt_resistant_time;
    out[15] = (u64)(u32)p->hurt_time;
    out[16] = (u64)(u32)p->dead;
    out[17] = 0; out[18] = 0; out[19] = 0; /* reserved ABI slots */
}

/* Deterministic scenario used by CPU/CUDA parity and smoke tests. */
MC_HD static inline void pvp_run_scenario(const McSinTable *st, u64 *out,
                                           int ticks) {
    PvpMatch m;
    int t, i;
    pvp_match_init(&m);
    for (t = 0; t < ticks; ++t) {
        PvpAction a[2];
        memset(a, 0, sizeof a);
        a[0].forward = 1.0F; a[1].forward = 1.0F;
        if (t >= 4) { a[0].attack = 1; a[1].attack = 1; }
        if (t == 8) { a[0].jump = 1; a[1].jump = 1; }
        (void)pvp_match_step(&m, st, a);
        for (i = 0; i < 2; ++i)
            pvp_emit_player(&m.player[i],
                            out + ((size_t)t * 2 + i) * PVP_STATE_FIELDS);
    }
}

#endif /* MC_PVP_ARENA_H */

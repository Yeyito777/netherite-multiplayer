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
#define PVP_WEAPON_SWORD 0
#define PVP_WEAPON_AXE 1
#define PVP_SWORD_DAMAGE 6.0F
#define PVP_AXE_DAMAGE 9.0F
#define PVP_SWORD_PERIOD 12.5F
#define PVP_AXE_PERIOD (20.0F / 0.9F)
#define PVP_IRON_ARMOR 15.0F
#define PVP_ARMOR_TOUGHNESS 0.0F
#define PVP_SHIELD_DURABILITY 336
#define PVP_SHIELD_DISABLE_TICKS 100
#define PVP_MAX_HURT_RESISTANT 20
#define PVP_MATCH_TICKS 1200
#define PVP_STATE_FIELDS 30

typedef struct {
    float forward, strafe;
    float dyaw, dpitch;
    int jump, sprint, attack;
    int weapon, block;
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
    int weapon;
    int using_shield;
    int blocking;
    int shield_use_ticks;
    int shield_disabled_ticks;
    int shield_durability;
    int sword_durability;
    int axe_durability;
    int armor_durability[4]; /* boots, leggings, chestplate, helmet */
} PvpPlayer;

typedef struct {
    float damage[PVP_PLAYERS];       /* damage dealt by each player */
    int hit[PVP_PLAYERS];            /* accepted hit by each player */
    int attempted[PVP_PLAYERS];
    int blocked[PVP_PLAYERS];        /* attacker's blow was shield-blocked */
    int shield_disabled[PVP_PLAYERS];/* attacker disabled target shield */
    int done;
    int winner;                      /* 0/1, -1 draw, -2 live */
} PvpStepResult;

typedef struct {
    PvpPlayer player[PVP_PLAYERS];
    int tick;
    int done;
    int winner;
    u64 rng;
} PvpMatch;

typedef struct {
    int target;
    int attempted;
    int in_reach;
    int sprint_knockback;
    int weapon;
    int blocked;
    int disable_shield;
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
    p->weapon = PVP_WEAPON_SWORD;
    p->shield_durability = PVP_SHIELD_DURABILITY;
    p->sword_durability = 250;
    p->axe_durability = 250;
    p->armor_durability[0] = 195;
    p->armor_durability[1] = 225;
    p->armor_durability[2] = 240;
    p->armor_durability[3] = 165;
}

MC_HD static inline void pvp_match_init(PvpMatch *m) {
    memset(m, 0, sizeof *m);
    pvp_player_init(&m->player[0], -2.0, 0.0, -90.0F);
    pvp_player_init(&m->player[1],  2.0, 0.0,  90.0F);
    m->winner = -2;
    m->rng = 0x9e3779b97f4a7c15ULL;
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
    if (p->using_shield) { strafe *= 0.2F; forward *= 0.2F; }
    p->sprinting = !p->using_shield && a->sprint && forward >= 0.8F;

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

MC_HD static inline float pvp_attack_period(const PvpPlayer *p) {
    return p->weapon == PVP_WEAPON_AXE ? PVP_AXE_PERIOD : PVP_SWORD_PERIOD;
}

MC_HD static inline float pvp_cooled_attack(const PvpPlayer *p) {
    return pvp_clampf(((float)p->ticks_since_attack + 0.5F) /
                      pvp_attack_period(p), 0.0F, 1.0F);
}

/* CombatRules.getDamageAfterAbsorb for fixed full iron armor (15, toughness 0). */
MC_HD static inline float pvp_armor_damage(float damage) {
    float lo = PVP_IRON_ARMOR * 0.2F;
    float effective = PVP_IRON_ARMOR - damage /
        (2.0F + PVP_ARMOR_TOUGHNESS / 4.0F);
    effective = pvp_clampf(effective, lo, 20.0F);
    return damage * (1.0F - effective / 25.0F);
}

MC_HD static inline int pvp_shield_faces(const McSinTable *st,
                                         const PvpPlayer *target,
                                         const PvpPlayer *attacker) {
    double dx = attacker->x - target->x;
    double dz = attacker->z - target->z;
    double n = sqrt(dx * dx + dz * dz);
    float rad, fx, fz;
    if (n < 1.0e-9) return 1;
    rad = target->yaw * 0.017453292F;
    fx = -mc_sin(st, rad);
    fz = mc_cos(st, rad);
    return (double)fx * dx / n + (double)fz * dz / n > 0.0;
}

MC_HD static inline u64 pvp_rng_next(u64 *state) {
    u64 x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

MC_HD static inline PvpAttackIntent pvp_attack_intent(
        const McSinTable *st, const PvpPlayer *a, const PvpPlayer *b,
        const PvpAction *act, int target, u64 *rng) {
    PvpAttackIntent out;
    float cooled;
    memset(&out, 0, sizeof out);
    out.target = target;
    if (!act->attack || a->dead || a->blocking) return out;
    out.attempted = 1;
    out.weapon = a->weapon;
    out.in_reach = !b->dead && pvp_ray_target(st, a, b) >= 0.0;
    cooled = pvp_cooled_attack(a);
    out.damage = (a->weapon == PVP_WEAPON_AXE ? PVP_AXE_DAMAGE : PVP_SWORD_DAMAGE)
        * (0.2F + cooled * cooled * 0.8F);
    out.sprint_knockback = a->sprinting && cooled > 0.9F;
    out.blocked = out.in_reach && b->blocking && b->shield_durability > 0 &&
                  b->shield_disabled_ticks == 0 && pvp_shield_faces(st, b, a);
    /* EntityPlayer.disableShield(true): 0.25 base + 0.75 axe bonus = 100%. */
    if (out.blocked && out.weapon == PVP_WEAPON_AXE)
        out.disable_shield = 1;
    (void)rng;
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
    if (p->shield_disabled_ticks > 0) --p->shield_disabled_ticks;
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

    for (i = 0; i < 2; ++i) {
        int weapon;
        pvp_tick_timers(&m->player[i]);
        weapon = action[i].weapon == PVP_WEAPON_AXE
            ? PVP_WEAPON_AXE : PVP_WEAPON_SWORD;
        /* EntityPlayer.resetCooldown when the main-hand stack changes. */
        if (weapon != m->player[i].weapon)
            m->player[i].ticks_since_attack = 0;
        m->player[i].weapon = weapon;
        m->player[i].using_shield = action[i].block &&
            m->player[i].shield_disabled_ticks == 0 &&
            m->player[i].shield_durability > 0;
        if (m->player[i].using_shield) ++m->player[i].shield_use_ticks;
        else m->player[i].shield_use_ticks = 0;
        /* EntityLivingBase.isActiveItemStackBlocking requires five use ticks. */
        m->player[i].blocking = m->player[i].using_shield &&
                                m->player[i].shield_use_ticks >= 5;
    }
    for (i = 0; i < 2; ++i)
        if (!m->player[i].dead)
            pvp_move_tick(st, &m->player[i], &action[i]);
    pvp_player_collision(&m->player[0], &m->player[1]);

    intent[0] = pvp_attack_intent(st, &m->player[0], &m->player[1],
                                  &action[0], 1, &m->rng);
    intent[1] = pvp_attack_intent(st, &m->player[1], &m->player[0],
                                  &action[1], 0, &m->rng);
    for (i = 0; i < 2; ++i) {
        r.attempted[i] = intent[i].attempted;
        /* Vanilla 1.11.2 resets the attack-strength ticker in
         * attackTargetEntityWithCurrentItem, not for a left-click that raycasts
         * only air. The real bridge likewise calls attackEntity only after an
         * entity hit. Resetting on every policy attack intent made training see
         * near-zero cooldown while deployment correctly remained fully cooled
         * during pursuit. */
        if (intent[i].in_reach) m->player[i].ticks_since_attack = 0;
    }

    /* Both intents were captured from the same pre-damage state. */
    for (i = 0; i < 2; ++i) {
        if (intent[i].in_reach) {
            int t = intent[i].target;
            float dealt = 0.0F;
            if (intent[i].blocked) {
                int wear = intent[i].damage >= 3.0F ? 1 + (int)intent[i].damage : 0;
                r.blocked[i] = 1;
                m->player[t].shield_durability -= wear;
                if (intent[i].disable_shield) {
                    m->player[t].shield_disabled_ticks = PVP_SHIELD_DISABLE_TICKS;
                    m->player[t].blocking = 0;
                    m->player[t].using_shield = 0;
                    m->player[t].shield_use_ticks = 0;
                    r.shield_disabled[i] = 1;
                }
                if (m->player[t].shield_durability <= 0) {
                    m->player[t].shield_durability = 0;
                    m->player[t].blocking = 0;
                    m->player[t].using_shield = 0;
                }
                pvp_knockback(&m->player[i], &m->player[t], 0.5F);
            } else {
                dealt = pvp_apply_damage(&m->player[t],
                                         pvp_armor_damage(intent[i].damage));
                if (dealt > 0.0F) {
                    int wear = (int)(intent[i].damage / 4.0F);
                    int slot;
                    if (wear < 1) wear = 1;
                    for (slot = 0; slot < 4; ++slot) {
                        m->player[t].armor_durability[slot] -= wear;
                        if (m->player[t].armor_durability[slot] < 0)
                            m->player[t].armor_durability[slot] = 0;
                    }
                }
            }
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
            if (intent[i].weapon == PVP_WEAPON_AXE) {
                if (m->player[i].axe_durability > 0) m->player[i].axe_durability -= 2;
            } else if (m->player[i].sword_durability > 0) {
                --m->player[i].sword_durability;
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
    out[17] = (u64)(u32)p->weapon;
    out[18] = (u64)(u32)p->blocking;
    out[19] = (u64)(u32)p->using_shield;
    out[20] = (u64)(u32)p->shield_use_ticks;
    out[21] = (u64)(u32)p->shield_disabled_ticks;
    out[22] = (u64)(u32)p->shield_durability;
    out[23] = (u64)(u32)p->sword_durability;
    out[24] = (u64)(u32)p->axe_durability;
    out[25] = (u64)(u32)p->armor_durability[0];
    out[26] = (u64)(u32)p->armor_durability[1];
    out[27] = (u64)(u32)p->armor_durability[2];
    out[28] = (u64)(u32)p->armor_durability[3];
    out[29] = 0;
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

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../core/pvp_arena.h"

static int closef(float a, float b) { return fabsf(a - b) < 1.0e-5F; }

static void ready(PvpMatch *m) {
    pvp_match_init(m);
    pvp_player_init(&m->player[0], -1.0, 0.0, -90.0F);
    pvp_player_init(&m->player[1],  1.0, 0.0,  90.0F);
}

static void raise_shield(PvpMatch *m, const McSinTable *st,
                         PvpAction a[2], int attacker_weapon) {
    int i;
    memset(a, 0, sizeof(PvpAction) * 2);
    a[0].weapon = attacker_weapon;
    a[1].block = 1;
    for (i = 0; i < 4; ++i) (void)pvp_match_step(m, st, a);
    m->player[0].ticks_since_attack = 1000;
}

int main(void) {
    McSinTable st;
    PvpMatch m;
    PvpAction a[2];
    PvpStepResult r;
    mc_sin_table_init(&st);

    assert(closef(pvp_armor_damage(6.0F), 3.12F));
    assert(closef(pvp_armor_damage(9.0F), 5.22F));

    ready(&m); raise_shield(&m, &st, a, PVP_WEAPON_SWORD);
    a[0].attack = 1;
    r = pvp_match_step(&m, &st, a);
    assert(r.blocked[0] == 1 && r.hit[0] == 0);
    assert(closef(m.player[1].health, 20.0F));
    assert(m.player[1].shield_durability == PVP_SHIELD_DURABILITY - 7);
    assert(m.player[0].sword_durability == 249);

    ready(&m); raise_shield(&m, &st, a, PVP_WEAPON_SWORD);
    m.player[1].yaw = -90.0F; /* attacker is behind this east-facing shield */
    a[0].attack = 1;
    r = pvp_match_step(&m, &st, a);
    assert(r.blocked[0] == 0 && r.hit[0] == 1);
    assert(closef(m.player[1].health, 20.0F - 3.12F));

    ready(&m); m.player[0].weapon = PVP_WEAPON_AXE;
    raise_shield(&m, &st, a, PVP_WEAPON_AXE);
    a[0].attack = 1;
    a[0].forward = 1.0F; a[0].sprint = 1; a[1].block = 1;
    r = pvp_match_step(&m, &st, a);
    assert(r.blocked[0] == 1 && r.shield_disabled[0] == 1);
    assert(m.player[1].shield_disabled_ticks == PVP_SHIELD_DISABLE_TICKS);
    assert(m.player[1].blocking == 0);
    assert(m.player[0].axe_durability == 248);

    ready(&m); memset(a, 0, sizeof a);
    a[0].weapon = PVP_WEAPON_AXE;
    (void)pvp_match_step(&m, &st, a);
    assert(closef(pvp_attack_period(&m.player[0]), PVP_AXE_PERIOD));
    a[0].weapon = PVP_WEAPON_SWORD;
    (void)pvp_match_step(&m, &st, a);
    assert(closef(pvp_attack_period(&m.player[0]), PVP_SWORD_PERIOD));

    puts("PASS V2 sword/axe/armor/shield combat");
    return 0;
}

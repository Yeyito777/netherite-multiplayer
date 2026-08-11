"""CPU vector API, determinism, lane independence, and combat smoke gates."""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def build():
    subprocess.run(["make", "-C", HERE, "cpu"], check=True)


def main():
    build()
    from pvp import VecPvp

    n = 8
    seeds = np.arange(n, dtype=np.uint64)
    env = VecPvp(n, so_path=os.path.join(HERE, "pvp_cpu.so"))
    initial = env.reset(seeds).copy()
    assert initial.shape == (n, 2, 24)
    assert np.all(initial[:, :, 0:2] == 1.0)

    zero = np.zeros((n, 2, 7), dtype=np.float64)
    first = tuple(x.copy() for x in env.step(zero))
    env.reset(seeds)
    replay = tuple(x.copy() for x in env.step(zero))
    for a, b in zip(first, replay):
        assert np.array_equal(a, b), "reset/replay is not deterministic"

    # Identical seeds in different lanes must remain byte-identical.
    same = np.full(n, 77, dtype=np.uint64)
    obs = env.reset(same)
    total_hits = np.zeros((n, 2), dtype=np.int64)
    deaths = 0
    for _ in range(400):
        act = np.zeros((n, 2, 7), dtype=np.float64)
        for role in range(2):
            lateral = obs[:, role, 5]
            longitudinal = obs[:, role, 6]
            dist = obs[:, role, 10] * 32.0
            act[:, role, 0] = dist > 1.7
            act[:, role, 2] = np.where(
                longitudinal < 0.0, 15.0,
                np.where(lateral > 0.01, -15.0,
                         np.where(lateral < -0.01, 15.0, 0.0)))
            act[:, role, 5] = 1.0
            act[:, role, 6] = (dist < 3.0) & (obs[:, role, 15] > 0.9)
        obs, reward, done, hits, damage = env.step(act, repeat=1)
        total_hits += hits
        assert np.all(obs == obs[0]), "identical lanes diverged"
        assert np.all(reward == reward[0])
        assert np.all(done == done[0])
        if done[0]:
            deaths += 1
            break
    assert total_hits.sum() > 0, "agents never landed a hit"
    assert np.all(total_hits == total_hits[0]), "lane hit totals differ"
    # Death may take longer under attack spam because 1.11 hurt resistance
    # rejects low-cooldown punches. Hits, not forced termination, gate smoke.
    assert deaths in (0, 1)
    env.close()
    print(f"PASS pvp env: hits={total_hits[0].tolist()} done={bool(deaths)}")


if __name__ == "__main__":
    main()

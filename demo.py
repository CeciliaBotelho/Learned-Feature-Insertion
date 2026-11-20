"""Exercise the mechanism on synthetic inputs with a fresh initialization."""
from collections import Counter
import json

import numpy as np
import torch

from mechanism import D_FEAT, MAX_OPS, N_C, Environment, bce, response
import blocks
from policy import FAM_DE, OPS, Politica, episodio, logp_episodio, mascaras
from mechanism import calibrate
from scipy.special import expit


def main():
    rng = np.random.default_rng(24)
    features = rng.uniform(0, np.pi, (222, D_FEAT))
    teacher = [{"bloco": 0, "k": 0, "w": [0.5, -1.0]}]
    labels = (rng.random(len(features)) < expit(2 * response(teacher, features) - 0.7)).astype(int)
    c, v, a = np.split(features, (N_C, N_C + 56))
    yc, yv, ya = np.split(labels, (N_C, N_C + 56))

    counts = Counter()
    env = Environment(c, yc, v, yv, counts)
    model = Politica(33)
    assert model.n_param() == 21483 and len(blocks.BIB6) == 36

    episodes = [episodio(model, env, rng, counts) for _ in range(8)]
    for episode in episodes:
        circuit = episode["circ"]
        assert len({step["k"] for step in circuit}) == len(circuit)
        assert sum(blocks.BIB6[step["bloco"]]["n_ops"] for step in circuit) <= MAX_OPS
        for step in episode["passos"]:
            action = step["acao"]
            if action is None:
                continue
            blocks_allowed, families_allowed = mascaras(step["livres"], step["rest"])
            assert families_allowed[action["fam"]] and blocks_allowed[action["bloco"]]
            assert FAM_DE[action["bloco"]] == action["fam"]
    assert not mascaras([], 0)[0].any() and OPS.min() == 2
    flipped_counts = Counter()
    flipped = Environment(c, yc, v[::-1], 1 - yv, flipped_counts)
    control_counts = Counter()
    control = Environment(c, yc, v, yv, control_counts)
    left = episodio(model, control, np.random.default_rng(17), control_counts)
    right = episodio(model, flipped, np.random.default_rng(17), flipped_counts)
    assert left["circ"] == right["circ"]
    for original, perturbed in zip(left["passos"], right["passos"], strict=True):
        np.testing.assert_array_equal(original["desc"], perturbed["desc"])
        np.testing.assert_array_equal(original["ctx"], perturbed["ctx"])
    best = min(episodes, key=lambda episode: episode["BCE_V"])
    assert any(episode["circ"] for episode in episodes)
    elite = [episode for episode in sorted(episodes, key=lambda episode: episode["BCE_V"])[:3]]
    loss = -torch.stack([logp_episodio(model, episode) for episode in elite]).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert counts["descriptor_states"] >= counts["aplicacoes_bloco_C"]

    circuit = best["circ"]
    evaluation = np.full(len(ya), yc.mean())
    if circuit:
        slope, intercept = calibrate(response(circuit, c), yc)
        evaluation = expit(slope * response(circuit, a) + intercept)
    print(json.dumps({"proposals": len(episodes), "selected_blocks": len(circuit),
                      "validation_bce": best["BCE_V"], "held_out_bce": bce(ya, evaluation),
                      "validation_isolation_checked": True,
                      "descriptor_states": counts["descriptor_states"],
                      "block_applications": counts["aplicacoes_bloco_C"]}, indent=2))


if __name__ == "__main__":
    main()

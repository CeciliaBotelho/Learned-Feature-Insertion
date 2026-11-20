"""Circuit construction and validation. Evaluation samples stay with the caller."""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import blocks as L

NB = L.NB
FAMS = L.FAMILIAS
D_FEAT, N_C, MAX_OPS = 16, 110, 16
TOKENS = np.array([-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0])


def bce(labels, probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities)))


def objective(weights, response, labels):
    slope, intercept = weights
    logits = slope * response + intercept
    return float(np.mean(np.logaddexp(0, logits) - labels * logits) + slope**2 / (2 * len(labels)))


def calibrate(response, labels):
    prevalence = labels.mean()
    if prevalence in (0, 1):
        return 0.0, float(np.log(np.clip(prevalence, 1e-12, 1 - 1e-12) /
                                 np.clip(1 - prevalence, 1e-12, 1 - 1e-12)))

    def gradient(weights, response, labels):
        slope, intercept = weights
        residual = expit(slope * response + intercept) - labels
        return np.array([(response @ residual + slope) / len(labels), residual.mean()])

    start = [0.0, np.log(prevalence / (1 - prevalence))]
    fitted = minimize(objective, start, args=(response, labels), jac=gradient,
                      method="BFGS", options={"gtol": 1e-9})
    if not np.isfinite(fitted.x).all() or np.linalg.norm(gradient(fitted.x, response, labels), np.inf) > 1e-6:
        raise RuntimeError("Calibration failed to converge")
    return tuple(fitted.x)


def response(circuit, inputs):
    state = L.estado_inicial(len(inputs))
    for item in circuit:
        state = L.aplica_bloco(state, L.BIB6[item["bloco"]], inputs[:, item["k"]], item["w"])
    return 2 * L.p0(state) - 1


class Environment:
    def __init__(self, construction, labels, validation, validation_labels, counters):
        self.T = {"ZC": construction}
        self.construction = construction
        self.labels = labels
        self.validation = validation
        self.validation_labels = validation_labels
        self.counters = counters
        self.baseline = bce(validation_labels, labels.mean())

    def descritores(self, state, circuit):
        from descriptors import entradas

        available = [j for j in range(D_FEAT) if j not in {item["k"] for item in circuit}]
        start = time.process_time()
        values, _ = entradas(state, self.construction, self.labels, circuit, available)
        self.counters["descriptor_states"] += 1
        self.counters["descriptor_cpu_s"] += time.process_time() - start
        return available, values[:, 1:13].astype(float), values[0, 13:].astype(float)

    def recompensa(self, circuit, state):
        if not circuit:
            return 0.0, self.baseline, bce(self.labels, self.labels.mean())
        construction_response = 2 * L.p0(state) - 1
        slope, intercept = calibrate(construction_response, self.labels)
        probabilities = expit(slope * response(circuit, self.validation) + intercept)
        loss = bce(self.validation_labels, probabilities)
        return self.baseline - loss, loss, objective((slope, intercept), construction_response, self.labels)

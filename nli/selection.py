import numpy as np
import util


def get_max_prob(probs, logits, un):
    return probs.max(axis=-1, keepdims=False)


def get_entropy(probs, logits, un):
    return util.compute_entropy(probs).sum(axis=-1, keepdims=False)


def get_var(probs, logits, un):
    return np.exp(un)


def get_vacuity(probs, logits, un):
    evidence = logits
    S = (evidence + 1).sum(axis=-1, keepdims=False)
    return logits.shape[-1] / S


def get_dissonance(probs, logits, un):
    evidence = evidence = logits
    S = (evidence + 1).sum(axis=-1, keepdims=True)
    belief = evidence / S

    belief_k = np.expand_dims(belief, -1)
    belief_j = np.expand_dims(belief, 1)
    balances = 1 - np.abs(belief_k - belief_j) / (belief_k + belief_j + 1e-7)  # Symmetric
    zero_diag = np.ones_like(balances[0])
    np.fill_diagonal(zero_diag, 0)
    balances *= np.expand_dims(zero_diag, 0)  # Set diagonal as 0
    diss_numerator = (np.expand_dims(belief, 1) * balances).sum(axis=-1, keepdims=False)
    diss_denominator = belief.sum(axis=-1, keepdims=True) - belief + 1e-7
    diss = (belief * diss_numerator / diss_denominator).sum(axis=-1, keepdims=False)
    return diss


def get_custom(probs, logits, un):
    vacuity_coef = 0.1
    vacuity = get_vacuity(probs, logits, un)
    dissonance = get_dissonance(probs, logits, un)
    return vacuity_coef * vacuity + dissonance


criterion_scoring = {
    'max_prob': get_max_prob,
    'entropy': get_entropy,
    'var': get_var,
    'vacuity': get_vacuity,
    'dissonance': get_dissonance,
    'custom': get_custom
}


def get_scoring_func(criterion):
    return criterion_scoring[criterion]


def get_selection_prelim(preds, probs, logits, un, criterion):
    scoring_func = get_scoring_func(criterion)
    scores = scoring_func(probs, logits, un)
    return scores

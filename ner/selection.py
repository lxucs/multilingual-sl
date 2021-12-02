import numpy as np
import util_tag
import util


def get_entity_prob(token_probs, token_tag_un, token_logits, span):
    s_i, e_i = span[1], span[2]
    return token_probs[s_i: e_i+1].max(axis=-1, keepdims=False).mean()


def get_entity_entropy(token_probs, token_tag_un, token_logits, span):
    s_i, e_i = span[1], span[2]
    return util.compute_entropy(token_probs[s_i: e_i+1]).sum(axis=-1).mean()


def get_entity_var(token_probs, token_tag_un, token_logits, span):
    s_i, e_i = span[1], span[2]
    return np.exp(token_tag_un[s_i: e_i+1]).mean()


def get_entity_vacuity(token_probs, token_tag_un, token_logits, span):
    s_i, e_i = span[1], span[2]
    evidence = token_logits[s_i: e_i+1]
    S = (evidence + 1).sum(axis=-1, keepdims=False)
    return (token_logits.shape[-1] / S).mean()


def get_entity_dissonance(token_probs, token_tag_un, token_logits, span):
    s_i, e_i = span[1], span[2]
    evidence = token_logits[s_i: e_i+1]
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
    return diss.mean()


def get_entity_custom(token_probs, token_tag_un, token_logits, span):
    vacuity_coef = 1e-2
    vacuity = get_entity_vacuity(token_probs, token_tag_un, token_logits, span)
    dissonance = get_entity_dissonance(token_probs, token_tag_un, token_logits, span)
    return vacuity_coef * vacuity + dissonance


criterion_scoring = {
    'max_prob': get_entity_prob,
    'entropy': get_entity_entropy,
    'var': get_entity_var,
    'vacuity': get_entity_vacuity,
    'dissonance': get_entity_dissonance,
    'custom': get_entity_custom
}


def get_scoring_func(criterion):
    return criterion_scoring[criterion]


def get_selection_prelim(dataset_name, indices, predictions, probs, tag_uns, logits, criterion):
    if dataset_name == 'panx':
        types = ['PER', 'LOC', 'ORG']
        tag_uns = tag_uns or ([None] * len(probs))
        logits = logits or ([None] * len(probs))
        # Build all entity scores
        type_scores = {t: [] for t in types}  # {type: [(inst idx in dataset, avg entity score)]}
        idx_more_than_one_type = []  # Indices whose instances have more than one entity type
        scoring_func = get_scoring_func(criterion)
        for idx, pred, prob, tag_un, logit in zip(indices, predictions, probs, tag_uns, logits):
            entities = util_tag.get_entities(pred)
            # Skip instance with invalid spans
            if any([e[-1] in ['I', 'E'] for e in entities]):
                continue

            # Take avg entity score per type for each instance
            inst_type_scores = {t: [] for t in types}
            for e in entities:
                entity_score = scoring_func(prob, tag_un, logit, e)
                inst_type_scores[e[0]].append(entity_score.item())
            # Get (idx, entity score) per entity type
            for t, entity_scores in inst_type_scores.items():
                if entity_scores:
                    type_scores[t].append((idx, np.mean(entity_scores).item()))
            # Keep idx with more than one type for later
            if sum([int(len(entity_scores) > 0) for entity_scores in inst_type_scores.values()]) > 1:
                idx_more_than_one_type.append(idx)
    else:
        raise ValueError(dataset_name)
    return type_scores, idx_more_than_one_type

"""Class-balanced and episodic samplers that respect micrograph (image) groups.

Every sampler draws *micrographs first, then a patch inside the micrograph*, so classes
with many patches from few micrographs (e.g. Papilloma: 693 train patches from 19 images)
do not dominate, and positives in a batch come from different micrographs when possible.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class GroupIndex:
    """class -> group (micrograph) -> list of dataset indices."""

    def __init__(self, labels, groups):
        labels = np.asarray(labels)
        if len(labels) != len(groups):
            raise ValueError("labels and groups must have the same length")
        tmp = defaultdict(lambda: defaultdict(list))
        for i, (y, g) in enumerate(zip(labels.tolist(), groups)):
            tmp[y][g].append(i)
        self.by_class = {y: {g: list(ix) for g, ix in d.items()} for y, d in tmp.items()}
        self.classes = sorted(self.by_class)
        # A group must belong to exactly one class.
        seen = {}
        for y, d in self.by_class.items():
            for g in d:
                if g in seen and seen[g] != y:
                    raise ValueError(f"group {g} appears in two classes")
                seen[g] = y

    def groups_of(self, y):
        return list(self.by_class[y].keys())

    def n_patches(self, y):
        return sum(len(v) for v in self.by_class[y].values())


def draw_image_balanced(rng, group_to_idx, group_order, n, allow_repeat):
    """Pick n indices cycling over micrographs in `group_order`, one unused patch per visit.

    If all patches are used: repeat (allow_repeat=True) or stop early (allow_repeat=False).
    """
    pools = {g: list(rng.permutation(group_to_idx[g])) for g in group_order}
    out = []
    if not group_order:
        return out
    while len(out) < n:
        progressed = False
        for g in group_order:
            if len(out) >= n:
                break
            if pools[g]:
                out.append(int(pools[g].pop()))
                progressed = True
        if not progressed:  # every patch used once
            if not allow_repeat:
                break
            pools = {g: list(rng.permutation(group_to_idx[g])) for g in group_order}
    return out


class PKBatchSampler(Sampler):
    """Yields batches of P classes x K samples (image-balanced inside each class)."""

    def __init__(self, labels, groups, classes_per_batch, samples_per_class, steps_per_epoch,
                 seed=0, image_balanced=True):
        if samples_per_class < 2:
            raise ValueError("samples_per_class (K) must be >= 2 for metric losses")
        self.index = GroupIndex(labels, groups)
        self.P = int(classes_per_batch)
        self.K = int(samples_per_class)
        self.steps = int(steps_per_epoch)
        self.seed = int(seed)
        self.image_balanced = image_balanced
        self.epoch = 0
        if self.P > len(self.index.classes):
            raise ValueError(f"P={self.P} > number of classes ({len(self.index.classes)})")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def _sample_class(self, y, rng):
        g2i = self.index.by_class[y]
        if self.image_balanced:
            groups = list(rng.permutation(list(g2i.keys())))
            return draw_image_balanced(rng, g2i, groups, self.K, allow_repeat=True)
        pool = [i for ix in g2i.values() for i in ix]
        replace = len(pool) < self.K
        return [int(i) for i in rng.choice(pool, size=self.K, replace=replace)]

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        classes = np.array(self.index.classes)
        for _ in range(self.steps):
            chosen = rng.choice(classes, size=self.P, replace=False)
            batch = []
            for y in chosen:
                batch.extend(self._sample_class(int(y), rng))
            yield batch


def sample_episode(rng, index: GroupIndex, classes, k_shot, n_query, image_disjoint=True,
                   allow_repeat_support=True, allow_repeat_query=False):
    """One N-way episode. Returns (support_idx, support_cls, query_idx, query_cls).

    image_disjoint=True: support and query patches of a class come from different micrographs
    (needs >= 2 micrographs). Support uses min(k_shot, n_images-1) micrographs; if fewer
    micrographs than k_shot, extra support patches are taken from the same micrographs.
    n_query=None -> every patch of the query micrographs.
    """
    s_idx, s_cls, q_idx, q_cls = [], [], [], []
    for y in classes:
        g2i = index.by_class[y]
        groups = list(rng.permutation(list(g2i.keys())))
        if image_disjoint:
            if len(groups) < 2:
                raise ValueError(f"class {y} has <2 micrographs; image-disjoint episode impossible")
            n_sup_groups = min(k_shot, len(groups) - 1)
            sup_groups, qry_groups = groups[:n_sup_groups], groups[n_sup_groups:]
        else:
            sup_groups = qry_groups = groups
        sup = draw_image_balanced(rng, g2i, sup_groups, k_shot, allow_repeat=allow_repeat_support)
        if image_disjoint:
            if n_query is None:
                qry = [i for g in qry_groups for i in g2i[g]]
            else:
                qry = draw_image_balanced(rng, g2i, qry_groups, n_query, allow_repeat=allow_repeat_query)
        else:
            used = set(sup)
            remaining = {g: [i for i in g2i[g] if i not in used] for g in groups}
            remaining = {g: v for g, v in remaining.items() if v}
            order = [g for g in groups if g in remaining]
            n = sum(len(v) for v in remaining.values()) if n_query is None else n_query
            qry = draw_image_balanced(rng, remaining, order, n, allow_repeat=allow_repeat_query)
        s_idx += sup
        s_cls += [y] * len(sup)
        q_idx += qry
        q_cls += [y] * len(qry)
    return s_idx, s_cls, q_idx, q_cls


class EpisodeBatchSampler(Sampler):
    """Yields [support..., query...] index lists for prototypical-loss training.

    Every class contributes exactly k_shot support and n_query query samples (repeats allowed),
    so the training step can split the batch at n_way * k_shot.
    """

    def __init__(self, labels, groups, n_way, k_shot, n_query, steps_per_epoch, seed=0):
        self.index = GroupIndex(labels, groups)
        # Image-disjoint episodes need >= 2 micrographs per class.
        self.eligible = [c for c in self.index.classes if len(self.index.by_class[c]) >= 2]
        skipped = sorted(set(self.index.classes) - set(self.eligible))
        if skipped:
            print(f"EpisodeBatchSampler: classes {skipped} have <2 micrographs and are not used in episodes")
        if n_way > len(self.eligible):
            raise ValueError(f"n_way={n_way} > number of classes with >=2 micrographs ({len(self.eligible)})")
        self.n_way, self.k_shot, self.n_query = int(n_way), int(k_shot), int(n_query)
        self.steps = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        classes = np.array(self.eligible)
        for _ in range(self.steps):
            chosen = [int(c) for c in rng.choice(classes, size=self.n_way, replace=False)]
            s_idx, _, q_idx, _ = sample_episode(rng, self.index, chosen, self.k_shot, self.n_query,
                                                image_disjoint=True, allow_repeat_support=True,
                                                allow_repeat_query=True)
            yield s_idx + q_idx

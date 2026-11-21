"""Export LightGBM boosters to flat arrays and evaluate them with numba.

Runtime needs only numpy + numba (no lightgbm). Node encoding per tree:
internal nodes are >= 0, leaves are encoded as ~leaf_index (negative).
Decision rule matches LightGBM numerical splits: go left if x <= threshold
(features are always finite here, so missing-value routing never triggers).
Features are passed as float32 and thresholds are stored as float32 (see
exact_f32_thresholds); child indices fit int16 because trees have < 32k nodes.
"""
from __future__ import annotations

import numba as nb
import numpy as np


def export_booster(booster) -> dict:
    """Flatten a LightGBM booster into concatenated node / leaf arrays."""
    dump = booster.dump_model()
    feat, thr, left, right, leaf_val = [], [], [], [], []
    node_off, leaf_off = [0], [0]

    for tree in dump["tree_info"]:
        nodes, leaves = [], []

        def walk(n):  # returns encoded index of n within this tree
            if "leaf_index" in n or "leaf_value" in n and "split_index" not in n:
                leaves.append(n.get("leaf_value", 0.0))
                return ~(len(leaves) - 1)
            if n.get("decision_type", "<=") != "<=":
                raise ValueError("only numerical <= splits supported")
            i = len(nodes)
            nodes.append(None)
            l_, r_ = walk(n["left_child"]), walk(n["right_child"])
            nodes[i] = (n["split_feature"], n["threshold"], l_, r_)
            return i

        root = walk(tree["tree_structure"])
        if not nodes:  # single-leaf tree: one dummy node that always lands on the leaf
            nodes.append((0, np.inf, root, root))
        for f, t, l_, r_ in nodes:
            feat.append(f); thr.append(t); left.append(l_); right.append(r_)
        leaf_val.extend(leaves)
        node_off.append(len(feat)); leaf_off.append(len(leaf_val))

    return {"feat": np.asarray(feat, np.int16), "thr": exact_f32_thresholds(np.asarray(thr, np.float64)),
            "left": np.asarray(left, np.int16), "right": np.asarray(right, np.int16),
            "leaf": np.asarray(leaf_val, np.float64),
            "node_off": np.asarray(node_off, np.int32), "leaf_off": np.asarray(leaf_off, np.int32)}


def exact_f32_thresholds(thr: np.ndarray) -> np.ndarray:
    """Largest float32 <= each float64 threshold.

    For any float32 x: x <= thr  <=>  x <= f32(thr), so float32 comparisons
    reproduce LightGBM's float64 decisions exactly (features are float32).
    """
    t = thr.astype(np.float32)
    over = t.astype(np.float64) > thr
    t[over] = np.nextafter(t[over], np.float32(-np.inf))
    return t


@nb.njit(cache=True)
def predict_row(x, feat, thr, left, right, leaf, node_off, leaf_off):
    s = 0.0
    for t in range(node_off.shape[0] - 1):
        base, lbase = node_off[t], leaf_off[t]
        i = 0
        while True:
            j = base + i
            i = left[j] if x[feat[j]] <= thr[j] else right[j]
            if i < 0:
                s += leaf[lbase + ~i]
                break
    return s


TREE_KEYS = ("feat", "thr", "left", "right", "leaf", "node_off", "leaf_off")


def merge_trees(m0: dict, m1: dict) -> dict:
    """Concatenate the exported ensembles of target 0 and target 1."""
    n_nodes, n_leaves = m0["node_off"][-1], m0["leaf_off"][-1]
    out = {k: np.concatenate([m0[k], m1[k]]) for k in ("feat", "thr", "left", "right", "leaf")}
    out["node_off"] = np.concatenate([m0["node_off"][:-1], m1["node_off"] + n_nodes]).astype(np.int32)
    out["leaf_off"] = np.concatenate([m0["leaf_off"][:-1], m1["leaf_off"] + n_leaves]).astype(np.int32)
    n0, n1 = len(m0["node_off"]) - 1, len(m1["node_off"]) - 1
    out["trange"] = np.array([0, n0, n0 + n1], np.int32)
    return out


@nb.njit(cache=True)
def _trees(x, feat, thr, left, right, leaf, node_off, leaf_off, t0, t1):
    s = 0.0
    for t in range(t0, t1):
        base, lbase = node_off[t], leaf_off[t]
        i = 0
        while True:
            j = base + i
            i = left[j] if x[feat[j]] <= thr[j] else right[j]
            if i < 0:
                s += leaf[lbase + ~i]
                break
    return s

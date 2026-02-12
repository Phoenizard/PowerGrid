"""
Debug script for SQ3 cascade plateau diagnosis.

Four diagnostic tests on m=0, instance 0:
  Test 1: Step-by-step DC cascade at alpha/alpha*=0.5 (plateau region)
  Test 2: Compare alpha/alpha*=0.76 vs 1.05 (plateau→jump)
  Test 3: DC flow correctness check (f_max vs second-largest)
  Test 4: Flow distribution statistics

Usage:
    python debug_cascade.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SQ2_DIR = os.path.join(PROJECT_ROOT, "sq2_data")
SQ4_DIR = os.path.join(PROJECT_ROOT, "sq4_data")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper_reproduction")

for _d in (SQ2_DIR, SQ4_DIR, PAPER_DIR, SCRIPT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from cascade_engine import (
    find_steady_state,
    compute_edge_flows,
    compute_dc_fmax,
    rebalance_power,
    _dc_power_flow,
    run_cascade_dc,
)
from kappa_pipeline import preload_ensemble
from data_loader import evaluate_power_vector


# ── helpers ────────────────────────────────────────────────────────────────

def _pcc_node(A_csr):
    """PCC is always the last node (index n-1)."""
    return A_csr.shape[0] - 1


def _is_pcc_edge(i, j, pcc):
    return i == pcc or j == pcc


def _edge_list(A_csr):
    """Return sorted edge list [(i,j), ...]."""
    edges = []
    n = A_csr.shape[0]
    for i in range(n):
        for idx in range(A_csr.indptr[i], A_csr.indptr[i + 1]):
            j = A_csr.indices[idx]
            if i < j:
                edges.append((i, int(j)))
    return edges


# ── verbose cascade ───────────────────────────────────────────────────────

def verbose_cascade_dc(A_csr, P, alpha, f_max_initial, pcc, label=""):
    """DC cascade with step-by-step printout including rebalance detail."""
    total_edges = A_csr.nnz // 2
    threshold = alpha * f_max_initial
    print(f"\n{'='*64}")
    print(f"CASCADE: {label}")
    print(f"  alpha/alpha* = {alpha:.4f}, f_max = {f_max_initial:.6f}")
    print(f"  threshold    = {threshold:.6f}")
    print(f"  Initial: {A_csr.shape[0]} nodes, {total_edges} edges")
    print(f"{'='*64}")

    surv, ovrl, depth = _verbose_inner(A_csr, P, alpha, f_max_initial, pcc, 0)
    S = surv / total_edges if total_edges > 0 else 0.0
    print(f"\n  >>> RESULT: S = {surv}/{total_edges} = {S:.4f}")
    print(f"      overloaded_total={ovrl}, max_depth={depth}")
    return S


def _verbose_inner(A_csr, P, alpha, f_max_initial, pcc, depth):
    indent = "  " + "  " * depth
    n = A_csr.shape[0]
    n_edges = A_csr.nnz // 2

    if n <= 1 or n_edges == 0:
        print(f"{indent}[d={depth}] trivial ({n}n, {n_edges}e) -> 0 survive")
        return 0, 0, depth

    flows = _dc_power_flow(A_csr, P)
    if not flows:
        print(f"{indent}[d={depth}] no flows -> 0 survive")
        return 0, 0, depth

    threshold = alpha * f_max_initial
    sorted_flows = sorted(flows.items(), key=lambda kv: abs(kv[1]), reverse=True)
    overloaded = [(e, f) for e, f in sorted_flows if abs(f) > threshold]

    # Print flow summary
    flow_abs = [abs(f) for _, f in sorted_flows]
    print(f"{indent}Step @ d={depth}: {n}n, {n_edges}e, "
          f"flow=[{flow_abs[-1]:.4f} .. {flow_abs[0]:.4f}], thr={threshold:.4f}")
    print(f"{indent}  Overloaded: {len(overloaded)}/{len(flows)}")

    if not overloaded:
        print(f"{indent}  -> STABLE, all {len(flows)} edges survive")
        return len(flows), 0, depth

    # Print every overloaded edge
    for (i, j), f in overloaded:
        tag = " [PCC]" if _is_pcc_edge(i, j, pcc) else ""
        print(f"{indent}  REMOVE ({i},{j}) |f|={abs(f):.4f}{tag}")

    # Remove
    A_lil = lil_matrix(A_csr)
    for (i, j), _ in overloaded:
        A_lil[i, j] = 0
        A_lil[j, i] = 0
    A_new = A_lil.tocsr()
    A_new.eliminate_zeros()

    n_comp, labels = connected_components(A_new, directed=False)
    comp_sizes = [int(np.sum(labels == c)) for c in range(n_comp)]
    print(f"{indent}  -> {n_comp} components, sizes={sorted(comp_sizes, reverse=True)[:8]}")

    total_surv = 0
    total_ovrl = len(overloaded)
    max_d = depth

    for comp_id in range(n_comp):
        nodes = np.where(labels == comp_id)[0]
        if len(nodes) <= 1:
            continue
        A_sub = A_new[np.ix_(nodes, nodes)].tocsr()
        P_sub = P[nodes]
        ne = A_sub.nnz // 2

        P_bal = rebalance_power(P_sub)
        if P_bal is None:
            if np.all(np.abs(P_sub) < 1e-5):
                print(f"{indent}  comp{comp_id}({len(nodes)}n,{ne}e): ALL-PASSIVE -> survive")
                total_surv += ne
            else:
                src = int(np.sum(P_sub > 1e-5))
                snk = int(np.sum(P_sub < -1e-5))
                print(f"{indent}  comp{comp_id}({len(nodes)}n,{ne}e): DEAD (src={src},snk={snk}) -> fail")
            continue

        # Show max flow after rebalance
        sub_flows = _dc_power_flow(A_sub, P_bal)
        if sub_flows:
            sub_fmax = max(abs(f) for f in sub_flows.values())
            ratio = sub_fmax / f_max_initial
            print(f"{indent}  comp{comp_id}({len(nodes)}n,{ne}e): rebalanced, "
                  f"max_flow={sub_fmax:.4f} ({ratio:.2%} of f_max), "
                  f"{'> thr -> recurse' if sub_fmax > threshold else '< thr -> all survive'}")
        else:
            print(f"{indent}  comp{comp_id}({len(nodes)}n,{ne}e): rebalanced, no flows")

        s, o, d = _verbose_inner(A_sub, P_bal, alpha, f_max_initial, pcc, depth + 1)
        total_surv += s
        total_ovrl += o
        max_d = max(max_d, d)

    return total_surv, total_ovrl, max_d


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  SQ3 CASCADE PLATEAU DIAGNOSIS")
    print("=" * 64)

    # ── Load instance 0 ──
    print("\n[LOAD] Pre-loading instance 0...")
    t0 = time.time()
    ensemble = preload_ensemble(n_ensemble=1, season="summer", seed=20260211)
    inst = ensemble[0]
    if inst is None:
        print("ERROR: load failed")
        return
    print(f"  Loaded in {time.time()-t0:.1f}s")

    T_NOON = 43200.0
    P_raw = evaluate_power_vector(inst.cons_interps, inst.pv_interps, T_NOON)
    P_norm = P_raw / inst.P_max
    A = inst.A_base
    pcc = _pcc_node(A)
    n = A.shape[0]
    n_edges = A.nnz // 2

    print(f"  n={n}, |E|={n_edges}, PCC=node {pcc}")
    print(f"  P_PCC = {P_norm[pcc]:.4f}")
    print(f"  sum(P) = {np.sum(P_norm):.2e}")

    # DC flows
    dc_flows, f_max_dc, dc_trigger = compute_dc_fmax(A, P_norm)

    # ────────────────────────────────────────────────────────────────────
    # TEST 3: DC flow correctness & concentration
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  TEST 3: DC Flow Distribution")
    print(f"{'='*64}")

    sorted_flows = sorted(dc_flows.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"\n  f_max_dc = {f_max_dc:.6f}  (trigger edge: {dc_trigger})")

    print(f"\n  Top 10 edges by |flow|:")
    for rank, ((i, j), f) in enumerate(sorted_flows[:10]):
        tag = " [PCC]" if _is_pcc_edge(i, j, pcc) else ""
        print(f"    #{rank+1:2d}  ({i:2d},{j:2d})  |f|={abs(f):.6f}  "
              f"({abs(f)/f_max_dc:.2%} of f_max){tag}")

    f_second = abs(sorted_flows[1][1]) if len(sorted_flows) > 1 else 0
    print(f"\n  f_max / f_second = {f_max_dc / f_second:.2f}" if f_second > 0 else "")

    # ────────────────────────────────────────────────────────────────────
    # TEST 4: Flow distribution statistics
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  TEST 4: Flow Distribution Statistics")
    print(f"{'='*64}")

    all_abs = np.array([abs(f) for _, f in sorted_flows])
    print(f"\n  Total edges: {len(all_abs)}")
    print(f"  f_max   = {all_abs[0]:.6f}")
    print(f"  f_second= {all_abs[1]:.6f}")
    print(f"  f_median= {np.median(all_abs):.6f}")
    print(f"  f_mean  = {np.mean(all_abs):.6f}")
    print(f"  f_min   = {all_abs[-1]:.6f}")
    print(f"\n  f_max / f_second = {all_abs[0]/all_abs[1]:.2f}")
    print(f"  f_median / f_max = {np.median(all_abs)/all_abs[0]:.4f}")

    for frac in [0.9, 0.5, 0.33, 0.1]:
        count = int(np.sum(all_abs > frac * f_max_dc))
        print(f"  Edges with |f| > {frac:.0%} f_max: {count}/{len(all_abs)}")

    # PCC vs non-PCC
    pcc_flows = [abs(f) for (i, j), f in sorted_flows if _is_pcc_edge(i, j, pcc)]
    non_pcc_flows = [abs(f) for (i, j), f in sorted_flows if not _is_pcc_edge(i, j, pcc)]
    print(f"\n  PCC edges ({len(pcc_flows)}):     "
          f"mean={np.mean(pcc_flows):.4f}, max={max(pcc_flows):.4f}")
    print(f"  Non-PCC edges ({len(non_pcc_flows)}): "
          f"mean={np.mean(non_pcc_flows):.4f}, max={max(non_pcc_flows):.4f}")
    print(f"  PCC_mean / non-PCC_mean = {np.mean(pcc_flows)/np.mean(non_pcc_flows):.1f}x")

    # ────────────────────────────────────────────────────────────────────
    # TEST 1: Step-by-step cascade at alpha/alpha*=0.5 (plateau)
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  TEST 1: Step-by-step Cascade at alpha/alpha*=0.5 (plateau)")
    print(f"{'='*64}")

    verbose_cascade_dc(A, P_norm, 0.5, f_max_dc, pcc,
                       label="alpha/alpha*=0.5 (PLATEAU REGION)")

    # ────────────────────────────────────────────────────────────────────
    # TEST 2: Compare alpha/alpha*=0.76 vs 1.05 (plateau→jump)
    # ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  TEST 2: Plateau edge — alpha/alpha*=0.76 vs 1.05")
    print(f"{'='*64}")

    for alpha_test in [0.76, 1.05]:
        verbose_cascade_dc(A, P_norm, alpha_test, f_max_dc, pcc,
                           label=f"alpha/alpha*={alpha_test}")

    # ── Quick S scan to locate exact plateau ──
    print(f"\n{'='*64}")
    print("  FINE SCAN: S vs alpha/alpha* around plateau")
    print(f"{'='*64}")
    alphas_fine = np.concatenate([
        np.arange(0.05, 0.4, 0.05),
        np.arange(0.4, 1.2, 0.02),
        np.arange(1.2, 2.6, 0.1),
    ])
    print(f"  {'alpha/a*':>10s}  {'S':>8s}  {'overloaded':>10s}  {'depth':>5s}")
    print(f"  {'-'*40}")
    for a in alphas_fine:
        r = run_cascade_dc(A, P_norm, a, f_max_dc)
        marker = " <-- PLATEAU" if 0.35 < a < 1.05 and abs(r.S - 0.76) < 0.02 else ""
        print(f"  {a:10.4f}  {r.S:8.4f}  {r.n_overload:10d}  {r.cascade_depth:5d}{marker}")

    print("\n  Done.")


if __name__ == "__main__":
    main()

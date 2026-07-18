"""Baselines and bounded lookahead reference for WPR experiments.

中文说明：
- EDF 是传统 deadline-first 在线调度；
- online_ready_greedy 只看当前 ready queue，不访问未来 DAG demand；
- dag_oracle_residency_greedy 可以访问 oracle_dag_demand_target，因此论文中必须标成
  DAG-oracle heuristic，而不是普通在线算法；
- lookahead_search_upper_reference 是有剪枝的 beam-search 参考值，不能写成 MILP optimal。
"""

from __future__ import annotations

import copy

import numpy as np

from dag_a2c.wpr_env import WPREnv


def random_matching(env: WPREnv) -> list[tuple[int, int, int, int]]:
    out = []
    used = set()
    for g in sorted(env.idle_gpus()):
        cand = env.feasible_actions_for_gpu(g, used)
        if not cand:
            continue
        a = cand[int(env.rng.integers(0, len(cand)))]
        out.append(a)
        used.add((a[0], a[1]))
    return out


def edf_matching(env: WPREnv) -> list[tuple[int, int, int, int]]:
    """EDF：优先服务绝对 deadline 最早的 workflow，并选择完成时间最短的模型/GPU。"""

    out = []
    used = set()
    for g in sorted(env.idle_gpus()):
        best = None
        best_key = None
        for a in env.feasible_actions_for_gpu(g, used):
            slot, sid, mid, gid = a
            wf = env.active[slot]
            finish = env.time + env.prep_time(mid, gid) + env.expected_exec_time(slot, sid, mid, gid)
            key = (wf.arrival + wf.template.deadline, finish, env.prep_time(mid, gid))
            if best_key is None or key < best_key:
                best_key = key
                best = a
        if best is not None:
            out.append(best)
            used.add((best[0], best[1]))
    return out


def online_ready_greedy(env: WPREnv) -> list[tuple[int, int, int, int]]:
    """真实在线贪心：只使用当前 ready stage、slack、执行时间和 resident hit。"""

    out = []
    used = set()
    for g in sorted(env.idle_gpus()):
        best = None
        best_score = -np.inf
        for a in env.feasible_actions_for_gpu(g, used):
            slot, sid, mid, gid = a
            wf = env.active[slot]
            slack = wf.arrival + wf.template.deadline - env.time
            prep = env.prep_time(mid, gid)
            duration = env.expected_exec_time(slot, sid, mid, gid)
            resident_bonus = 1.0 if env.resident_model[gid] == mid else 0.0
            ready_wait = max(0.0, env.time - float(wf.ready_times[sid]))
            score = 2.2 * wf.template.weight / max(0.5, slack) + 0.7 * resident_bonus + 0.04 * ready_wait - 0.30 * prep - 0.22 * duration
            if score > best_score:
                best_score = float(score)
                best = a
        if best is not None:
            out.append(best)
            used.add((best[0], best[1]))
    return out


def dag_oracle_residency_greedy(env: WPREnv) -> list[tuple[int, int, int, int]]:
    """DAG-oracle residency greedy：可访问 oracle future demand 的强启发式基线。"""

    future = env.oracle_dag_demand_target()
    out = []
    used = set()
    for g in sorted(env.idle_gpus()):
        best = None
        best_score = -np.inf
        for a in env.feasible_actions_for_gpu(g, used):
            slot, sid, mid, gid = a
            wf = env.active[slot]
            stage = wf.template.stages[sid]
            slack = wf.arrival + wf.template.deadline - env.time
            prep = env.prep_time(mid, gid)
            duration = env.expected_exec_time(slot, sid, mid, gid)
            resident_bonus = 1.0 if env.resident_model[gid] == mid else 0.0
            score = 2.0 * wf.template.weight / max(0.5, slack) + 1.4 * future[mid] + 0.8 * resident_bonus - 0.35 * prep - 0.22 * duration + 0.08 * stage.work
            if score > best_score:
                best_score = float(score)
                best = a
        if best is not None:
            out.append(best)
            used.add((best[0], best[1]))
    return out


def lookahead_search_upper_reference(seed: int, horizon: float = 18.0, arrival_rate: float = 0.18) -> float:
    """小规模 bounded beam-search reference。

    这不是严格 optimal/MILP。它枚举 WAIT 和每台 GPU 的 top-k dispatch 候选，并用
    beam width 控制复杂度，适合作为 lookahead upper reference 或 sanity reference。
    """

    def clone_env(e: WPREnv) -> WPREnv:
        return copy.deepcopy(e)

    def state_score(e: WPREnv) -> float:
        active_potential = sum(wf.template.weight for wf in e.active)
        return float(e.completed_value + 0.15 * active_potential - 0.2 * len(e.dropped_workflows) - 0.2 * len(e.rejected_workflows))

    def candidate_sets(e: WPREnv, top_k: int = 3, max_sets: int = 24) -> list[list[tuple[int, int, int, int]]]:
        per_gpu: list[list[tuple[int, int, int, int]]] = []
        for g in sorted(e.idle_gpus()):
            scored = []
            for a in e.feasible_actions_for_gpu(g):
                slot, sid, mid, gid = a
                wf = e.active[slot]
                slack = wf.arrival + wf.template.deadline - e.time
                score = wf.template.weight / max(0.5, slack) - 0.2 * e.prep_time(mid, gid) - 0.15 * e.expected_exec_time(slot, sid, mid, gid)
                scored.append((score, a))
            gpu_actions = [a for _, a in sorted(scored, reverse=True)[:top_k]]
            per_gpu.append(gpu_actions)
        if not per_gpu:
            return [[(-1, -1, -1, -1)]] if e.has_future_external_event() else [[]]
        sets: list[list[tuple[int, int, int, int]]] = [[]]
        for cand in per_gpu:
            new_sets: list[list[tuple[int, int, int, int]]] = []
            for base in sets:
                used = {(a[0], a[1]) for a in base if a[0] >= 0}
                for a in cand:
                    if a[0] < 0 or (a[0], a[1]) not in used:
                        new_sets.append(base + [a])
            sets = new_sets[:max_sets]
        if e.has_future_external_event():
            sets.append([(-1, -1, -1, -1)])
        return sets or [[(-1, -1, -1, -1)]]

    init = WPREnv(horizon=horizon, arrival_rate=arrival_rate, max_active=3, seed=seed)
    init.reset(seed)
    beam = [init]
    for _ in range(60):
        expanded = []
        for env in beam:
            if env.done:
                expanded.append(env)
                continue
            for assn in candidate_sets(env):
                nxt = clone_env(env)
                try:
                    nxt.step(assn)
                except RuntimeError as exc:
                    if "Zero-time transition without state change" not in str(exc):
                        raise
                    continue
                expanded.append(nxt)
        if not expanded:
            break
        expanded.sort(key=state_score, reverse=True)
        beam = expanded[:28]
        if all(env.done for env in beam):
            break
    best = 0.0
    for env in beam:
        env = clone_env(env)
        while not env.done:
            env.step(dag_oracle_residency_greedy(env))
        best = max(best, env.final_metrics()["weighted_completed_value"])
    return float(best)


def run_policy_episode(env: WPREnv, policy) -> dict[str, float]:
    env.reset(env.seed)
    while not env.done:
        env.step(policy(env))
    return env.final_metrics()

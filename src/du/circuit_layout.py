from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

N_RESTARTS = 400
SOFTMAX_TEMP = 3.0     # >0: how greedily the walk follows the best edge

WONG_BLUE = "#0072B2"
WONG_ORANGE = "#E69F00"
WONG_VERMILION = "#D55E00"
GRAY = "#D0D0D0"


def n_system_qubits(T: float, x_min: float = 1) -> int:
    return math.floor(T + 1 - x_min) + math.ceil(2 * T)


def x_min_for_depth(T: float, lc_depth: float, x_floor: float = 1) -> float:
    # campaign convention: fixed depth into the light cone, so x_min rides
    # up with T (clamped at x_floor for shallow circuits)
    return max(T - lc_depth, x_floor)


def _device_tables(backend) -> dict:
    target = backend.target
    nq = target.num_qubits

    edge_err: dict[tuple[int, int], float] = {}
    for pair in target["cz"]:  # miami / Nighthawk is CZ-native
        props = target["cz"][pair]
        if props is not None and props.error is not None:
            a, b = sorted(pair)
            e = float(props.error)
            if e < 0.5:  # error ~1.0 is a calibration placeholder: edge is dead
                edge_err[(a, b)] = min(e, edge_err.get((a, b), 1.0))

    adj: dict[int, set[int]] = {q: set() for q in range(nq)}
    for a, b in edge_err:
        adj[a].add(b)
        adj[b].add(a)

    t2 = np.array([target.qubit_properties[q].t2 or 0.0 for q in range(nq)])
    t2_bad = 0.2 * float(np.median(t2[t2 > 0]))

    ro = np.zeros(nq)
    if "measure" in target.operation_names:
        for (q,), props in target["measure"].items():
            if props is not None and props.error is not None:
                ro[q] = float(props.error)

    good = {q for q in range(nq) if t2[q] >= t2_bad and adj[q]}
    return {"adj": adj, "edge_err": edge_err, "ro": ro, "good": good, "t2": t2}


def _edge(e, a, b):
    return e[(min(a, b), max(a, b))]


def _cost(tables, chain_edges, partner_edges, qubits) -> float:
    c = sum(-np.log1p(-_edge(tables["edge_err"], a, b)) for a, b in chain_edges)
    c += sum(-np.log1p(-_edge(tables["edge_err"], a, b)) for a, b in partner_edges)
    c += sum(-np.log1p(-min(tables["ro"][q], 0.5)) for q in qubits)
    return float(c)


def _grid_coords(nq: int) -> dict[int, tuple[float, float]]:
    # miami / Nighthawk is a 12x10 square lattice, qubit = 10*row + col
    return {q: (q // 10, -(q % 10)) for q in range(nq)}


def _draw_map(ax, coords, edges, edge_col, edge_w, node_col, outline=None,
              outline_color=WONG_VERMILION):
    for a, b in edges:
        (x1, y1), (x2, y2) = coords[a], coords[b]
        ax.plot([x1, x2], [y1, y2], color=edge_col[(a, b)],
                linewidth=edge_w[(a, b)], zorder=1)
    xs = [coords[q][0] for q in coords]
    ys = [coords[q][1] for q in coords]
    cols = [node_col[q] for q in coords]
    lws = [2.2 if outline and q in outline else 0.0 for q in coords]
    ax.scatter(xs, ys, s=150, c=cols, edgecolors=outline_color, linewidths=lws,
               zorder=2)
    for q, (x, y) in coords.items():
        ax.annotate(str(q), (x, y), ha="center", va="center_baseline",
                    fontsize=5, zorder=3, color="white")
    ax.set_xlim(min(x for x, _ in coords.values()) - 0.6,
                max(x for x, _ in coords.values()) + 0.6)
    ax.set_ylim(min(y for _, y in coords.values()) - 0.6,
                max(y for _, y in coords.values()) + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")


@dataclass
class CircuitLayout:
    system: list[int]
    partners: list[int]
    backend_name: str
    complete: bool
    cost: float = field(default=float("nan"))

    @property
    def n_system(self) -> int:
        return len(self.system)

    @property
    def qubits(self) -> list[int]:
        return list(self.system) + list(self.partners)

    @property
    def chain_edges(self) -> list[tuple[int, int]]:
        return [tuple(sorted(p)) for p in zip(self.system, self.system[1:])]

    @property
    def partner_edges(self) -> list[tuple[int, int]]:
        return [tuple(sorted(p)) for p in zip(self.system, self.partners)]

    def initial_layout(self) -> list[int]:
        # build_circuit register order: system 0..n-1, partners n..2n-1
        return self.qubits

    def score(self, backend) -> float:
        # -log(success) of one traversal: chain CZs + Bell-prep CXs +
        # readout on every layout qubit. Re-run against fresh calibration to
        # detect drift; lower is better, values only comparable per length.
        self.cost = _cost(_device_tables(backend), self.chain_edges,
                          self.partner_edges, self.qubits)
        return self.cost

    def health(self, backend) -> dict:
        t = _device_tables(backend)
        return {
            "chain_e2q": [_edge(t["edge_err"], a, b) for a, b in self.chain_edges],
            "partner_e2q": [_edge(t["edge_err"], a, b) for a, b in self.partner_edges],
            "readout": [t["ro"][q] for q in self.qubits],
            "t2": [t["t2"][q] for q in self.qubits],
            "bad_qubits": [q for q in self.qubits if q not in t["good"]],
        }

    def validate(self, backend, T: float, x_min: float = 1) -> None:
        # zero-swap check: with routing disabled, the transpiler raises iff
        # the doubled circuit cannot be pinned to this layout as-is
        from qiskit import generate_preset_pass_manager
        from qiskit.transpiler.exceptions import TranspilerError
        from du.simulation import build_circuit

        if n_system_qubits(T, x_min) > self.n_system:
            raise ValueError(f"layout too short for T={T}, x_min={x_min}: "
                             f"{self.n_system} < {n_system_qubits(T, x_min)}")
        qc = build_circuit(T, x_min)
        n = qc.num_qubits // 2
        layout = self.system[:n] + self.partners[:n]
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend,
                                          initial_layout=layout,
                                          routing_method="none")
        try:
            pm.run(qc)
        except TranspilerError as exc:
            raise ValueError(f"layout cannot host the T={T} circuit without "
                             f"routing: {exc}") from exc

    def plot(self, backend, filename: str | None = None,
             error_map: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        coords = _grid_coords(backend.target.num_qubits)
        edges = sorted({tuple(sorted(p)) for p in backend.coupling_map})
        tables = _device_tables(backend)
        ch_e, pt_e = set(self.chain_edges), set(self.partner_edges)

        fig, axes = plt.subplots(1, 2 if error_map else 1,
                                 figsize=(13 if error_map else 6.5, 6.5),
                                 squeeze=False)
        ax = axes[0, 0]
        node_col = {q: WONG_BLUE if q in set(self.system)
                    else WONG_ORANGE if q in set(self.partners) else GRAY
                    for q in coords}
        edge_col = {e: WONG_BLUE if e in ch_e
                    else WONG_ORANGE if e in pt_e else GRAY
                    for e in edges}
        edge_w = {e: 2.4 if e in ch_e or e in pt_e else 0.8 for e in edges}
        _draw_map(ax, coords, edges, edge_col, edge_w, node_col)
        ax.set_title("circuit layout")
        handles = [
            Line2D([0], [0], marker="o", color=WONG_BLUE, markersize=9,
                   linewidth=2, label="system register"),
            Line2D([0], [0], marker="o", color=WONG_ORANGE, markersize=9,
                   linewidth=2, label="partner register"),
            Line2D([0], [0], marker="o", color=GRAY, markersize=9,
                   linewidth=1, label="unused"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True,
                  framealpha=0.95, facecolor="white", edgecolor="black",
                  fontsize=10, fancybox=True, borderpad=0.8)

        if error_map:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import LogNorm

            ax2 = axes[0, 1]
            e_err = tables["edge_err"]
            live = [e for e in edges if e in e_err]
            enorm = LogNorm(vmin=min(e_err[e] for e in live),
                            vmax=max(e_err[e] for e in live))
            rnorm = LogNorm(vmin=max(min(tables["ro"][tables["ro"] > 0]), 1e-4),
                            vmax=max(tables["ro"]))
            ecmap, rcmap = plt.cm.viridis, plt.cm.magma
            edge_col2 = {e: ecmap(enorm(e_err[e])) if e in e_err else "#B22222"
                         for e in edges}
            edge_w2 = {e: 2.0 if e in e_err else 1.0 for e in edges}
            node_col2 = {q: rcmap(rnorm(max(tables["ro"][q], 1e-4)))
                         for q in coords}
            _draw_map(ax2, coords, edges, edge_col2, edge_w2, node_col2,
                      outline=set(self.qubits))
            ax2.set_title("calibration errors (utilised qubits outlined)")
            cb1 = fig.colorbar(ScalarMappable(norm=enorm, cmap=ecmap), ax=ax2,
                               fraction=0.03, pad=0.01)
            cb1.ax.set_title("CZ", fontsize=8, pad=10)
            cb2 = fig.colorbar(ScalarMappable(norm=rnorm, cmap=rcmap), ax=ax2,
                               fraction=0.03, pad=0.08)
            cb2.ax.set_title("readout", fontsize=8, pad=10)

        fig.tight_layout()
        if filename:
            from du.plotting import save_figure
            save_figure(fig, filename)
        return fig

    def to_dict(self) -> dict:
        return {"system": list(self.system), "partners": list(self.partners),
                "backend_name": self.backend_name, "complete": self.complete,
                "cost": self.cost}

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitLayout":
        return cls(system=list(d["system"]), partners=list(d["partners"]),
                   backend_name=d["backend_name"], complete=bool(d["complete"]),
                   cost=float(d.get("cost", float("nan"))))


def find_circuit_layout(backend, n_system: int, n_restarts: int = N_RESTARTS,
                        seed: int = 0) -> CircuitLayout | None:
    tables = _device_tables(backend)
    adj, edge_err, good = tables["adj"], tables["edge_err"], tables["good"]
    rng = np.random.default_rng(seed)
    starts = sorted(good)
    best = None

    for _ in range(n_restarts):
        start = starts[rng.integers(len(starts))]
        system, partners, used = [], [], set()

        def claim_partner(site):
            # prefer the partner that is least useful as future chain
            cands = [q for q in adj[site] if q in good and q not in used]
            if not cands:
                return None
            cands.sort(key=lambda q: (len([n for n in adj[q]
                                           if n in good and n not in used]),
                                      _edge(edge_err, site, q)))
            return cands[0]

        node = start
        while True:
            used.add(node)
            partner = claim_partner(node)
            if partner is None:
                used.discard(node)
                break
            system.append(node)
            partners.append(partner)
            used.add(partner)
            if len(system) == n_system:
                break
            nxt = [q for q in adj[node] if q in good and q not in used]
            if not nxt:
                break
            w = np.array([-np.log(_edge(edge_err, node, q)) for q in nxt])
            p = np.exp(SOFTMAX_TEMP * (w - w.max()))
            node = nxt[rng.choice(len(nxt), p=p / p.sum())]

        if len(system) < 2:
            continue
        cand = CircuitLayout(system=system, partners=partners,
                             backend_name=backend.name,
                             complete=len(system) == n_system)
        cand.cost = _cost(tables, cand.chain_edges, cand.partner_edges,
                          cand.qubits)
        if (best is None or cand.n_system > best.n_system
                or (cand.n_system == best.n_system and cand.cost < best.cost)):
            best = cand

    return best

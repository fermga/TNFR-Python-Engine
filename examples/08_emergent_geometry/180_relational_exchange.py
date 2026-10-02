"""Conditional relational execution and three read-only capture certificates.

The graph and initial contrast are supplied. This illustrates shared engine/SDK
execution and retained numerical defects, not autonomous formation or physics.
Separate prepared two-ring snapshots compare sufficient ideal-law certificates
without running another trajectory or changing a frozen experimental verdict.
"""

from __future__ import annotations

import json
import math

import networkx as nx

from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict


def capture_comparison():
    """Compare independent theorem scopes on supplied asymmetric snapshots."""
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    phase = (4 * math.pi / 5, -4 * math.pi / 5, -2 * math.pi / 5, 0, 2 * math.pi / 5)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=phase[node % 5], nu_f=1.0)
    # One supplied contrast breaks exact copy/reflection without projecting it
    # away. The full-state local and sector theorems can still be admitted.
    graph.nodes[7]["EPI"] = 1 / 1024
    cycles = (tuple(range(5)), tuple(range(5, 10)))
    rows = []
    for preparation, beta, heterogeneous in (
        ("asymmetric_unit_capacity_and_storage_scale", 1.0, False),
        ("same_form_phase_positive_heterogeneous_capacity_beta_2", 2.0, True),
    ):
        snapshot = graph.copy()
        if heterogeneous:
            for node in snapshot:
                snapshot.nodes[node]["nu_f"] = 1.0 + node / 16
        network = Network(snapshot)
        model = RelationalExchangeModel(beta, phase_domain="positive_resultant")
        certificates = {
            "reflected": network.relational_capture(model, cycles=cycles),
            "local": network.relational_local_capture(model, cycles=cycles),
            "sector": network.relational_sector_capture(model, cycles=cycles),
        }
        summaries = {}
        for name, certificate in certificates.items():
            # The shared projection retains complete exact evidence if saved;
            # this console comparison prints only the independent verdicts.
            projection = relational_report_to_dict(certificate)
            report = projection["report"]
            summaries[name] = {
                "report_type": projection["report_type"],
                "status": report["status"],
                "target_sector": report["target_sector"],
                "unavailable_reasons": report["unavailable_reasons"],
            }
        rows.append({"preparation": preparation, "certificates": summaries})
    return {
        "supplied_support": "two_C5_rings_with_two_matching_adjacent_port_bridges",
        "supplied_state": "aligned_unit_twist_and_one_form_contrast_1_over_1024",
        "snapshots": rows,
        "scope": "ideal_continuation_from_each_snapshot_no_additional_trajectory_or_frozen_verdict_revision",
    }


def main():
    graph = nx.path_graph(3)
    for node, form, capacity in ((0, 0.1, 1.0), (1, 0.0, 0.0), (2, -0.1, 2.0)):
        graph.nodes[node].update(EPI=form, theta=0.0, nu_f=capacity)
    network = Network(graph)
    model = RelationalExchangeModel(storage_scale=1.0)
    initial = network.relational_exchange(model)
    records = []
    for _ in range(4):
        step = network.step_relational(model, dt=0.01)
        records.append(
            {
                "structural_time": step.t_after,
                "storage": float(step.after.storage),
                "energy_change": float(step.energy_change),
                "euler_storage_defect": float(step.energy_step_defect),
                "represented_work_residual": float(step.after.balance_residual),
            }
        )
    pattern = network.relational_pattern(
        model,
        reference_phase={node: 0.0 for node in graph},
        regions=((0, 1), (2,)),
    )
    # A zero capacity excludes the older positive-full-capacity transport
    # metric. The new boundary rates remain available for region (2,), since
    # its own capacity is positive; region (0, 1) keeps cut/work evidence but
    # explicitly marks division by its zero capacity unavailable.
    # Both regions also retain phase_response: actual unweighted rate, shared
    # cut contribution, mobility/form covariance and exact rounding residual.
    # Its squared covariance bound is derived, not a calibrated threshold.
    regions = relational_report_to_dict(pattern)["report"]["regions"]
    print(
        json.dumps(
            {
                "model": "conditional_capacity_separable_relational_exchange",
                "preparation": "unit_P3; supplied_form; capacities_1_0_2; phase_consensus",
                "initial_phase_rate": initial.phase_rate,
                "steps": records,
                "regional_observations": regions,
                "read_only_capture_comparison": capture_comparison(),
                "scope": "finite_unclipped_Euler_execution_not_formation_or_physical_selection",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""One reviewed post-hoc rational diagnostic; no scientific producer imports."""
from fractions import Fraction as Q
from pathlib import Path
import time

from tnfr.research.artifact_io import (
    decode_exact_tree, encode_exact_tree, exact_record, read_bytes_bounded,
    sha256_bytes, write_json_once,
)
from tnfr.utils.io import json_dumps, json_loads

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREFIX = 'artifacts/formed-two-port-attempt-20261029/'
EXPECTED = {
    PREFIX+'attempt/admission.json': (37118, 'abc9aedd426f7bbd1319453669659cc6e8784bf3fb2edab5ad78b08f305fe112'),
    PREFIX+'attempt/steps/0003.json': (34413, 'b9fc7c6946db8dfe7c42b16dbdd17e5a1f26560cb5e8154bbcae406377d6ce60'),
    PREFIX+'attempt/steps/certificate-fields.json': (12410, '31b008de3160b6e31ee1f6efc56df4b26a1f2d7572aea6298acfb4adb17754bc'),
    PREFIX+'attempt/failed-step.json': (7403, 'd25f655ae4831571e4653f6dc3edb9bcda1cd29e0fd9b141aa44001be6b67df6'),
    PREFIX+'audit.json': (2706, '73457c3783ec9fcc8620d8b2eb23cc7d8c3a5e09b932cfc16e76db8a1eeb9010'),
    PREFIX+'producing-protocol.json': (172037, '94b7e48015a3c7767996a1c41ad355f96fd69887a26b01819c13191e829f125a'),
    'src/tnfr/research/formed_two_port_reference.py': (20488, '0a141ab0ec38b468a96a4ddc65e09884fe2687e34e26e2f341f99eb950cf1345'),
    'src/tnfr/mathematics/_validated_taylor.py': (19411, '404c9914862e03c1e7da69dd60ecffe9e4208a02521a59bb8aac380a59ac1863'),
    'src/tnfr/mathematics/_rational_interval.py': (11345, 'fa711fdb6405c26f5ab62cae2d72bcf8f803aa970f40827e50b9406f0b50bb00'),
}
SCALE = 1 << 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def down(value):
    return Q((value.numerator * SCALE) // value.denominator, SCALE)


def up(value):
    return Q(-((-value.numerator * SCALE) // value.denominator), SCALE)


def pair(row):
    require(isinstance(row, dict) and set(row) == {'lo', 'hi'}, 'interval shape')
    lo, hi = exact_record(row['lo']), exact_record(row['hi'])
    require(lo <= hi, 'reversed interval')
    return lo, hi


def box(rows):
    require(len(rows) == 18, 'eighteen coordinates required')
    return tuple(pair(row) for row in rows)


def observed(value):
    return {'exact': value, 'approximate_display': format(float(value), '.12g')}


def edge_name(edge):
    def node(i):
        return ('D' if i < 9 else 'R') + str(i % 9)
    return node(edge[0]) + '->' + node(edge[1])


def analyze(label, values, bases, edges, pi, rho, radius, projection):
    rows = []
    for index, ((left, right), base) in enumerate(zip(edges, bases, strict=True)):
        lo = base[0] + values[right][0] - values[left][1]
        hi = base[1] + values[right][1] - values[left][0]
        # Same scalar rounding sequence as I(base) + right - left.
        dlo = down(down(base[0] + values[right][0]) - values[left][1])
        dhi = up(up(base[1] + values[right][1]) - values[left][0])
        raw = pi[0] / 2 - max(abs(lo), abs(hi))
        owner_raw = down(down(pi[0] / 2) - max(abs(dlo), abs(dhi)))
        # Fixed corners, opposite primitive endpoints: sufficient witnesses,
        # not an inference from a negative lower bound alone.
        positive_corner_upper = pi[1]/2 - (base[0] + values[right][1] - values[left][0])
        negative_corner_upper = pi[1]/2 + (base[1] + values[right][0] - values[left][1])
        corner_upper = min(positive_corner_upper, negative_corner_upper)
        rows.append({
            'index': index, 'edge': edge, 'label': edge_name(edge),
            'exact_gap': {'lo': lo, 'hi': hi},
            'owner_outward_gap': {'lo': dlo, 'hi': dhi},
            'exact_raw_margin_lower': observed(raw),
            'exact_guard_slack_lower': observed(raw-rho),
            'owner_raw_margin_lower': observed(owner_raw),
            'owner_guard_slack_lower': observed(owner_raw-rho),
            'box_gap_midpoint': observed((lo+hi)/2),
            'box_gap_halfwidth': observed((hi-lo)/2),
            'carried_ball_edge_uncertainty_bound': observed((projection[left]+projection[right])*radius),
            'chosen_corner_raw_margin_upper': observed(corner_upper),
            'chosen_corner_guard_slack_upper': observed(corner_upper-rho),
            'box_corner_guard_failure_certified': corner_upper-rho <= 0,
            'box_corner_nonacute_certified': corner_upper <= 0,
        })
    minimum = min(row['owner_guard_slack_lower']['exact'] for row in rows)
    return {
        'label': label,
        'minimum_owner_guard_slack': observed(minimum),
        'minimizing_edges': [row['label'] for row in rows if row['owner_guard_slack_lower']['exact'] == minimum],
        'nonpositive_guard_edges': [row['label'] for row in rows if row['owner_guard_slack_lower']['exact'] <= 0],
        'nonpositive_raw_margin_edges': [row['label'] for row in rows if row['owner_raw_margin_lower']['exact'] <= 0],
        'maximum_coordinate_halfwidth': observed(max((hi-lo)/2 for lo, hi in values)),
        'edges': rows,
    }


def main():
    started = time.monotonic()
    output = OUT/'stored-tube-diagnostic.json'
    require(not output.exists(), 'diagnostic output already exists; no retry')
    records, receipts = {}, []
    for name, (size, digest) in EXPECTED.items():
        raw = read_bytes_bounded(ROOT/name, max_bytes=512*1024)
        require(len(raw) == size and sha256_bytes(raw) == digest, 'input changed: '+name)
        receipts.append({'path': name, 'bytes': size, 'sha256': digest})
        if name.endswith('.json'):
            records[name.removeprefix(PREFIX)] = decode_exact_tree(json_loads(raw))
    admission, last, common, failed, audit, protocol = (
        records[name] for name in ('attempt/admission.json', 'attempt/steps/0003.json',
        'attempt/steps/certificate-fields.json', 'attempt/failed-step.json', 'audit.json',
        'producing-protocol.json')
    )
    frozen = {row['path']: row for row in protocol['sources']}
    for row in receipts:
        if row['path'].startswith('src/'):
            require(frozen[row['path']] == row, 'mathematical source differs from producing freeze')
    require(audit['verified_prefix_steps'] == 4 and audit['source_associations_verified'] is True, 'retained audit scope')
    require(failed['index'] == 4 and failed['reason'] == 'formed_two_port_whole_tube_margin_unavailable', 'different failure')
    require(last['index'] == 3 and last['source'] == failed['source'] == admission['source'], 'source/step association')
    require(failed['previous_complete_step'] == {'path':'steps/0003.json', 'bytes':34413,
        'sha256':EXPECTED[PREFIX+'attempt/steps/0003.json'][1]}, 'last receipt')
    cert = {**last['certificate'], **common['fields']}
    require(cert['time'] == Q(3,4) and cert['duration'] == Q(1,4), 'last step clock')
    center = tuple(exact_record(v) for v in cert['endpoint_center'])
    radius = exact_record(cert['endpoint_radius'])
    require(tuple(failed['initial_center']) == center and failed['initial_radius'] == radius, 'failed step reset source')
    ref = admission['reference']
    edges = tuple(tuple(edge) for edge in ref['oriented_edges'])
    expected_edges = tuple((s+j, s+(j+1)%9) for s in (0,9) for j in range(9)) + ((4,13),(5,14))
    require(edges == expected_edges, 'original edge geometry')
    pi, rho = pair(ref['pi']), exact_record(ref['reference_margin'])
    require(rho == Q(1,2048), 'rho changed')
    bases = tuple(pair(v) for v in ref['base_angles'])
    require(len(bases) == 20, 'twenty retained base angles')
    projection = tuple(exact_record(v) for v in cert['coordinate_projection_factors'])
    require(len(center) == len(projection) == 18 and radius >= 0, 'ball shape')
    boxes = (box(cert['tube']), box(cert['endpoint']), box(failed['failed_tube']))
    reconstructed_endpoint = tuple((down(c-p*radius), up(c+p*radius))
        for c,p in zip(center,projection,strict=True))
    require(boxes[1] == reconstructed_endpoint, 'endpoint projection does not match retained ball')
    labels = ('last_certified_whole_time_tube_3over4_to_1',
              'starting_endpoint_box_at_1', 'rejected_unvalidated_candidate_1_to_5over4')
    results = [analyze(label, values, bases, edges, pi, rho, radius, projection)
               for label, values in zip(labels, boxes, strict=True)]
    require(tuple(row['owner_guard_slack_lower']['exact'] for row in results[0]['edges'])
            == tuple(cert['domain_lower_bounds']), 'reconstructed last-tube domain differs from recorded domain')
    require(results[0]['minimum_owner_guard_slack']['exact'] > 0, 'last tube not admitted')
    require(results[2]['minimum_owner_guard_slack']['exact'] <= 0, 'retained failed-domain reason not reproduced')
    result = {
        'schema': 'tnfr.retained-three-box-diagnostic.v1', 'post_hoc': True,
        'scientific_producer_calls': 0, 'new_reference_steps': 0, 'new_target_sign_admissions': 0,
        'input_receipts': receipts, 'carried_metric_radius': observed(radius),
        'radius_over_rho': observed(radius/rho),
        'maximum_carried_coordinate_uncertainty': observed(max(p*radius for p in projection)),
        'failed_candidate_contains_starting_box': all(fl <= sl <= sh <= fh
            for (fl,fh),(sl,sh) in zip(boxes[2],boxes[1],strict=True)),
        'box_observations': results, 'actual_trajectory_exit_proved': False,
        'actual_source_nonentry_proved': False, 'iteration_count_known': False,
        'boundary': 'Retained-box rational geometry only. Rejected candidate is not a certified trajectory enclosure; lower-bound failure, inadmissible corner and actual orbit are distinct.',
    }
    require(time.monotonic()-started < 25, 'diagnostic work budget exhausted')
    data = (json_dumps(encode_exact_tree(result), indent=2, allow_nan=False)+'\n').encode()
    require(len(data) <= 256*1024, 'diagnostic output quota')
    write_json_once(output, result)
    within = time.monotonic()-started < 30
    print({'output': str(output), 'boxes':3, 'edges_per_box':20,
           'post_write_within_30_seconds':within, 'scientific_producer_calls':0})
    return 0 if within else 2


if __name__ == '__main__':
    raise SystemExit(main())

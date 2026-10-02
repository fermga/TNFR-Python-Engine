"""Internal phase depends on a frame; the inherited nodal dynamics does not.

These are exact coordinate identities on the retained pure-EPI prism. Local
orthogonal frame choices transform the edge transport as well as coordinates.
They neither transform primitive theta nor install phase or connection laws.
The time-dependent controls differentiate symbolic curves, not engine runs.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    GRAM,
    INDUCED,
    LIFT,
    PROJECTION,
    _exact_generator,
    _graph,
)
from tnfr.physics.support_transport import observe_support_transport


def _chart():
    s = pytest.importorskip("sympy")
    source = observe_support_transport(_graph())
    scale = s.diag(*(s.sqrt(value) for value in GRAM * 2))
    lift = s.Matrix(LIFT) * scale.inv()
    projection = scale * s.Matrix(PROJECTION)
    fine = s.Matrix(_exact_generator(source))
    generator = scale * s.Matrix(INDUCED) * scale.inv()
    assert lift.T * lift == projection * lift == s.eye(4)
    assert projection == lift.T
    assert fine * lift == lift * generator
    assert generator == s.Matrix(INDUCED)
    return s, lift, projection, fine, generator


def test_independent_orthogonal_frames_preserve_transport_energy_and_work():
    s, lift, projection, fine, generator = _chart()
    e = s.Symbol("e", positive=True)
    left = s.Matrix(
        [[s.Rational(3, 5), -s.Rational(4, 5)], [s.Rational(4, 5), s.Rational(3, 5)]]
    )
    right = s.diag(1, -1)  # O(2) includes orientation-reversing bases.
    frame = s.diag(left, right)
    edge = left.T * right
    assert frame.T * frame == s.eye(4)
    assert edge.det() == -1
    new_lift, new_projection = lift * frame, frame.T * projection
    new_generator = frame.T * (e * generator) * frame
    expected = (
        e
        * s.BlockMatrix([[-4 * s.eye(2), edge], [edge.T, -4 * s.eye(2)]]).as_explicit()
        / 3
    )
    assert new_generator == expected
    assert (e * fine * new_lift - new_lift * new_generator).applyfunc(
        s.simplify
    ) == s.zeros(6, 4)
    assert (new_projection * new_lift).applyfunc(s.simplify) == s.eye(4)

    original = s.Matrix([1, 0, 1, 0]) / 16
    displayed = frame.T * original
    assert (new_lift * displayed - lift * original).applyfunc(s.simplify) == s.zeros(
        6, 1
    )
    local, other = displayed[:2, 0], displayed[2:, 0]
    physical_alignment = original[:2, 0].dot(original[2:, 0])
    assert local.dot(edge * other) == physical_alignment
    assert local.dot(other) != physical_alignment
    # Signed area changes with local orientation; the scalar alignment does
    # not. It would be incorrect to retain raw angle differences after this
    # independent O(2) basis change or to interpret a reflection as a motion.
    squared = (local - edge * other).dot(local - edge * other)
    energy = e * (squared + 3 * displayed.dot(displayed)) / 2
    stiffness = -3 * e * generator
    assert energy == (original.T * stiffness * original)[0] / 2

    tangent = s.Matrix(
        [s.Rational(1, 7), s.Rational(-1, 5), s.Rational(1, 11), s.Rational(1, 13)]
    )
    pressure = e * generator * original
    fine_work = (e * fine * lift * original).dot(3 * lift * tangent)
    chart_work = (frame.T * pressure).dot(3 * frame.T * tangent)
    assert s.simplify(fine_work - chart_work) == 0
    assert s.simplify(chart_work - 3 * pressure.dot(tangent)) == 0
    # Pressure and tangent are transformed together. No extra force is
    # produced by the new matrix entries of the inherited edge connection.


def test_passive_local_frames_are_distinct_from_active_form_rotations():
    s, lift, _, fine, generator = _chart()
    quarter_turn = s.Matrix([[0, -1], [1, 0]])
    common = s.diag(quarter_turn, quarter_turn)
    independent = s.diag(s.eye(2), quarter_turn)
    assert common * generator == generator * common
    assert independent * generator != generator * independent
    field = s.Matrix([1, 0, 1, 0]) / 16
    assert lift * common * field != lift * field
    assert fine * lift * common * field == lift * common * generator * field
    assert fine * lift * independent * field != lift * independent * generator * field
    stiffness = -3 * generator
    assert common.T * stiffness * common == stiffness
    assert independent.T * stiffness * independent != stiffness
    before = (field.T * stiffness * field)[0] / 2
    after = (field.T * independent.T * stiffness * independent * field)[0] / 2
    assert before == s.Rational(3, 256)
    assert after == s.Rational(1, 64)
    # With the old basis held fixed, rotating coordinates actively changes
    # fine EPI. Common O(2) is a symmetry of this linear pure-EPI model only;
    # no primitive-phase source, operator or full-engine symmetry follows.


def test_moving_frames_add_an_apparent_clock_and_require_covariant_work():
    s, lift, _, _, generator = _chart()
    time = s.Symbol("time", real=True)
    e, radius, omega0, omega1 = s.symbols("e radius omega0 omega1", positive=True)
    skew = s.Matrix([[0, -1], [1, 0]])

    def rotation(angle):
        return s.Matrix([[s.cos(angle), -s.sin(angle)], [s.sin(angle), s.cos(angle)]])

    frames = (rotation(omega0 * time), rotation(omega1 * time))
    connection = frames[0].T * frames[1]
    spin = (omega0 * skew, omega1 * skew)
    assert (
        connection.diff(time) + spin[0] * connection - connection * spin[1]
    ).applyfunc(s.trigsimp) == s.zeros(2)
    # A genuine common-mode solution has no internal angular motion in the
    # fixed basis. Independent rotating displays create apparent phase drift.
    physical = radius * s.exp(-e * time) * s.Matrix([1, 0])
    displayed = tuple(frame.T * physical for frame in frames)
    for frame, angular_velocity, local, other, edge in (
        (frames[0], omega0, displayed[0], displayed[1], connection),
        (frames[1], omega1, displayed[1], displayed[0], connection.T),
    ):
        assert (frame * local - physical).applyfunc(s.trigsimp) == s.zeros(2, 1)
        covariant_rate = local.diff(time) + angular_velocity * skew * local
        inherited = e * (edge * other - 4 * local) / 3
        assert (covariant_rate - inherited).applyfunc(s.trigsimp) == s.zeros(2, 1)
        raw_angle_rate = s.det(s.Matrix.hstack(local, local.diff(time))) / local.dot(
            local
        )
        true_angle_rate = s.det(s.Matrix.hstack(local, covariant_rate)) / local.dot(
            local
        )
        assert s.simplify(raw_angle_rate) == -angular_velocity
        assert s.simplify(true_angle_rate) == 0
    rotated_lift = lift * s.diag(*frames)
    assert (
        rotated_lift * s.Matrix.vstack(*displayed)
        - lift * s.Matrix.vstack(physical, physical)
    ).applyfunc(s.trigsimp) == s.zeros(6, 1)

    # The same connection is required for work at a moving-frame jet. This
    # uses two actual fine decay modes rather than adding a forcing law.
    r, q, omega = s.symbols("r q omega", positive=True)
    state = s.Matrix([r, q, r, -q])
    pressure = e * generator * state
    frame_spin = s.diag(omega * skew, s.zeros(2))
    raw_tangent = pressure - frame_spin * state  # frames equal I at this instant
    physical_work = 3 * pressure.dot(pressure)
    naive_work = 3 * pressure.dot(raw_tangent)
    covariant_work = 3 * pressure.dot(raw_tangent + frame_spin * state)
    assert s.simplify(naive_work - physical_work) == 2 * e * omega * r * q
    assert s.simplify(covariant_work - physical_work) == 0
    # Omitting the frame velocity fabricates work and a clock. The fine
    # field, primitive theta, capacity and graph have not been transformed.

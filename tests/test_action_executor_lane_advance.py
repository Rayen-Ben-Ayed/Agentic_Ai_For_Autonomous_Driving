"""Unit tests for ActionExecutor._advance_along_lane junction freezing.

The frozen-merge lookahead must stop advancing at a junction boundary instead
of letting waypoint.next() pick an arbitrary (curving) junction connector, which
is what dragged the ego back out of the target lane after a right merge.
"""
from simulation.action_executor import ActionExecutor, LANE_ADVANCE_STEP_M


class FakeWaypoint:
    """Minimal waypoint over a 1m-resolution chain of is_junction flags."""

    def __init__(self, chain, idx):
        self._chain = chain
        self._idx = idx

    @property
    def idx(self):
        return self._idx

    @property
    def is_junction(self):
        return self._chain[self._idx]

    def next(self, dist):
        step = max(1, int(round(dist)))
        nxt = self._idx + step
        if nxt >= len(self._chain):
            return []
        return [FakeWaypoint(self._chain, nxt)]


def _executor():
    return ActionExecutor(carla_client=None)


def test_advance_freezes_at_junction_entry():
    # Road for indices 0..4, junction from index 5 onward.
    chain = [False] * 5 + [True] * 20
    wp = FakeWaypoint(chain, 0)
    out = _executor()._advance_along_lane(wp, 10.0)
    # Last pose strictly before the junction (step is 2m: 0->2->4, 6 is junction).
    assert not out.is_junction
    assert out.idx == 4


def test_advance_walks_full_distance_on_open_road():
    chain = [False] * 40
    wp = FakeWaypoint(chain, 0)
    out = _executor()._advance_along_lane(wp, 10.0)
    assert out.idx == 10


def test_advance_inside_junction_falls_back_to_next():
    # Already inside a junction: no pre-junction pose to hold, advance normally.
    chain = [True] * 40
    wp = FakeWaypoint(chain, 0)
    out = _executor()._advance_along_lane(wp, 8.0)
    assert out.idx == 8


def test_advance_zero_distance_is_identity():
    chain = [False] * 10
    wp = FakeWaypoint(chain, 3)
    out = _executor()._advance_along_lane(wp, 0.0)
    assert out is wp


def test_advance_step_constant_is_positive():
    assert LANE_ADVANCE_STEP_M > 0

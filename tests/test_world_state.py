from simulation.world_state import _actor_ahead_in_ego_lane, _actor_in_ego_lane


def _neighbor_npc():
    return {
        "same_lane": False,
        "lane_relation": "neighbor_lane",
        "type": "vehicle.audi.tt",
        "is_scenario_npc": True,
    }


def _same_lane_npc():
    return {
        "same_lane": True,
        "lane_relation": "same_lane",
        "type": "vehicle.audi.tt",
        "is_scenario_npc": True,
    }


def test_neighbor_lane_actor_not_in_ego_lane():
    assert not _actor_in_ego_lane(_neighbor_npc(), 4.32)


def test_same_lane_actor_in_ego_lane():
    assert _actor_in_ego_lane(_same_lane_npc(), 0.04)


def test_neighbor_lane_ahead_does_not_block_ego_path():
    assert not _actor_ahead_in_ego_lane(_neighbor_npc(), 17.68, 4.32, 18.0)


def test_same_lane_ahead_blocks_ego_path():
    assert _actor_ahead_in_ego_lane(_same_lane_npc(), 17.68, 0.04, 18.0)


def test_geometric_fallback_when_lane_relation_unknown():
    actor = {"lane_relation": "unknown", "type": "vehicle.audi.tt"}
    assert _actor_in_ego_lane(actor, 1.0)
    assert not _actor_in_ego_lane(actor, 4.0)


def test_different_road_still_blocks_when_geometrically_in_lane():
    """road_id mismatch must not hide an in-lane NPC (run 905 regression)."""
    actor = {
        "same_lane": False,
        "lane_relation": "different_road",
        "type": "vehicle.audi.tt",
        "is_scenario_npc": True,
    }
    assert _actor_in_ego_lane(actor, 0.04)
    assert _actor_ahead_in_ego_lane(actor, 50.0, 0.04, 70.0)

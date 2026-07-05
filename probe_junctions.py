"""Manual probe: validate junction scanning against a live CARLA map.

Requires a running CARLA server (read-only: no ticking, no actor spawning).
Walks every spawn point, reports the junction scan the agent would see, and
freezes a plan for each available exit so classification and path geometry
can be eyeballed before an agent run.

Usage: python probe_junctions.py [--host 127.0.0.1] [--port 2000] [--limit 15]
"""
import argparse

import carla

from simulation import junction_planner as jp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=15, help="Max junctions to report")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    carla_map = world.get_map()
    print(f"Connected: {carla_map.name}")

    reported = 0
    for i, spawn in enumerate(carla_map.get_spawn_points()):
        wp = carla_map.get_waypoint(
            spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        scan = jp.scan_ahead(wp)
        if scan.kind != "junction":
            continue
        options = scan.options
        preferred = jp.preferred_junction_action(options)
        print(
            f"\nspawn#{i} road={wp.road_id} lane={wp.lane_id} "
            f"yaw={wp.transform.rotation.yaw:.1f} -> junction in {scan.distance_m}m "
            f"options={[d for d, ok in options.items() if ok]} preferred={preferred}"
        )
        for action in jp.JUNCTION_ACTIONS:
            plan = jp.build_junction_plan(wp, action)
            if plan is None:
                continue
            print(
                f"  {action:>11}: {len(plan.poses)} poses, {plan.cum_s[-1]:.1f}m, "
                f"entry yaw={plan.poses[0].yaw_deg:.1f} exit yaw={plan.poses[-1].yaw_deg:.1f} "
                f"exit road={plan.exit_road_id} lane={plan.exit_lane_id}"
            )
        reported += 1
        if reported >= args.limit:
            break
    if reported == 0:
        print("No junctions within detection range of any spawn point.")


if __name__ == "__main__":
    main()

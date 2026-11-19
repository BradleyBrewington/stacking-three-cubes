"""Trusted runtime planning adapter. Reads installed geometry, never moves hardware."""
import hashlib
import json
import math
from pathlib import Path

try:
    from . import robot_collision, robot_planning
except ImportError:
    import robot_collision
    import robot_planning


def plan_to(resources, state, target_xyz_m, approach_direction, clearance_m, *,
            collision_config, axes=None, tilts_deg=None, roll_deg=None, offset_m=None):
    """Plan joint-space transit to preapproach then straight Cartesian approach.

    collision_config is trusted installed station data, not a client payload.
    Expected keys: geometry_sha256, table_z_m, uncertainty_margin_m,
    tracking_margin_m, execution_tolerance_deg, station_review and optional
    spheres. station_review must explicitly accept physical model, table,
    jaw mechanical range and exclusions. Scene/config changes invalidate plans.
    CLI owner must cache successful plans under opaque server handles, and
    execute only cached plans with matching live profile/config/code hashes.
    """
    failed = {"schema_version": 1, "status": "failed", "motion_commanded": False,
              "waypoints": [], "collision_checked": False, "current_state": state}
    if not isinstance(collision_config, dict):
        return {**failed, "reason": "collision_config_unavailable"}
    review = collision_config.get("station_review", {})
    required = ("geometry_matches_station", "table_reference_accepted",
                "jaw_range_contains_motion", "excluded_contacts_accepted")
    missing = [k for k in required if review.get(k) is not True]
    if missing:
        return {**failed, "reason": "collision_config_unreviewed", "missing_review": missing}
    jaw_range = collision_config.get("jaw_angle_range_rad")
    opening_range = collision_config.get("gripper_opening_percent_range")
    if jaw_range is not None:
        if not isinstance(opening_range, (list, tuple)) or len(opening_range) != 2:
            return {**failed, "reason": "gripper_mapping_range_unavailable"}
        lo, hi = robot_planning._vector(opening_range, 2, "gripper opening range")
        current_opening = state.get("joints", {}).get("gripper")
        if not 0 <= lo <= hi <= 100 or not isinstance(current_opening, (float, int)) or not math.isfinite(current_opening) or not lo <= current_opening <= hi:
            return {**failed, "reason": "gripper_outside_modeled_opening"}
    geometry_path = Path(__file__).with_name("so101_collision_geometry.json")
    raw = geometry_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if collision_config.get("geometry_sha256") != digest:
        return {**failed, "reason": "collision_geometry_hash_mismatch"}
    geometry = json.loads(raw)
    model, sweep = robot_collision.cad_model(
        resources, geometry, table_z_m=collision_config["table_z_m"],
        uncertainty_margin_m=collision_config["uncertainty_margin_m"],
        tracking_margin_m=collision_config["tracking_margin_m"],
        spheres=collision_config.get("spheres", []), scope=collision_config.get("scope", "full"),
        gripper_phase_deg=collision_config.get("gripper_phase_deg", 0.),
        gripper_phase_uncertainty_deg=collision_config.get("gripper_phase_uncertainty_deg", 0.),
        jaw_angle_range_rad=jaw_range,
        joint_tracking_tolerance_deg=collision_config["execution_tolerance_deg"])
    names = resources.JOINTS[:-1]
    start = {n: state["joints"][n] for n in names}
    bounds = {n: state["limits"][n] for n in names}
    if any(state.get("faults", {}).values()):
        return {**failed, "reason": "motor_fault_present"}
    target = robot_planning._vector(target_xyz_m, 3, "target")
    offset = robot_planning._vector(offset_m or [0., 0., 0.], 3, "offset")
    if math.sqrt(sum(x*x for x in offset)) > .2:
        raise ValueError("tool offset norm exceeds 0.2 metres")
    if axes is not None and tilts_deg is not None:
        raise ValueError("choose explicit axes or tilt candidates")
    if axes is None:
        # Reuse the existing radial family; no object or benchmark assumptions.
        azimuth = math.atan2(target[1], target[0]-.0388353)
        tilts = list(range(0, 91, 15)) if tilts_deg is None else list(tilts_deg)
        tilts = robot_planning._vector(tilts, len(tilts), "tilts")
        if any(not -90 <= t <= 90 for t in tilts):
            raise ValueError("tilt must be -90..90 degrees from down")
        axes = [[math.sin(math.radians(t))*math.cos(azimuth),
                 math.sin(math.radians(t))*math.sin(azimuth), -math.cos(math.radians(t))] for t in tilts]

    def solve(point, axis, seed, limits):
        return resources.solve(point, axis, seed, limits, roll=roll_deg, offset=offset)["joints"]

    def pose(q):
        transform = resources.tool_pose(q)
        return {"point": (transform[:3, 3]+transform[:3, :3]@offset).tolist(),
                "axis": transform[:3, 2].tolist()}

    plan = robot_planning.plan_target(
        start, target, approach_direction, clearance_m, axes, bounds,
        solver=solve, pose=pose, collision_check=model, collision_model=model.metadata,
        link_sweep_bound=sweep, execution_tolerance_deg=collision_config["execution_tolerance_deg"],
        joint_space_transit=True, timeout_s=collision_config.get("planning_timeout_s", 10.))
    plan["current_state"] = state
    plan["dependency_hashes"] = {"so101_collision_geometry.json": digest}
    plan["collision_config_sha256"] = hashlib.sha256(json.dumps(
        collision_config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    plan["tool_offset_m"] = offset
    if jaw_range is not None:
        plan["required_gripper_range_percent"] = list(opening_range)
    return plan

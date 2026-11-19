"""Offline Cartesian approach planning. No hardware imports or motion commands.

All checks are discrete and relative to caller-supplied geometry. A successful
plan is not a continuous collision certificate or permission to operate hardware.
"""
import math
import time


class PlanningFailure(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def _vector(value, size, name):
    if len(value) != size or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) for x in value):
        raise ValueError(name + " must contain finite numbers")
    return list(map(float, value))


def _distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def _unit(a):
    a = _vector(a, 3, "axis")
    n = _distance(a, [0, 0, 0])
    if n < 1e-12:
        raise ValueError("axis must be nonzero")
    return [x/n for x in a]


def _mix(a, b, f):
    return [x+(y-x)*f for x, y in zip(a, b)]


def plan_approach(start_joints, waypoints_xyz_m, axes, bounds, *, solver,
                  pose, collision_check=None, collision_model=None,
                  position_tolerance_m=.003, axis_tolerance_deg=5.,
                  max_joint_step_deg=1., max_cartesian_step_m=.005,
                  max_joint_segment_deg=15., max_samples=10000,
                  link_sweep_bound=None, max_link_sweep_m=.005, timeout_s=10.,
                  execution_tolerance_deg=None, joint_space_transit=False,
                  max_transit_joint_delta_deg=180.):
    """Search each caller-selected axis over the complete explicit Cartesian path.

    solver(xyz, axis, seed, bounds) -> named joint degrees (no gripper).
    pose(q) -> {point: base_link XYZ metres, axis: tool +Z in base_link}.
    collision_check(q) -> bool; True means collision-free *in the supplied model*.
    collision_model must have nonempty id and coverage strings. No model is
    assumed from camera calibration. Callbacks must be pure and deterministic.
    link_sweep_bound(a,b) must bound travel of every modeled surface point
    along linear joint interpolation and scale linearly with joint increments.

    Each candidate first transitions from measured start to its orientation at
    the same point. This transition checks position, joints and collision, but
    allows orientation change. Subsequent segments enforce the selected axis
    at all samples. IK samples follow each straight Cartesian segment; all
    connecting joint interpolations are checked against its linear corridor.
    Output includes every validated sample; execution must not shortcut them.
    Invalid input raises ValueError; planning failures return structured data.
    The deadline is cooperative: callbacks must themselves have bounded runtime.
    """
    scalars = [position_tolerance_m, axis_tolerance_deg, max_joint_step_deg,
               max_cartesian_step_m, max_joint_segment_deg, max_link_sweep_m, timeout_s,
               max_transit_joint_delta_deg]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 for v in scalars):
        raise ValueError("tolerances and resolutions must be positive finite numbers")
    if axis_tolerance_deg > 180 or type(max_samples) is not int or max_samples < 2:
        raise ValueError("invalid angular tolerance or sample budget")
    names = tuple(bounds)
    if not names or set(start_joints) != set(names):
        raise ValueError("start joints must exactly match bounds")
    checked_bounds = {}
    for name in names:
        lo, hi = _vector(bounds[name], 2, "bounds")
        if lo >= hi:
            raise ValueError("empty joint bound")
        checked_bounds[name] = (lo, hi)
    bounds = checked_bounds

    def joints(q):
        if set(q) != set(names):
            raise PlanningFailure("invalid_solution", "joint names do not match bounds")
        values = _vector([q[n] for n in names], len(names), "joints")
        if any(not bounds[n][0] <= v <= bounds[n][1] for n, v in zip(names, values)):
            raise PlanningFailure("joint_limit", "joint exceeds supplied bounds")
        return dict(zip(names, values))

    start = joints(start_joints)
    points = [_vector(p, 3, "waypoint") for p in waypoints_xyz_m]
    directions = [_unit(a) for a in axes]
    if not points or not directions:
        raise ValueError("at least one waypoint and orientation candidate required")
    result = {"schema_version": 1, "status": "failed", "reason": None,
              "motion_commanded": False, "frame": "base_link", "joint_units": "degrees",
              "collision_checked": False, "continuous_collision_checked": False,
              "collision_model": collision_model, "waypoints": [], "candidates": [],
              "resolution": {"max_joint_step_deg": max_joint_step_deg,
                             "max_cartesian_step_m": max_cartesian_step_m,
                             "max_link_sweep_m": max_link_sweep_m},
              "tolerances": {"position_m": position_tolerance_m, "axis_deg": axis_tolerance_deg}}
    if collision_check is None or not isinstance(collision_model, dict) or any(
            not isinstance(collision_model.get(k), str) or not collision_model[k].strip()
            for k in ("id", "coverage")):
        result["reason"] = "collision_model_unavailable"
        return result
    if link_sweep_bound is None:
        result["reason"] = "link_sweep_bound_unavailable"
        return result
    result["execution_tolerance_deg"] = None
    if execution_tolerance_deg is not None:
        reaches = collision_model.get("downstream_reach_m", {})
        margin = collision_model.get("tracking_margin_m", 0.)
        if set(execution_tolerance_deg) != set(names) or set(reaches) != set(names):
            raise ValueError("execution tolerance and reach bounds must name every planned joint")
        tolerances = _vector([execution_tolerance_deg[n] for n in names], len(names), "execution tolerance")
        lengths = _vector([reaches[n] for n in names], len(names), "downstream reach")
        if any(v < 0 for v in tolerances) or any(v <= 0 for v in lengths):
            raise ValueError("execution tolerances must be nonnegative and reach bounds positive")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
            raise ValueError("invalid tracking margin")
        required = sum(r*math.radians(t) for r, t in zip(lengths, tolerances))
        relative_budget = collision_model.get("joint_tracking_tolerance_deg")
        if relative_budget is not None:
            if set(relative_budget) != set(names) or any(relative_budget[n] < t for n, t in zip(names, tolerances)):
                result["reason"] = "relative_tracking_budget_exceeded"
                return result
        if collision_model.get("tracking_margin_applied") is not True or required > margin:
            result["reason"] = "tracking_margin_insufficient"
            return result
        result["execution_tolerance_deg"] = dict(zip(names, tolerances))
        result["tracking_displacement_bound_m"] = required

    def observe(q):
        p = pose(dict(q))
        return _vector(p["point"], 3, "FK point"), _unit(p["axis"])

    total_samples = 0
    deadline = time.monotonic()+timeout_s

    def check_deadline():
        if time.monotonic() >= deadline:
            raise PlanningFailure("planning_timeout", "cooperative planning deadline exceeded")

    for index, axis in enumerate(directions):
        path = []
        try:
            origin, _ = observe(start)

            def check(q, wanted, enforce_axis, enforce_position=True):
                nonlocal total_samples
                check_deadline()
                total_samples += 1
                if total_samples > max_samples:
                    raise PlanningFailure("sample_budget_exceeded", "planning sample budget exhausted")
                q = joints(q)
                actual, actual_axis = observe(q)
                if enforce_position and _distance(actual, wanted) > position_tolerance_m:
                    raise PlanningFailure("cartesian_tolerance", "interpolated path leaves Cartesian corridor")
                angle = math.degrees(math.acos(max(-1., min(1., sum(x*y for x, y in zip(axis, actual_axis))))))
                if enforce_axis and angle > axis_tolerance_deg:
                    raise PlanningFailure("axis_tolerance", "orientation infeasible along entire approach")
                clear = collision_check(dict(q))
                check_deadline()
                if type(clear) is not bool:
                    raise PlanningFailure("collision_check_invalid", "checker must return a boolean")
                if not clear:
                    raise PlanningFailure("collision", "sample rejected by supplied collision model")
                path.append(q)

            check(start, origin, False)
            previous, previous_point = start, origin
            # The first solve establishes orientation without an implicit detour.
            targets = [(points[0], False, False)] if joint_space_transit else [(origin, False, True)]
            segment_start = points[0] if joint_space_transit else origin
            for point in (points[1:] if joint_space_transit else points):
                count = max(1, math.ceil(_distance(segment_start, point)/max_cartesian_step_m))
                if count > max_samples:
                    raise PlanningFailure("sample_budget_exceeded", "Cartesian segment exceeds budget")
                targets.extend((_mix(segment_start, point, i/count), True, True) for i in range(1, count+1))
                if len(targets) > max_samples:
                    raise PlanningFailure("sample_budget_exceeded", "Cartesian path exceeds budget")
                segment_start = point
            for target_point, enforce_axis, enforce_position in targets:
                check_deadline()
                target = joints(solver(list(target_point), list(axis), dict(previous), dict(bounds)))
                check_deadline()
                delta = max(abs(target[n]-previous[n]) for n in names)
                segment_bound = max_joint_segment_deg if enforce_position else max_transit_joint_delta_deg
                if delta > segment_bound:
                    raise PlanningFailure("joint_discontinuity", "IK branch jump exceeds segment bound")
                steps = max(1, math.ceil(delta/max_joint_step_deg))
                sweep = link_sweep_bound(dict(previous), dict(target))
                if isinstance(sweep, bool) or not isinstance(sweep, (int, float)) or not math.isfinite(sweep) or sweep < 0:
                    raise PlanningFailure("invalid_sweep_bound", "link sweep bound must be finite nonnegative metres")
                steps = max(steps, math.ceil(sweep/max_link_sweep_m))
                for i in range(1, steps+1):
                    f = i/steps
                    q = {n: previous[n]+f*(target[n]-previous[n]) for n in names}
                    check(q, _mix(previous_point, target_point, f), enforce_axis or i == steps,
                          enforce_position or i == steps)
                previous, previous_point = target, target_point
            result.update(status="ok", reason=None, collision_checked=True,
                          waypoints=path, axis=axis, selected_candidate=index,
                          samples_checked=total_samples)
            result["transit_mode"] = "joint_space" if joint_space_transit else "cartesian"
            return result
        except PlanningFailure as exc:
            failure = {"index": index, "code": exc.code, "detail": str(exc)}
        except Exception as exc:
            failure = {"index": index, "code": "callback_error",
                       "detail": type(exc).__name__ + ": " + str(exc)}
        result["candidates"].append(failure)
        if failure["code"] in ("sample_budget_exceeded", "planning_timeout"):
            break
    result.update(reason="no_feasible_candidate", samples_checked=total_samples)
    return result


def resource_callbacks(resources):
    """Adapt existing robot_resources FK/IK without connecting to hardware.

    Existing configured tool translation is retained. No additional offset is
    introduced, and each solution is independently revalidated by the planner.
    """
    def solve(point, axis, seed, bounds):
        return resources.solve(point, axis, seed, bounds)["joints"]

    def pose(q):
        transform = resources.tool_pose(q)
        return {"point": transform[:3, 3].tolist(), "axis": transform[:3, 2].tolist()}

    return solve, pose


def plan_target(start_joints, target_xyz_m, approach_direction, clearance_m,
                axes, bounds, **options):
    """Generate a generic pre-approach and final straight approach.

    approach_direction points toward target in base_link; clearance_m is the
    caller-selected approach length, not a certified obstacle clearance. Both
    transit to pre-approach and final approach are checked by plan_approach.
    This is a bounded local planner, not a global obstacle-detour search.
    """
    target = _vector(target_xyz_m, 3, "target")
    direction = _unit(approach_direction)
    if isinstance(clearance_m, bool) or not isinstance(clearance_m, (int, float)) or not math.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("clearance_m must be positive and finite")
    pre = [p-clearance_m*d for p, d in zip(target, direction)]
    result = plan_approach(start_joints, [pre, target], axes, bounds, **options)
    result["requested_path"] = {"target_xyz_m": target, "approach_direction": direction,
                                "approach_length_m": clearance_m, "preapproach_xyz_m": pre}
    return result

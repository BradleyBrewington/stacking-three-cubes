"""Task-independent execution of caller-planned joint paths.

No connection, torque enable, reset, or force-limit changes. Caller must hold the
station's exclusive writer lease and supply live authorization/fault callbacks.
Interpolation is piecewise linear; this module does not certify collisions,
acceleration, Cartesian accuracy, or object retention. Bus calls must have bounded
transport timeouts: a Python deadline cannot interrupt a blocked device driver.
"""
import math
import time
import copy
import hashlib
import json
from contextlib import contextmanager


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positions(q, bounds, names):
    if not isinstance(q, dict) or set(q) != set(names):
        raise ValueError("Joint names must be complete and consistent")
    for j in names:
        if not _number(q[j]) or not bounds[j][0] <= q[j] <= bounds[j][1]:
            raise ValueError("Nonfinite or out-of-bounds joint: " + j)
    return dict(q)


def validate_trajectory(waypoints, bounds, max_speed):
    """Copy/validate entire path before any write; times are seconds, joints deg/%"""
    if not isinstance(waypoints, (list, tuple)) or not 2 <= len(waypoints) <= 10000:
        raise ValueError("Expected 2..10000 waypoints")
    names = tuple(waypoints[0]["joints"])
    if not names:
        raise ValueError("Empty joint path")
    for j in names:
        b = bounds.get(j)
        if (not isinstance(b, (list, tuple)) or len(b) != 2 or
                not all(_number(v) for v in b) or b[0] >= b[1]):
            raise ValueError("Missing/invalid accepted joint bounds: " + j)
        if not _number(max_speed.get(j)) or max_speed[j] <= 0:
            raise ValueError("Missing/invalid speed bound: " + j)
    result = []
    for w in waypoints:
        t = w["time_s"]
        q = _positions(w["joints"], bounds, names)
        if not _number(t) or t < 0 or t > 60:
            raise ValueError("Waypoint times must be finite, 0..60 seconds")
        if result:
            dt = t - result[-1]["time_s"]
            if dt <= 0:
                raise ValueError("Waypoint times must strictly increase")
            if any(abs(q[j] - result[-1]["joints"][j]) / dt > max_speed[j] + 1e-9 for j in names):
                raise ValueError("Segment exceeds joint speed bound")
        elif t != 0:
            raise ValueError("First waypoint must have time_s=0")
        result.append({"time_s": float(t), "joints": q})
    return result


def time_parameterize(joint_path, max_speed, minimum_segment_s=0.05):
    """Preserve every planner sample; add conservative linear segment times."""
    if not _number(minimum_segment_s) or minimum_segment_s <= 0:
        raise ValueError("Invalid minimum segment time")
    if not isinstance(joint_path, (list, tuple)) or len(joint_path) < 2:
        raise ValueError("Expected at least two joint samples")
    names = set(joint_path[0])
    if not names or any(not _number(max_speed.get(j)) or max_speed[j] <= 0 for j in names):
        raise ValueError("Missing positive speed limits")
    result, elapsed = [], 0.
    for q in joint_path:
        if set(q) != names or not all(_number(v) for v in q.values()):
            raise ValueError("Invalid joint sample")
        if result:
            elapsed += max(minimum_segment_s, max(abs(q[j]-result[-1]["joints"][j])/max_speed[j] for j in names))
        result.append({"time_s": elapsed, "joints": dict(q)})
    return result


class _Abort(Exception):
    pass


def identity(value):
    """SHA-256 of canonical JSON, rejecting NaN; provenance, not authorization."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


@contextmanager
def bounded_transport(bus, *, io_timeout_s=.1, clock=time.monotonic, sleep=time.sleep):
    """Execution-scoped guard for LeRobot/Feetech PortHandler, no motor writes.

    SDK 1.0.0 uses unbounded serial.write/flush and wall-clock packet deadlines.
    Bound writes, replace output flush with bounded drain polling, and use a
    monotonic packet clock. Exclusivity must cover the entire context. This
    cannot preempt an OS/kernel failure inside an ioctl/property access.
    """
    if not _number(io_timeout_s) or not .01 <= io_timeout_s <= .5:
        raise ValueError("Transport timeout must be 0.01..0.5 seconds")
    port = bus.port_handler
    serial = port.ser
    if serial is None or not all(hasattr(port, k) for k in (
            "clearPort", "getCurrentTime", "setPacketTimeout", "setPacketTimeoutMillis")):
        raise ValueError("Unsupported transport; expected connected Feetech PortHandler")
    # Validate access before changing any method or serial configuration.
    old_timeout, old_write_timeout = serial.timeout, serial.write_timeout
    if not isinstance(serial.out_waiting, int):
        raise ValueError("Transport cannot report pending output")
    methods = ("clearPort", "getCurrentTime", "setPacketTimeout", "setPacketTimeoutMillis", "readPort")
    absent = object()
    old = {name: port.__dict__.get(name, absent) for name in methods}
    original_packet_timeout = port.setPacketTimeout
    original_packet_millis = port.setPacketTimeoutMillis
    original_read = port.readPort
    packet_state = (port.packet_start_time, port.packet_timeout)
    receive_deadline = None

    def drain():
        deadline = clock()+io_timeout_s
        while serial.out_waiting:
            if clock() >= deadline:
                raise TimeoutError("Serial output drain deadline")
            sleep(min(.001, io_timeout_s))

    def packet_timeout(length):
        nonlocal receive_deadline
        original_packet_timeout(length)
        port.packet_timeout = min(port.packet_timeout, io_timeout_s*1000.)
        receive_deadline = clock()+port.packet_timeout/1000.

    def packet_millis(milliseconds):
        nonlocal receive_deadline
        if not _number(milliseconds) or milliseconds <= 0:
            raise ValueError("Invalid packet timeout")
        original_packet_millis(min(milliseconds, io_timeout_s*1000.))
        receive_deadline = clock()+port.packet_timeout/1000.

    def read(length):
        # SDK rxPacket skips its own timeout branches while continuously
        # receiving malformed/wrong-ID packets. Enforce deadline at every read.
        if receive_deadline is None or clock() >= receive_deadline:
            raise TimeoutError("Serial packet receive deadline")
        return original_read(length)

    try:
        serial.timeout = 0
        serial.write_timeout = io_timeout_s
        port.clearPort = drain
        port.getCurrentTime = lambda: clock()*1000.
        port.setPacketTimeout = packet_timeout
        port.setPacketTimeoutMillis = packet_millis
        port.readPort = read
        yield {"write_timeout_s": io_timeout_s, "packet_timeout_cap_s": io_timeout_s,
               "drain_timeout_s": io_timeout_s, "packet_clock": "monotonic"}
    finally:
        for name, value in old.items():
            if value is absent:
                port.__dict__.pop(name, None)
            else:
                setattr(port, name, value)
        port.packet_start_time, port.packet_timeout = packet_state
        serial.timeout, serial.write_timeout = old_timeout, old_write_timeout


def prepare_execution(plan, profile_sha256, *, max_speed, bounds):
    """Bind a successful sampled plan to an immutable, hash-checked envelope."""
    plan = copy.deepcopy(plan)
    if (plan.get("schema_version") != 1 or plan.get("status") != "ok"
            or plan.get("collision_checked") is not True):
        raise ValueError("Expected successful collision-checked planner output")
    model = plan.get("collision_model", {})
    if not isinstance(model, dict) or any(not isinstance(model.get(k), str) or not model[k].strip() for k in ("id", "coverage")):
        raise ValueError("Missing collision model identity/coverage")
    if (not isinstance(profile_sha256, str) or len(profile_sha256) != 64
            or any(c not in "0123456789abcdef" for c in profile_sha256)):
        raise ValueError("Expected exact profile SHA-256")
    path = time_parameterize(plan["waypoints"], max_speed)
    validate_trajectory(path, bounds, max_speed)
    payload = {"schema_version": 1, "plan": plan, "profile_sha256": profile_sha256,
               "waypoints": path, "bounds": copy.deepcopy(bounds), "max_speed": dict(max_speed)}
    return {"payload": payload, "sha256": identity(payload)}


def execute_prepared(bus, envelope, *, current_profile_sha256, authorize, **kwargs):
    """Check plan/profile identity before writes and profile identity every cycle.

    current_profile_sha256() must hash the actual installed profile bytes. Caller
    must freeze runtime/collision resources for the lease as well. Hashes do not
    authenticate a planner or certify the truth of supplied collision coverage.
    """
    envelope = copy.deepcopy(envelope)
    payload = envelope["payload"]
    if identity(payload) != envelope["sha256"]:
        raise ValueError("Execution envelope changed")
    # Rebuild from the bound planner samples so independently altered timed
    # paths/bounds cannot masquerade as the planner path.
    rebuilt = prepare_execution(payload["plan"], payload["profile_sha256"],
                                max_speed=payload["max_speed"], bounds=payload["bounds"])
    if rebuilt != envelope:
        raise ValueError("Execution envelope does not match planner path")

    def still_authorized():
        return authorize() is True and current_profile_sha256() == payload["profile_sha256"]

    result = execute_trajectory(bus, payload["waypoints"], payload["bounds"],
                                max_speed=payload["max_speed"], authorize=still_authorized, **kwargs)
    result.update(execution_sha256=envelope["sha256"], profile_sha256=payload["profile_sha256"])
    return result


def execute_station_plan(bus, envelope, *, lease_valid, profile_path,
                         tracking_tolerance, final_tolerance, timeout_s,
                         feedback=None):
    """Adapter for a bus yielded by robot_resources.hardware().

    Fleet must provide real exclusivity beyond the helper's advisory flock.
    Requires already-enabled position-mode motors and accepted encoder ranges;
    never enables torque or changes limits. Lazy imports keep offline use pure.
    """
    from pathlib import Path
    from calibrate_workspace import validate_target
    try:
        from . import robot_resources as resources
    except ImportError:  # Installed flat station tools directory.
        import robot_resources as resources

    payload = envelope["payload"]
    names = tuple(payload["waypoints"][0]["joints"])
    validate_execution_budget(payload["plan"], tracking_tolerance, final_tolerance)
    gripper_range = payload["plan"].get("required_gripper_range_percent")
    if gripper_range is not None:
        if (not isinstance(gripper_range, (list, tuple)) or len(gripper_range) != 2
                or not all(_number(v) for v in gripper_range)
                or not 0 <= gripper_range[0] < gripper_range[1] <= 100):
            raise ValueError("Invalid collision gripper opening range")
    if not names or any(j not in resources.JOINTS[:-1] for j in names):
        raise ValueError("Station trajectory supports arm joints only; use contact controller for gripper")
    accepted = resources.limits(bus.calibration)
    for j in names:
        requested = payload["bounds"][j]
        if requested[0] < accepted[j][0] or requested[1] > accepted[j][1]:
            raise ValueError("Envelope expands accepted operating limits")
        if payload["max_speed"][j] > 12.:
            raise ValueError("Envelope expands existing arm speed limit")
        if tracking_tolerance[j] > 9. or final_tolerance[j] > 5.:
            raise ValueError("Envelope expands existing tracking tolerances")
    for waypoint in payload["waypoints"]:
        validate_target(waypoint["joints"], bus.calibration)
        resources.validate_operating_target(waypoint["joints"])

    def faults():
        if gripper_range is not None:
            opening = bus.sync_read("Present_Position", ["gripper"]).get("gripper")
            if not _number(opening) or not gripper_range[0] <= opening <= gripper_range[1]:
                raise _Abort("gripper_collision_envelope_changed")
        statuses = {}
        raw = bus.sync_read("Present_Position", list(names), normalize=False)
        observed = {register: bus.sync_read(register, list(names), normalize=False)
                    for register in ("Status", "Operating_Mode", "Torque_Enable")}
        for j in names:
            c = bus.calibration[j]
            status = observed["Status"].get(j)
            if type(status) is not int or status < 0:
                status = 1
            for register, expected in (("Operating_Mode", 0), ("Torque_Enable", 1)):
                if observed[register].get(j) != expected:
                    status = status or 1
            if not _number(raw.get(j)) or not c.range_min <= raw[j] <= c.range_max:
                status = status or 1
            statuses[j] = status
        return statuses

    def profile_digest():
        return hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()

    with bounded_transport(bus) as transport:
        result = execute_prepared(bus, envelope, current_profile_sha256=profile_digest,
                                  authorize=lease_valid, read_faults=faults,
                                  tracking_tolerance=tracking_tolerance,
                                  final_tolerance=final_tolerance, timeout_s=timeout_s,
                                  feedback=feedback)
    result["transport"] = transport
    return result


def validate_execution_budget(plan, tracking_tolerance, final_tolerance):
    """Enforce planner-reserved error envelope; absent budget is offline-only.

    Capsule model reserves tracking margin once against environment and twice
    for self pairs. Reach values must bound all modeled surfaces about each
    joint. Metadata provenance is a coordinator/model-review responsibility.
    """
    names = set(plan["waypoints"][0])
    budget = plan.get("execution_tolerance_deg")
    model = plan.get("collision_model", {})
    reaches = model.get("downstream_reach_m", {})
    margin = model.get("tracking_margin_m")
    if (not isinstance(budget, dict) or set(budget) != names
            or not isinstance(reaches, dict) or set(reaches) != names
            or model.get("tracking_margin_applied") is not True
            or not _number(margin) or margin <= 0):
        raise ValueError("Missing planner-reserved execution error budget")
    if any(not _number(budget[j]) or budget[j] <= 0 or
           not _number(reaches[j]) or reaches[j] <= 0 for j in names):
        raise ValueError("Invalid planner execution budget")
    required = sum(reaches[j]*math.radians(budget[j]) for j in names)
    if not math.isfinite(required) or required > margin:
        raise ValueError("Execution deviation exceeds reserved collision margin")
    for tolerances in (tracking_tolerance, final_tolerance):
        if set(tolerances) != names or any(not _number(tolerances[j]) or
                not 0 < tolerances[j] <= budget[j] for j in names):
            raise ValueError("Execution tolerance exceeds planner collision budget")
    return {"maximum_modeled_surface_deviation_m": required,
            "reserved_tracking_margin_m": margin}


def execute_trajectory(bus, waypoints, bounds, *, max_speed, tracking_tolerance,
                       final_tolerance, timeout_s, read_faults, authorize,
                       period_s=0.05, settle_s=0.5, feedback=None,
                       clock=time.monotonic, sleep=time.sleep):
    """Run a validated path with fresh feedback and a structured terminal result.

    authorize() must return True while the actual exclusive lease is held.
    read_faults() must return a complete mapping for commanded joints (0 healthy).
    feedback(sample) may return False to stop; exceptions also stop. All callbacks
    run synchronously and must themselves be bounded. No automatic error retry.
    On abort, attempt a hold at fresh valid encoder positions only while authorized.
    A failed hold is reported explicitly; this is not an emergency-stop mechanism.
    """
    began = clock()
    result = {"status": "invalid", "reason": None, "writes": 0,
              "object_held_verified": False, "hold": "not_needed"}
    active = False
    names = ()
    try:
        bounds = copy.deepcopy(bounds)
        max_speed = dict(max_speed)
        tracking_tolerance = dict(tracking_tolerance)
        final_tolerance = dict(final_tolerance)
        path = validate_trajectory(waypoints, bounds, max_speed)
        names = tuple(path[0]["joints"])
        for value in (timeout_s, period_s, settle_s):
            if not _number(value) or value <= 0:
                raise ValueError("Execution timing must be finite and positive")
        if timeout_s > 65 or timeout_s < path[-1]["time_s"] + settle_s or not .01 <= period_s <= .2:
            raise ValueError("Invalid deadline or sampling period")
        for tolerances in (tracking_tolerance, final_tolerance):
            if any(not _number(tolerances.get(j)) or tolerances[j] <= 0 for j in names):
                raise ValueError("Missing positive tracking/final tolerance")
        if any(final_tolerance[j] > tracking_tolerance[j] for j in names):
            raise ValueError("Final tolerance exceeds tracking tolerance")
        last_time = began

        def check():
            nonlocal last_time
            now = clock()
            if not _number(now) or now < last_time:
                raise _Abort("clock_invalid")
            last_time = now
            if now - began >= timeout_s:
                raise _Abort("deadline")
            if authorize() is not True:
                raise _Abort("authorization_lost")
            faults = read_faults()
            if not isinstance(faults, dict) or any(j not in faults or faults[j] != 0 for j in names):
                raise _Abort("motor_fault_or_missing_status")
            q = _positions(dict(bus.sync_read("Present_Position", list(names))), bounds, names)
            if clock() - began >= timeout_s:
                raise _Abort("deadline")
            return q

        measured = check()
        if any(abs(measured[j]-path[0]["joints"][j]) > final_tolerance[j] for j in names):
            raise _Abort("start_mismatch")
        active = True
        result["status"] = "stopped"
        # Send every validated waypoint. Slow feedback stretches each segment;
        # never catch up to wall-clock schedule by skipping geometric samples.
        last_goal = dict(path[0]["joints"])
        result["requested_path_duration_s"] = path[-1]["time_s"]
        result["timing_policy"] = "stretch_segments_without_skipping_samples"
        for a, b in zip(path, path[1:]):
            segment_start = clock()
            segment_duration = b["time_s"]-a["time_s"]
            while True:
                measured = check()
                elapsed = clock() - segment_start
                fraction = min(1., max(0., elapsed/segment_duration))
                goal = {j: a["joints"][j]+fraction*(b["joints"][j]-a["joints"][j]) for j in names}
                errors = {j: measured[j]-last_goal[j] for j in names}
                if any(abs(errors[j]) > tracking_tolerance[j] for j in names):
                    raise _Abort("tracking_error")
                # A prior-goal check alone allows the next step to push the
                # demanded deviation outside the collision inflation budget.
                if any(abs(measured[j]-goal[j]) > tracking_tolerance[j] for j in names):
                    raise _Abort("next_goal_tracking_error")
                sample = {"elapsed_s": clock()-began, "measured": measured,
                          "commanded": dict(last_goal), "tracking_error": errors}
                result["last_sample"] = sample
                if feedback is not None and feedback(sample) is False:
                    raise _Abort("feedback_stop")
                if clock()-began >= timeout_s:
                    raise _Abort("deadline")
                if authorize() is not True:
                    raise _Abort("authorization_lost")
                bus.sync_write("Goal_Position", goal)
                result["writes"] += 1
                last_goal = goal
                if fraction == 1.:
                    break
                sleep(min(period_s, max(0., segment_duration-elapsed)))
        settling = clock()
        while True:
            measured = check()
            errors = {j: measured[j]-last_goal[j] for j in names}
            result["final_error"] = errors
            sample = {"elapsed_s": clock()-began, "measured": measured,
                      "commanded": dict(last_goal), "tracking_error": errors}
            result["last_sample"] = sample
            if feedback is not None and feedback(sample) is False:
                raise _Abort("feedback_stop")
            if clock()-began >= timeout_s:
                raise _Abort("deadline")
            if authorize() is not True:
                raise _Abort("authorization_lost")
            if all(abs(errors[j]) <= final_tolerance[j] for j in names):
                result.update(status="completed", reason="within_joint_tolerance")
                break
            if any(abs(errors[j]) > tracking_tolerance[j] for j in names):
                raise _Abort("tracking_error")
            if clock()-settling >= settle_s:
                raise _Abort("final_tolerance_not_reached")
            sleep(period_s)
    except (Exception, KeyboardInterrupt) as exc:
        result["reason"] = str(exc) or type(exc).__name__
        result["error_type"] = type(exc).__name__
        if isinstance(exc, _Abort):
            result["status"] = "stopped"
        if active:
            try:
                if authorize() is not True:
                    result["hold"] = "skipped_authorization_lost"
                else:
                    q = _positions(dict(bus.sync_read("Present_Position", list(names))), bounds, names)
                    if authorize() is not True:
                        raise RuntimeError("Authorization lost before hold write")
                    bus.sync_write("Goal_Position", q)
                    result["hold"] = "commanded_fresh_position"
            except Exception as hold_error:
                result["hold"] = "failed"
                result["hold_error"] = str(hold_error)
    result["elapsed_s"] = clock()-began
    return result


def classify_contact(samples, *, closed_position, closed_tolerance,
                     contact_effort, stable_samples=3, stationary_tolerance=None):
    """Classify caller-supplied gripper feedback; never infer a retained object.

    Samples: position and nonnegative effort proxy in caller-consistent units,
    fault integer and strictly increasing time_s. Thresholds are station-owned.
    No force units, default force thresholds, detector or motion policy is added.
    """
    output = {"outcome": "unknown", "object_held_verified": False,
              "effort_is_force": False}
    if (not _number(closed_position) or not _number(closed_tolerance) or closed_tolerance <= 0
            or not _number(contact_effort) or contact_effort <= 0
            or type(stable_samples) is not int or stable_samples < 2):
        raise ValueError("Invalid contact classification thresholds")
    stationary_tolerance = closed_tolerance if stationary_tolerance is None else stationary_tolerance
    if not _number(stationary_tolerance) or stationary_tolerance <= 0:
        raise ValueError("Invalid stationary tolerance")
    previous = None
    for sample in samples:
        if any(not _number(sample.get(k)) for k in ("position", "effort", "time_s")) or sample["effort"] < 0:
            return {**output, "reason": "invalid_feedback"}
        if "fault" not in sample or type(sample["fault"]) is not int:
            return {**output, "reason": "missing_fault_status"}
        if sample["fault"]:
            return {**output, "outcome": "fault"}
        if previous is not None and sample["time_s"] <= previous:
            return {**output, "reason": "nonmonotonic_feedback"}
        previous = sample["time_s"]
    if len(samples) < stable_samples:
        return {**output, "reason": "insufficient_samples"}
    tail = samples[-stable_samples:]
    if all(abs(s["position"]-closed_position) <= closed_tolerance and s["effort"] < contact_effort for s in tail):
        output["outcome"] = "empty_close"
    elif (all(abs(s["position"]-closed_position) > closed_tolerance and s["effort"] >= contact_effort for s in tail)
          and max(s["position"] for s in tail)-min(s["position"] for s in tail) <= stationary_tolerance):
        output["outcome"] = "contact_candidate"
    else:
        output["reason"] = "ambiguous_or_unstable_feedback"
    return output


def controller_contact_outcome(controller, error=None):
    """Read existing AdaptiveGrip history without issuing any bus commands.

    Uses that controller's accepted raw closed endpoint and established contact
    rule (load >= min(60, effort*.4), stationary range <=3 encoder ticks).
    Empty endpoint tolerance 12 ticks matches its acquire empty-close check.
    Error type is reported, but exception prose is never treated as evidence.
    """
    unknown = {"outcome": "unknown", "object_held_verified": False,
               "effort_is_force": False}
    try:
        effort = controller.effort
        if not _number(effort) or effort <= 0:
            raise ValueError("Invalid controller effort threshold")
        samples = [{"time_s": s["time"], "position": s["position"],
                    "effort": s["load"], "fault": s["fault"]}
                   for s in controller.history]
        result = classify_contact(samples, closed_position=controller.closed,
            closed_tolerance=12., contact_effort=min(60., effort*.4),
            stationary_tolerance=3.)
        result["samples_available"] = len(samples)
        # Existing controller reports stable hold only after its own timed
        # checks. This merely preserves the claim; it does not verify an object.
        result["controller_contact_detected"] = getattr(controller, "contact", False) is True
        result["controller_stable_seconds_required"] = controller.stable_seconds
    except (AttributeError, KeyError, TypeError, ValueError):
        result = {**unknown, "reason": "controller_feedback_unavailable_or_invalid"}
    if error is not None:
        result["control_error_type"] = type(error).__name__
        # Preserve bounded diagnostic evidence, never infer contact or faults
        # from exception prose. A controller may fail before history is saved.
        try:
            message = str(error)
        except Exception:
            message = "Controller error message unavailable"
        result["control_error_message"] = message[:512]
        result["control_error_message_truncated"] = len(message) > 512
        result["control_completed"] = False
    else:
        result["control_completed"] = True
    return result

"""Task-independent state, images, kinematics and explicit joint motion.

Run under arm-shell. Importing this module never connects to hardware.
"""
import argparse
import base64
import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time
from functools import lru_cache

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_roll", "gripper")


def operating_limits():
    """Optional measured operating envelope; never expands encoder limits."""
    path = Path(__file__).with_name("robot_geometry.json")
    data = json.loads(path.read_text()) if path.exists() else {}
    result = data.get("operating_limits_deg", {})
    for joint, bound in result.items():
        if (joint not in JOINTS[:-1] or not isinstance(bound, list) or len(bound) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bound)
                or bound[0] >= bound[1]):
            raise ValueError("Invalid profile operating limit: " + joint)
    return result


def validate_operating_target(target):
    for joint, bound in operating_limits().items():
        if joint in target and not bound[0] <= target[joint] <= bound[1]:
            raise ValueError(f"{joint}: target outside measured operating range {bound}")


def limits(calibration):
    result = {j: [0., 100.] if j == "gripper" else
            [-(c.range_max-c.range_min)*180/4095, (c.range_max-c.range_min)*180/4095]
            for j, c in calibration.items()}
    for joint, bound in operating_limits().items():
        if joint not in result:
            raise ValueError("Operating limit names an absent motor: " + joint)
        result[joint] = [max(result[joint][0], bound[0]), min(result[joint][1], bound[1])]
        if result[joint][0] >= result[joint][1]:
            raise ValueError("Operating range does not overlap encoder limits: " + joint)
    return result


@lru_cache(maxsize=1)
def chain():
    import xml.etree.ElementTree as ET
    import numpy as np
    from scipy.spatial.transform import Rotation
    from fk import URDF
    tree = ET.fromstring(URDF)
    by_child = {j.find("child").get("link"): j for j in tree.findall("joint")}
    joints, link = [], "gripper_frame_link"
    while link != "base_link":
        joint = by_child[link]
        joints.append(joint)
        link = joint.find("parent").get("link")
    result = []
    for joint in reversed(joints):
        origin = joint.find("origin")
        F = np.eye(4)
        F[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        F[:3, :3] = Rotation.from_euler(
            "xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]).as_matrix()
        axis = None
        if joint.get("type") != "fixed":
            axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
            axis = axis/np.linalg.norm(axis)
        result.append((joint.get("name"), F, axis))
    return result


def pose(joints):
    """URDF transform, without a device connection or default missing joints."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    for name, F, axis in chain():
        T = T @ F
        if axis is not None:
            angle = float(joints[name])
            if not math.isfinite(angle):
                raise ValueError("Nonfinite joint angle")
            M = np.eye(4)
            M[:3, :3] = Rotation.from_rotvec(
                axis*math.radians(angle)).as_matrix()
            T = T @ M
    return T


@lru_cache(maxsize=1)
def configured_tool_translation():
    path = Path(__file__).with_name("robot_geometry.json")
    data = json.loads(path.read_text()) if path.exists() else {}
    offset = data.get("tool_reference", {}).get("translation_from_urdf_m", [0., 0., 0.])
    if len(offset) != 3 or not all(math.isfinite(v) for v in offset) or sum(v*v for v in offset) > .05**2:
        raise ValueError("Invalid calibrated tool translation")
    return tuple(offset)


def tool_pose(joints):
    T = pose(joints)
    T[:3, 3] += T[:3, :3] @ configured_tool_translation()
    return T


def geometry_capabilities(profile):
    """Describe supplied resources without inferring missing calibration."""
    tool = profile.get("tool_reference", {})
    return {
        "camera_intrinsics": all(k in profile for k in ("camera_matrix", "dist_coeffs", "image_size")),
        "pixel_projection": ("camera_extrinsics" if profile.get("projection") else
                             "measured_planes_only" if profile.get("planes") else "unavailable"),
        "tool_point": ("profile_offset" if "translation_from_urdf_m" in tool else "nominal_urdf"),
        "aperture_axis": tool.get("aperture_axis_tool") is not None,
        "surface_reference": bool(profile.get("surface_reference")),
        "operating_limits": bool(profile.get("operating_limits_deg")),
    }


def planning_capabilities():
    root = Path(__file__).resolve().parent
    path = root/"robot_collision.json"
    if not path.is_file():
        return {"configured": False, "reason": "station_collision_configuration_missing"}
    try:
        config = json.loads(path.read_text())
        review = config.get("station_review", {})
        required = ("geometry_matches_station", "table_reference_accepted",
                    "jaw_range_contains_motion", "excluded_contacts_accepted")
        missing = [name for name in required if review.get(name) is not True]
        return {"configured": not missing, "missing_review": missing,
                "requires_live_writer_lease": True,
                "coverage": "installed geometric model; inspect returned plan assumptions"}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {"configured": False, "reason": "invalid_collision_configuration", "detail": str(exc)}


def state(bus):
    from calibrate_workspace import fault_bits, read_register
    q = dict(bus.sync_read("Present_Position"))
    T = tool_pose(q)
    geometry = Path(__file__).with_name("robot_geometry.json")
    profile = json.loads(geometry.read_text()) if geometry.exists() else {}
    aperture = profile.get("tool_reference", {}).get("aperture_axis_tool")
    if aperture is not None and (len(aperture) != 3 or
            not all(math.isfinite(v) for v in aperture) or
            abs(sum(v*v for v in aperture)-1) > .001):
        raise ValueError("Invalid measured aperture axis")
    return {
        "time_unix": time.time(), "joints": q,
        "limits": limits(bus.calibration),
        "faults": {j: fault_bits(bus, j) for j in JOINTS},
        "goals": dict(bus.sync_read("Goal_Position")),
        "torque_limits_raw": {j: read_register(bus, "Torque_Limit", j)[0] for j in JOINTS},
        "tip_xyz_m": T[:3, 3].tolist(),
        "tip_axis_base": T[:3, 2].tolist(),
        "tip_transform": T.tolist(),
        "aperture_axis_tool": aperture,
        "aperture_axis_base": (T[:3, :3] @ aperture).tolist() if aperture is not None else None,
        "tool_translation_from_urdf_m": configured_tool_translation(),
        "geometry_status": ("URDF with supplied profile tool-point offset; not external metrology"
                            if "translation_from_urdf_m" in profile.get("tool_reference", {}) else
                            "Nominal URDF tool point; no measured fingertip offset supplied; not external metrology"),
        "geometry_capabilities": geometry_capabilities(profile),
        "planning_capabilities": planning_capabilities(),
    }


@contextlib.contextmanager
def hardware():
    import fcntl
    from recording import serial_port
    from calibrate_workspace import connected_bus
    with contextlib.ExitStack() as stack:
        # The recorder runtime is shared with fresh-agent mount namespaces.
        # Read-only descriptors can take flock locks and remain accessible when
        # another authorized account created the lock file.
        runtime = Path(os.environ.get("ARMFARM_RECORDING_SOCKET",
                                      "/run/armfarm-recording-armfarm/control.sock")).parent
        fd = os.open(runtime/"robot-resources.lock", os.O_RDONLY | os.O_CREAT, 0o644)
        f = stack.enter_context(os.fdopen(fd, "r"))
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Historical state is deliberately absent in fresh-agent isolation.
        legacy = () if os.environ.get("ARMFARM_ISOLATED") == "1" else (
            "ssh-control.lock", "act-benchmark.lock", "pen-operation.lock")
        for name in legacy:
            fd = os.open(Path("/var/lib/armfarm/stations/armfarm/state")/name,
                         os.O_RDONLY | os.O_CREAT, 0o644)
            f = stack.enter_context(os.fdopen(fd, "r"))
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with connected_bus(serial_port()) as bus:
            yield bus


def settle_position(bus, target):
    """Bounded encoder feedback for explicit arm targets; excludes contact gripper."""
    from calibrate_workspace import validate_target, fault_bits
    wanted = {j: v for j, v in target.items() if j != "gripper"}
    if not wanted:
        return {"reached": None, "skipped": True,
                "reason": "Precision settling applies only to arm joints; no arm target supplied"}
    command = dict(wanted)
    for _ in range(20):
        measured = bus.sync_read("Present_Position", list(wanted))
        if any(fault_bits(bus, j) for j in wanted):
            raise RuntimeError("Motor fault during precision settling")
        errors = {j: wanted[j]-measured[j] for j in wanted}
        if all(abs(e) <= .4 for e in errors.values()):
            return {"reached": True, "error_deg": errors}
        proposed = {j: command[j]+max(-.5, min(.5, errors[j])) for j in wanted}
        if any(abs(proposed[j]-wanted[j]) > 4 for j in wanted):
            return {"reached": False, "reason": "4-degree compensation bound",
                    "error_deg": errors, "holding_goal": command}
        try:
            validate_target(proposed, bus.calibration)
            validate_operating_target(proposed)
        except ValueError:
            return {"reached": False, "reason": "live joint limit",
                    "error_deg": errors, "holding_goal": command}
        command = proposed
        bus.sync_write("Goal_Position", command)
        time.sleep(.35)
    return {"reached": False, "reason": "settling time limit",
            "error_deg": errors, "holding_goal": command}


def move(bus, target, seconds, precision=False):
    """Finite, explicit joint interpolation. No task planning or fault recovery."""
    from calibrate_workspace import validate_target, fault_bits, enable_at_current_position
    validate_target(target, bus.calibration)
    validate_operating_target(target)
    if not math.isfinite(seconds) or not 0.1 <= seconds <= 60:
        raise ValueError("seconds must be finite, 0.1..60")
    start = dict(bus.sync_read("Present_Position"))
    raw = bus.sync_read("Present_Position", list(target), normalize=False)
    for j in target:
        c = bus.calibration[j]
        if not c.range_min <= raw[j] <= c.range_max or fault_bits(bus, j):
            raise RuntimeError(f"{j}: start outside calibration or motor fault")
    # Smoothstep's peak speed is 1.5 times its mean. These bounds concern
    # joint speed only; they do not certify Cartesian collision clearance.
    duration = max(seconds, max(1.5*abs(target[j]-start[j]) /
                   (40. if j == "gripper" else 12.) for j in target))
    enable_at_current_position(bus, list(target))
    began = time.monotonic()
    precision_result = None
    try:
        while True:
            f = min((time.monotonic()-began)/duration, 1.)
            alpha = f*f*(3-2*f)
            goal = {j: start[j]+alpha*(target[j]-start[j]) for j in target}
            now = bus.sync_read("Present_Position", list(target))
            for j in target:
                if fault_bits(bus, j):
                    raise RuntimeError(f"{j}: motor fault")
                if j != "gripper" and abs(now[j]-goal[j]) > 9:
                    raise RuntimeError(f"{j}: tracking gap over 9 degrees")
            bus.sync_write("Goal_Position", goal)
            if f == 1:
                break
            time.sleep(.05)
        time.sleep(.6)
        now = bus.sync_read("Present_Position", list(target))
        if any(abs(now[j]-target[j]) > 5 for j in target if j != "gripper"):
            raise RuntimeError("Target not reached within 5 degrees")
        if precision:
            precision_result = settle_position(bus, target)
    except BaseException:
        bus.sync_write("Goal_Position", bus.sync_read("Present_Position", list(target)))
        raise
    final = state(bus)
    intended = tool_pose({**start, **target})
    tip_error = math.sqrt(sum((final["tip_xyz_m"][i]-intended[i, 3])**2 for i in range(3)))
    return {"elapsed_s": time.monotonic()-began, "requested": target, "state": final,
            "precision": precision_result,
            "tracking_error_deg_or_percent": {j: final["joints"][j]-target[j] for j in target},
            "nominal_tip_error_m": tip_error}


def grip(bus):
    """Explicit effort-regulated closure using the station's existing controller."""
    from calibrate_workspace import AdaptiveGrip, fault_bits, read_register
    from trajectory_execution import controller_contact_outcome
    if any(fault_bits(bus, j) for j in JOINTS):
        raise RuntimeError("Clear existing motor faults before grip")

    class NoResetGrip(AdaptiveGrip):
        def get(self, name):
            value, faults = read_register(self.bus, name, "gripper")
            if faults or (name == "Status" and value):
                raise RuntimeError("Gripper fault: closure stopped; no automatic reset")
            if name == "Present_Temperature" and value >= 60:
                raise RuntimeError("Gripper temperature margin exhausted")
            return value

        def recover(self):
            self.stop()
            raise RuntimeError("Gripper fault: closure stopped; no automatic reset")

    controller = NoResetGrip(bus)
    began = time.monotonic()
    error = None
    result = None
    try:
        result = controller.acquire()
        final = state(bus)
        if any(final["faults"].values()):
            raise RuntimeError("Motor fault after grip")
    except Exception as exc:
        controller.stop()
        error = exc
        result = None
        final = state(bus)
    except BaseException:
        controller.stop()
        raise
    feedback = controller_contact_outcome(controller, error)
    if any(final["faults"].values()):
        feedback["outcome"] = "fault"
    return {"elapsed_s": time.monotonic()-began, "contact_hold": result,
            "feedback": feedback,
            "state": final, "output_ceiling_raw": controller.limit,
            "previous_output_ceiling_raw": controller.original_limit,
            "object_held_verified": False,
            "note": ("Closure stopped after a controller error; no retained object is verified."
                     if error else "Effort is a motor proxy, not measured jaw force. Hold target and station-configured volatile output ceiling remain active; images must verify retention.")}


def solve(xyz, axis, seed, bounds, roll=None, offset=None):
    """Position plus optional approach-axis IK; reads no devices, moves nothing."""
    import numpy as np
    from scipy.optimize import least_squares
    xyz = np.asarray(xyz, float)
    offset = np.zeros(3) if offset is None else np.asarray(offset, float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .2:
        raise ValueError("offset must contain three finite tool-frame metres, norm at most 0.2")
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("xyz must contain three finite metres")
    wanted = None
    if axis is not None:
        wanted = np.asarray(axis, float)
        if wanted.shape != (3,) or not np.isfinite(wanted).all() or np.linalg.norm(wanted) < 1e-8:
            raise ValueError("axis must be a finite nonzero vector")
        wanted = wanted/np.linalg.norm(wanted)
    # Direction alone leaves roll underconstrained. Preserve measured roll
    # unless a caller explicitly requests another value, rather than allowing
    # an arbitrary null-space rotation to appear in a position solution.
    roll = seed["wrist_roll"] if roll is None else float(roll)
    names = list(JOINTS[:4])
    seed = {**seed, "wrist_roll": roll}
    if not bounds["wrist_roll"][0] <= roll <= bounds["wrist_roll"][1]:
        raise ValueError("roll outside live limits")
    lo, hi = np.array([bounds[j][0] for j in names]), np.array([bounds[j][1] for j in names])
    def residual(q):
        T = tool_pose({**seed, **dict(zip(names, q))})
        r = (T[:3, 3]+T[:3, :3]@offset-xyz)*100
        return r if wanted is None else np.r_[r, (T[:3, 2]-wanted)*2]
    initial = np.clip([seed[j] for j in names], lo+1e-6, hi-1e-6)
    guesses = [initial, (lo+hi)/2]
    candidates = [least_squares(residual, q, bounds=(lo, hi), max_nfev=150) for q in guesses]
    def rank(result):
        T = tool_pose({**seed, **dict(zip(names, result.x))})
        error = float(np.linalg.norm(T[:3,3]+T[:3,:3]@offset-xyz))
        angle = 0 if wanted is None else math.degrees(math.acos(float(np.clip(T[:3,2]@wanted,-1,1))))
        feasible = error < .003 and angle < 5
        return (0, float(np.sum(((result.x-initial)/(hi-lo))**2))) if feasible else (1, result.cost)
    best = min(candidates, key=rank)
    q = {**seed, **dict(zip(names, map(float, best.x)))}
    q.pop("gripper", None)
    T = tool_pose(q)
    point = T[:3, 3]+T[:3, :3]@offset
    error = float(np.linalg.norm(point-xyz))
    angle = None if wanted is None else math.degrees(math.acos(float(np.clip(T[:3, 2]@wanted, -1, 1))))
    geometry_path = Path(__file__).with_name("robot_geometry.json")
    profile = json.loads(geometry_path.read_text()) if geometry_path.exists() else {}
    aperture = profile.get("tool_reference", {}).get("aperture_axis_tool")
    if aperture is not None:
        aperture = np.asarray(aperture, dtype=float)
        if (aperture.shape != (3,) or not np.isfinite(aperture).all()
                or abs(float(aperture @ aperture)-1.) > .001):
            raise ValueError("Invalid measured aperture axis")
    return {"joints": q, "position_error_m": error, "axis_error_deg": angle,
            "tip_xyz_m": T[:3, 3].tolist(),
            "tip_transform": T.tolist(), "tip_axis_base": T[:3, 2].tolist(),
            "aperture_axis_base": (T[:3, :3] @ aperture).tolist() if aperture is not None else None,
            "reference_offset_base_m": (T[:3, :3] @ offset).tolist(),
            "target_point_xyz_m": point.tolist(), "tool_offset_m": offset.tolist(),
            "usable_kinematically": error < .003 and (angle is None or angle < 5),
            "collision_checked": False, "motion_commanded": False}


def orientation_options(xyz, seed, bounds, roll=None, offset=None):
    """Enumerate downward-to-horizontal radial directions, with no task choice."""
    azimuth = math.atan2(xyz[1], xyz[0]-.0388353)
    solutions = []
    for tilt in range(0, 91, 10):
        angle = math.radians(tilt)
        axis = [math.sin(angle)*math.cos(azimuth),
                math.sin(angle)*math.sin(azimuth), -math.cos(angle)]
        result = solve(xyz, axis, seed, bounds, roll, offset)
        if result["usable_kinematically"]:
            solutions.append({"tilt_from_down_deg": tilt, "axis": axis, **result})
    return {"solutions": solutions, "motion_commanded": False,
            "collision_checked": False}


def cartesian_plan(xyz, seed, bounds, axis=None, tilt=None, roll=None, offset=None):
    """Resolve an explicit Cartesian endpoint and inspect its joint interpolation."""
    import numpy as np
    if tilt is not None:
        if not math.isfinite(tilt) or not -90 <= tilt <= 90:
            raise ValueError("tilt must be -90..90 degrees from down")
        azimuth = math.atan2(xyz[1], xyz[0]-.0388353)
        a = math.radians(tilt)
        axis = [math.sin(a)*math.cos(azimuth), math.sin(a)*math.sin(azimuth), -math.cos(a)]
    if axis is None:
        axis = tool_pose(seed)[:3,2]
    plan = solve(xyz, axis, seed, bounds, roll, offset)
    if not plan["usable_kinematically"]:
        raise ValueError("Endpoint is unreachable at this orientation: "+json.dumps(plan))
    target = plan["joints"]
    path = [tool_pose({j: seed[j]+f*(target[j]-seed[j]) for j in target})[:3,3]
            for f in np.linspace(0,1,41)]
    dip = min(path[0][2],path[-1][2])-min(p[2] for p in path)
    if dip > .005:
        raise ValueError("Joint interpolation dips below endpoint heights; explicit intermediate waypoints are required")
    plan["minimum_interpolated_tip_z_m"] = float(min(p[2] for p in path))
    plan["max_joint_delta_deg"] = max(abs(target[j]-seed[j]) for j in target)
    return plan


def project(pixel, height):
    """Project a raw image pixel onto a specified horizontal plane."""
    data = json.loads((Path(__file__).parent/"robot_geometry.json").read_text())
    return project_data(data, pixel, height)


def project_data(data, pixel, height):
    import cv2
    import numpy as np
    if not math.isfinite(height) or len(pixel) != 2 or not all(math.isfinite(v) for v in pixel):
        raise ValueError("Pixel and height must be finite")
    if not data.get("projection") and not data.get("planes"):
        raise ValueError("Pixel projection unavailable: no camera-to-robot transform or measured robot-frame planes supplied. Camera intrinsics alone cannot map pixels to robot coordinates.")
    if not 0 <= pixel[0] < data["image_size"][0] or not 0 <= pixel[1] < data["image_size"][1]:
        raise ValueError("Pixel is outside the calibrated image dimensions")
    if "projection" in data:
        calibration = data["projection"]
        R = np.array(calibration["R_base_to_camera"])
        t = np.array(calibration["t_base_to_camera_m"])
        ray = np.r_[cv2.undistortPoints(np.array([[pixel]], float),
                     np.array(data["camera_matrix"]), np.array(data["dist_coeffs"]))[0, 0], 1.]
        origin, direction = -R.T@t, R.T@ray
        if abs(direction[2]) < 1e-6:
            raise ValueError("Camera ray is parallel to this plane")
        distance = (height-origin[2])/direction[2]
        if distance <= 0:
            raise ValueError("Plane intersection is behind the camera")
        point = origin+distance*direction
        return {"xyz_m": point.tolist(), "height_is_caller_supplied": True,
                "validation": data.get("validation"),
                "coordinate_quality": "relative to nominal live-motor FK, not external metrology",
                "motion_commanded": False}
    plane = min(data["planes"], key=lambda p: abs(height-p["z_m"]))
    if abs(height-plane["z_m"]) > .001:
        raise ValueError(f"Only measured plane z={plane['z_m']}m is available; no other-height transform")
    p = cv2.undistortPoints(np.array([[pixel]], float),
                           np.array(data["camera_matrix"]), np.array(data["dist_coeffs"]),
                           P=np.array(data["camera_matrix"]))[0, 0]
    q = np.array(plane["H"]) @ np.r_[p, 1.]
    return {"xyz_m": [float(q[0]/q[2]), float(q[1]/q[2]), height],
            "fit_rms_m": plane["rms_m"], "independently_validated": False,
            "image_size": data["image_size"], "motion_commanded": False}


def measure_height(data, base_pixel, top_pixel, base_height):
    """Height from a caller-identified vertical edge, not an object detector."""
    import cv2
    import numpy as np
    if "projection" not in data:
        raise ValueError("Height measurement requires calibrated camera extrinsics")
    base = np.array(project_data(data, base_pixel, base_height)["xyz_m"])
    # Also validate the upper pixel through the standard projection interface.
    project_data(data, top_pixel, base_height)
    calibration = data["projection"]
    R = np.array(calibration["R_base_to_camera"])
    origin = -R.T@np.array(calibration["t_base_to_camera_m"])
    ray = np.r_[cv2.undistortPoints(np.array([[top_pixel]], float),
                 np.array(data["camera_matrix"]), np.array(data["dist_coeffs"]))[0, 0], 1.]
    direction = R.T@ray
    denominator = float(direction[:2]@direction[:2])
    if denominator < .0025:
        raise ValueError("Vertical edge too close to the camera's vertical line for reliable height")
    distance = float(direction[:2]@(base[:2]-origin[:2])/denominator)
    top = origin+distance*direction
    residual = float(np.linalg.norm(top[:2]-base[:2]))
    height = float(top[2]-base_height)
    # A world-vertical edge generally is not vertical in image coordinates.
    # Supply a geometric ruler, independent of any object or chosen task.
    heights = [0., .01, .02, .05, .1]
    points = np.array([base+np.array([0.,0.,h]) for h in heights])
    pixels = cv2.projectPoints(points, cv2.Rodrigues(R)[0],
                               np.array(calibration["t_base_to_camera_m"]),
                               np.array(data["camera_matrix"]), np.array(data["dist_coeffs"]))[0].reshape(-1,2)
    reason = ("top_ray_behind_camera" if distance <= 0 else
              "inconsistent_vertical_edge" if residual >= .005 else
              "height_outside_supported_range" if not 0 <= height <= .3 else None)
    usable = reason is None
    # A rejected fit is not a measured coordinate. Keep its residual and
    # explicit inputs for diagnosis, without presenting an invalid height as
    # an actionable result alongside a flag a caller can accidentally overlook.
    return {"height_above_base_m": height if usable else None,
            "top_xyz_m": top.tolist() if usable else None,
            "base_xyz_m": base.tolist(), "vertical_alignment_error_m": residual,
            "usable_measurement": usable, "reason": reason,
            "inputs": {"base_pixel_uv": list(map(float, base_pixel)),
                       "top_pixel_uv": list(map(float, top_pixel)),
                       "base_height_m": base_height},
            "world_vertical_ruler": [{"height_above_base_m": h, "pixel_uv": uv.tolist()}
                                     for h,uv in zip(heights,pixels)],
            "ruler_definition": "Projection of constant base X,Y with increasing world Z. Image-up is not world-up; a long image edge may lie in a horizontal plane.",
            "assumptions": "Caller pixels are bottom and top of the same base-Z-parallel edge; lower endpoint lies on caller's height plane.",
            "motion_commanded": False}


def camera(role):
    from camd_client import read_jpeg
    image, meta = read_jpeg(role)
    return {"role": role, "metadata": meta, "jpeg_base64": base64.b64encode(image).decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    sub.add_parser("status")
    sub.add_parser("snapshot")
    for operation, help_text in (("plan-to", "Plan an entire approach using installed collision geometry; no motion"),
                                  ("approach-to", "Plan and execute a caller-selected approach with fresh images")):
        planned = sub.add_parser(operation, help=help_text)
        planned.add_argument("xyz", type=float, nargs=3)
        planned.add_argument("--approach", type=float, nargs=3, required=True)
        planned.add_argument("--clearance", type=float, required=True, help="Approach length in metres, not a collision-clearance claim")
        planned.add_argument("--tilts", type=float, nargs="+")
        planned.add_argument("--roll", type=float)
        reference = planned.add_mutually_exclusive_group()
        reference.add_argument("--offset", type=float, nargs=3)
        reference.add_argument("--contact-span", type=float, help="Caller-supplied parallel contact separation in metres")
    execution = sub.add_parser("execute-plan", help="Execute a server-stored one-use plan bound to live resources and lease")
    execution.add_argument("query", help="JSON plan result or execution handle")
    sub.add_parser("geometry-status", help="Report measured coordinate and tool capabilities without motion")
    midpoint = sub.add_parser("aperture-midpoint", help="Measured opening-dependent midpoint offset; no motion")
    midpoint.add_argument("opening", type=float)
    align = sub.add_parser("align-features", help="Bounded correction from caller-matched image features and Jacobian; no motion")
    align.add_argument("query", help="JSON object of observed_uv, target_uv, jacobian_px_per_m, max_step_m, axis_names, frame")
    jacobian = sub.add_parser("camera-jacobian", help="Local image Jacobian for caller-selected feature, height and base axes; no motion")
    jacobian.add_argument("query", help="JSON object of pixel_uv, plane_z_m, axis_names; optional pixel_sigma_px, height_sigma_m, max_condition")
    cam = sub.add_parser("camera")
    cam.add_argument("role", choices=["top", "wrist"])
    mov = sub.add_parser("move")
    mov.add_argument("targets", nargs="+", help="joint=value (degrees; gripper percent)")
    mov.add_argument("--seconds", type=float, default=2)
    mov.add_argument("--precision", action="store_true",
                     help="Bounded encoder settling to 0.4 degrees; arm joints only")
    cart = sub.add_parser("move-to", help="Move to an explicit Cartesian tool-point target; returns fresh images")
    cart.add_argument("xyz", type=float, nargs=3)
    orientation = cart.add_mutually_exclusive_group()
    orientation.add_argument("--axis", type=float, nargs=3)
    orientation.add_argument("--tilt", type=float, help="Radial direction in degrees from down")
    cart.add_argument("--roll", type=float)
    reference = cart.add_mutually_exclusive_group()
    reference.add_argument("--offset", type=float, nargs=3)
    reference.add_argument("--contact-span", type=float)
    cart.add_argument("--seconds", type=float, default=3)
    shift = sub.add_parser("shift", help="Translate the current tool point while preserving its direction and roll")
    shift.add_argument("delta", type=float, nargs=3)
    shift.add_argument("--frame", choices=["base","tool"], default="base")
    shift.add_argument("--seconds", type=float, default=3)
    ik = sub.add_parser("ik")
    ik.add_argument("xyz", type=float, nargs=3)
    ik.add_argument("--axis", type=float, nargs=3)
    ik.add_argument("--roll", type=float)
    reference = ik.add_mutually_exclusive_group()
    reference.add_argument("--offset", type=float, nargs=3, help="Target point's offset from the tool origin, in tool-frame metres")
    reference.add_argument("--contact-span", type=float)
    options = sub.add_parser("ik-options", help="Feasible radial orientations, 0..90 degrees from down, every 10 degrees; read only")
    options.add_argument("xyz", type=float, nargs=3)
    options.add_argument("--roll", type=float)
    reference = options.add_mutually_exclusive_group()
    reference.add_argument("--offset", type=float, nargs=3)
    reference.add_argument("--contact-span", type=float)
    many = sub.add_parser("ik-many")
    many.add_argument("queries", help='JSON list of {"xyz":[x,y,z],"axis":[ax,ay,az],"roll":degrees}; axis/roll optional')
    pix = sub.add_parser("project")
    pix.add_argument("uv", type=float, nargs=2)
    pix.add_argument("--height", type=float, required=True)
    height = sub.add_parser("measure-height", help="Estimate height from two pixels on a caller-identified vertical edge")
    height.add_argument("pixels", type=float, nargs=4, help="BASE_U BASE_V TOP_U TOP_V")
    height.add_argument("--base-height", type=float, required=True)
    sub.add_parser("grip", help="Effort-regulated closure and stable contact hold; no object localization, lift or fault reset")
    args = parser.parse_args()
    if getattr(args, "contact_span", None) is not None:
        from robot_geometry import parallel_contact_offset
        profile = json.loads(Path(__file__).with_name("robot_geometry.json").read_text())
        args.offset = parallel_contact_offset(profile, args.contact_span)["offset_tool_m"]
    with contextlib.redirect_stdout(sys.stderr):
        if args.op == "camera":
            result = camera(args.role)
        elif args.op == "project":
            result = project(args.uv, args.height)
        elif args.op == "measure-height":
            data = json.loads((Path(__file__).parent/"robot_geometry.json").read_text())
            result = measure_height(data, args.pixels[:2], args.pixels[2:], args.base_height)
        elif args.op in ("geometry-status", "aperture-midpoint", "align-features", "camera-jacobian"):
            import robot_geometry
            data = json.loads((Path(__file__).parent/"robot_geometry.json").read_text())
            if args.op == "geometry-status":
                result = robot_geometry.capability_report(data)
                result["planning"] = planning_capabilities()
            elif args.op == "aperture-midpoint":
                result = robot_geometry.aperture_midpoint(data, args.opening)
            else:
                query = json.loads(args.query)
                if not isinstance(query, dict):
                    raise ValueError("Feature query must be an object")
                if args.op == "camera-jacobian":
                    from robot_camera import camera_jacobian
                    result = camera_jacobian(data, **query)
                else:
                    result = robot_geometry.feature_alignment(**query)
        else:
            with hardware() as bus:
                if args.op in ("plan-to", "approach-to", "execute-plan"):
                    import motion_commands
                    if args.op in ("plan-to", "approach-to"):
                        result = motion_commands.plan_command(sys.modules[__name__], bus, args)
                        if args.op == "approach-to" and result.get("execution_handle"):
                            result = motion_commands.execute_command(sys.modules[__name__], bus, result)
                    else:
                        result = motion_commands.execute_command(sys.modules[__name__], bus, json.loads(args.query))
                elif args.op == "status":
                    result = state(bus)
                elif args.op == "snapshot":
                    result = {"state": state(bus), "images": [camera(r) for r in ("top", "wrist")]}
                elif args.op == "ik-many":
                    queries = json.loads(args.queries)
                    if not isinstance(queries, list) or not 1 <= len(queries) <= 24:
                        raise ValueError("Expected 1..24 queries")
                    seed, bounds = bus.sync_read("Present_Position"), limits(bus.calibration)
                    result = {"solutions": [solve(q["xyz"], q.get("axis"), seed, bounds, q.get("roll"), q.get("offset")) for q in queries]}
                elif args.op == "grip":
                    result = grip(bus)
                    result = {"images": [camera(r) for r in ("top", "wrist")], **result}
                elif args.op in ("move", "move-to", "shift"):
                    plan = None
                    if args.op == "move":
                        pairs = [v.split("=", 1) for v in args.targets]
                        target = {j: float(v) for j, v in pairs}
                        if len(target) != len(pairs):
                            raise ValueError("Duplicate joints")
                        precision = args.precision
                    else:
                        seed = bus.sync_read("Present_Position")
                        if args.op == "shift":
                            import numpy as np
                            delta = np.array(args.delta)
                            if not np.isfinite(delta).all() or np.linalg.norm(delta) > .1:
                                raise ValueError("shift must be finite and at most 0.1 metres")
                            T = tool_pose(seed)
                            delta = T[:3,:3]@delta if args.frame == "tool" else delta
                            plan = cartesian_plan(T[:3,3]+delta, seed, limits(bus.calibration))
                            if plan["max_joint_delta_deg"] > 30:
                                raise ValueError("Relative shift requires over 30 degrees at a joint; use smaller explicit waypoints")
                        else:
                            plan = cartesian_plan(args.xyz, seed, limits(bus.calibration),
                                                  args.axis,args.tilt,args.roll,args.offset)
                        target, precision = plan["joints"], True
                    result = move(bus, target, args.seconds, precision)
                    if plan is not None:
                        result["plan"] = plan
                    images = []
                    for role in ("top", "wrist"):
                        try:
                            images.append(camera(role))
                        except Exception as exc:
                            result.setdefault("image_errors", {})[role] = str(exc)
                    result = {"images": images, **result}
                elif args.op == "ik-options":
                    result = orientation_options(args.xyz, bus.sync_read("Present_Position"),
                                                 limits(bus.calibration), args.roll, args.offset)
                else:
                    result = solve(args.xyz, args.axis, bus.sync_read("Present_Position"),
                                   limits(bus.calibration), args.roll, args.offset)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

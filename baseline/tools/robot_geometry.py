"""Offline, task-independent frame geometry and caller-selected visual alignment.

Transforms map column vectors from source into target; translations are metres.
No function connects to hardware or certifies collision clearance.
"""
import copy
import math
import numpy as np


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def rigid_transform(matrix, source, target):
    t = _array(matrix, (4, 4), "transform")
    r = t[:3, :3]
    if not source or not target or not isinstance(source, str) or not isinstance(target, str):
        raise ValueError("Explicit nonempty source and target frames required")
    if (not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not math.isclose(float(np.linalg.det(r)), 1., abs_tol=1e-6)):
        raise ValueError("Transform must be a proper rigid transform")
    return {"source": source, "target": target, "matrix": t.tolist(), "units": "m"}


def _validated(transform):
    if transform.get("units") != "m":
        raise ValueError("Transform units must be m")
    return rigid_transform(transform["matrix"], transform["source"], transform["target"])


def invert_transform(transform):
    t = _validated(transform)
    m = np.array(t["matrix"])
    inverse = np.eye(4)
    inverse[:3, :3] = m[:3, :3].T
    inverse[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return rigid_transform(inverse, t["target"], t["source"])


def compose_transforms(outer, inner):
    a, b = _validated(outer), _validated(inner)
    if a["source"] != b["target"]:
        raise ValueError("Transform frames do not connect")
    return rigid_transform(np.array(a["matrix"]) @ np.array(b["matrix"]), b["source"], a["target"])


def transform_point(transform, xyz_m, frame):
    t = _validated(transform)
    if frame != t["source"]:
        raise ValueError("Point frame does not match transform source")
    p = _array(xyz_m, (3,), "point")
    with np.errstate(over='ignore', invalid='ignore'):
        output = (np.array(t["matrix"]) @ np.r_[p, 1])[:3]
    return {"xyz_m": _array(output, (3,), "transformed point").tolist(), "frame": t["target"]}


def profile_frames(profile, T_base_urdf=None):
    """Preserve existing profile semantics; never invent absent extrinsics."""
    frames = {}
    if profile.get("projection"):
        p = profile["projection"]
        t = np.eye(4)
        t[:3, :3] = _array(p["R_base_to_camera"], (3, 3), "camera rotation")
        t[:3, 3] = _array(p["t_base_to_camera_m"], (3,), "camera translation")
        frames["base_to_camera"] = rigid_transform(t, "base_link", "camera_optical")
        frames["camera_to_base"] = invert_transform(frames["base_to_camera"])
    tool = profile.get("tool_reference", {})
    if "translation_from_urdf_m" in tool:
        t = np.eye(4)
        t[:3, 3] = _array(tool["translation_from_urdf_m"], (3,), "tool translation")
        if np.linalg.norm(t[:3, 3]) > .05:
            raise ValueError("Tool translation exceeds existing 0.05 m profile limit")
        frames["reference_to_urdf"] = rigid_transform(t, "tool_reference", "gripper_frame_link")
    if T_base_urdf is not None:
        frames["urdf_to_base"] = rigid_transform(T_base_urdf, "gripper_frame_link", "base_link")
        if "reference_to_urdf" in frames:
            frames["reference_to_base"] = compose_transforms(frames["urdf_to_base"], frames["reference_to_urdf"])
    return frames


def _intrinsics(profile):
    k = _array(profile["camera_matrix"], (3, 3), "camera matrix")
    size = _array(profile["image_size"], (2,), "image size")
    d = np.asarray(profile["dist_coeffs"], dtype=float)
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
            or k[0, 1] != 0 or k[1, 0] != 0 or (size <= 0).any()
            or not np.equal(size, np.floor(size)).all()
            or d.ndim != 1 or d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all()):
        raise ValueError("Invalid OpenCV camera intrinsics")
    return k, d, size


def capability_report(profile):
    result = {"absolute_position_uncertainty_m": None, "motion_commanded": False,
              "validation": copy.deepcopy(profile.get("validation")),
              "surface_reference": copy.deepcopy(profile.get("surface_reference")),
              "note": "Fit residuals and held-out FK-relative errors are not absolute metrology uncertainty."}
    try:
        _intrinsics(profile)
        result["intrinsics"] = "available"
    except (KeyError, ValueError, TypeError) as exc:
        result["intrinsics"] = "missing" if isinstance(exc, KeyError) else "invalid"
    try:
        frames = profile_frames({"projection": profile.get("projection")})
        result["extrinsics"] = "available" if "base_to_camera" in frames else "missing"
    except (KeyError, ValueError, TypeError) as exc:
        result.update(extrinsics="invalid", extrinsics_error=str(exc))
    try:
        frames = profile_frames({"tool_reference": profile.get("tool_reference", {})})
        result["tool_reference"] = "profile_translation" if "reference_to_urdf" in frames else "nominal_urdf_only"
    except (KeyError, ValueError, TypeError) as exc:
        result.update(tool_reference="invalid", tool_reference_error=str(exc))
    result["pixel_projection"] = ("camera_extrinsics" if result["intrinsics"] == "available" and result["extrinsics"] == "available"
                                  else "measured_planes_unvalidated" if profile.get("planes") else "unavailable")
    return result


def aperture_midpoint(profile, opening_percent):
    """Measured offset from fixed fingertip to aperture midpoint in tool axes.

Optional schema: aperture_midpoint_model.samples entries contain opening_percent
and offset_from_reference_tool_m. Interpolation is restricted to measured range;
no jaw gap, sign, zero point or opening calibration is inferred from URDF.
"""
    model = profile.get("aperture_midpoint_model")
    if model is None:
        return {"available": False, "reason": "missing_opening_dependent_midpoint_calibration",
                "offset_from_reference_tool_m": None, "motion_commanded": False}
    if model.get("reference") != "tool_reference" or not model.get("provenance"):
        raise ValueError("Midpoint model requires tool_reference and measurement provenance")
    samples = model.get("samples", [])
    if len(samples) < 2:
        raise ValueError("At least two measured midpoint samples required")
    openings = np.array([s['opening_percent'] for s in samples], dtype=float)
    offsets = _array([s['offset_from_reference_tool_m'] for s in samples], (len(samples), 3), "midpoint offsets")
    if (not np.isfinite(openings).all() or (np.diff(openings) <= 0).any()
            or openings[0] < 0 or openings[-1] > 100 or not math.isfinite(opening_percent)):
        raise ValueError("Opening samples must increase within 0..100 percent")
    if not openings[0] <= opening_percent <= openings[-1]:
        raise ValueError("Opening outside measured midpoint range; extrapolation forbidden")
    offset = [float(np.interp(opening_percent, openings, offsets[:, i])) for i in range(3)]
    return {"available": True, "offset_from_reference_tool_m": offset,
            "frame": "tool_reference", "opening_percent": opening_percent,
            "method": "piecewise_linear_measured_offset", "provenance": copy.deepcopy(model['provenance']),
            "absolute_position_uncertainty_m": None, "motion_commanded": False}


def parallel_contact_offset(profile, separation_m):
    """Caller-specified parallel contact span, along the measured aperture axis.

This is an explicit geometric reference convention, not an opening-percent
model, object detector, contact assertion or jaw-reachability determination.
"""
    if isinstance(separation_m, bool) or not math.isfinite(separation_m) or separation_m < 0:
        raise ValueError('Contact separation must be finite nonnegative metres')
    tool = profile.get('tool_reference', {})
    if 'translation_from_urdf_m' not in tool:
        raise ValueError('Measured fixed-tip reference required')
    profile_frames({'tool_reference': tool})
    if 'aperture_axis_tool' not in tool:
        raise ValueError('Measured aperture axis required')
    axis = _array(tool['aperture_axis_tool'], (3,), 'aperture axis')
    if not math.isclose(float(np.linalg.norm(axis)), 1., abs_tol=.001):
        raise ValueError('Aperture axis must be unit length')
    offset = axis*(separation_m/2)
    return {'offset_tool_m': offset.tolist(), 'frame': 'tool_reference',
            'reference': 'parallel_contact_midpoint', 'contact_span_m': separation_m,
            'span_is_caller_supplied': True, 'jaw_opening_percent': None,
            'contact_or_reachability_verified': False, 'motion_commanded': False,
            'note': 'Offset from calibrated fixed fingertip; no axial hinge retraction modeled'}


def project_pixel(profile, pixel_uv, plane_z_m, pixel_sigma_px=0.):
    """Raw pixel to a caller-supplied base-Z plane, with local noise propagation.

Covariance excludes calibration, height, tool and joint uncertainty; it is not
a total accuracy bound. This API requires extrinsics, not a plane homography.
"""
    import cv2
    k, d, size = _intrinsics(profile)
    uv = _array(pixel_uv, (2,), "pixel")
    if (uv < 0).any() or (uv >= size).any():
        raise ValueError("Pixel outside calibrated image")
    if not math.isfinite(plane_z_m) or not math.isfinite(pixel_sigma_px) or pixel_sigma_px < 0:
        raise ValueError("Finite height and nonnegative pixel sigma required")
    frames = profile_frames(profile)
    if "camera_to_base" not in frames:
        raise ValueError("Missing camera extrinsics; intrinsics alone cannot produce base coordinates")
    t = np.array(frames["camera_to_base"]["matrix"])
    def project(uv):
        ray = np.r_[cv2.undistortPoints(uv.reshape(1, 1, 2), k, d)[0, 0], 1.]
        direction, origin = t[:3, :3] @ ray, t[:3, 3]
        if abs(direction[2]) < 1e-8:
            raise ValueError("Ray parallel to requested plane")
        distance = (plane_z_m-origin[2])/direction[2]
        if distance <= 0:
            raise ValueError("Plane intersection behind camera")
        with np.errstate(over='ignore', invalid='ignore'):
            point = origin + distance*direction
        return _array(point, (3,), "projected point")
    point = project(uv)
    eps = .001
    jac = np.column_stack([(project(uv+e*eps)-project(uv-e*eps))/(2*eps) for e in np.eye(2)])
    with np.errstate(over='ignore', invalid='ignore'):
        weighted_jac = jac * pixel_sigma_px
        covariance = weighted_jac @ weighted_jac.T
    covariance = _array(covariance, (3, 3), "pixel covariance")
    return {"xyz_m": point.tolist(), "frame": "base_link", "plane_z_m": plane_z_m,
            "pixel_noise_covariance_m2": covariance.tolist(),
            "absolute_position_uncertainty_m": None, "height_is_caller_supplied": True,
            "uncertainty_excludes": ["camera_calibration", "plane_height", "tool_reference", "joint_kinematics"],
            "motion_commanded": False}


def feature_alignment(observed_uv, target_uv, jacobian_px_per_m, max_step_m,
                      *, axis_names, frame, damping=0., max_condition=1000.):
    """Bounded least-squares correction for caller-matched features and axes.

Jacobian is d(observed pixel coordinates)/d(caller axis displacement).
Caller must acquire fresh observations and revalidate the resulting path.
"""
    observed, target = np.asarray(observed_uv, float), np.asarray(target_uv, float)
    if (observed.ndim != 2 or observed.shape[1] != 2 or observed.shape[0] == 0
            or target.shape != observed.shape or not np.isfinite(observed).all() or not np.isfinite(target).all()):
        raise ValueError("Matching finite Nx2 observed/target features required")
    if (not axis_names or len(set(axis_names)) != len(axis_names)
            or not all(isinstance(a, str) and a for a in axis_names) or not isinstance(frame, str) or not frame):
        raise ValueError("Explicit unique axes and frame required")
    j = _array(jacobian_px_per_m, (observed.size, len(axis_names)), "Jacobian")
    if not all(math.isfinite(v) for v in (max_step_m, damping, max_condition)) or max_step_m <= 0 or damping < 0 or max_condition < 1:
        raise ValueError("Invalid step, damping or condition bound")
    with np.errstate(over='ignore', invalid='ignore'):
        error = (target-observed).ravel()
    error = _array(error, (observed.size,), "feature error")
    u, s, vt = np.linalg.svd(j, full_matrices=False)
    rank = int(np.linalg.matrix_rank(j))
    with np.errstate(over='ignore', invalid='ignore'):
        condition = float(s[0]/s[-1]) if s[-1] > 0 else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    valid = rank == len(axis_names) and condition is not None and condition <= max_condition
    result = {"usable": valid, "reason": None if valid else "unobservable_or_ill_conditioned_axes",
              "rank": rank, "condition": condition, "frame": frame, "axis_names": list(axis_names),
              "motion_commanded": False, "collision_checked": False}
    if not valid:
        return {**result, "delta_m": None}
    # Avoid squaring very small/large singular values; never return NaN motion.
    denominator = np.hypot(s, damping)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        delta = vt.T @ (((s/denominator)/denominator)*(u.T @ error))
    if not np.isfinite(delta).all():
        return {**result, "usable": False, "reason": "numerical_range_exceeded", "delta_m": None}
    norm = math.hypot(*delta)
    if not math.isfinite(norm):
        return {**result, "usable": False, "reason": "numerical_range_exceeded", "delta_m": None}
    if norm > max_step_m:
        delta *= max_step_m/norm
    with np.errstate(over='ignore', invalid='ignore'):
        predicted_error = math.hypot(*(error-j @ delta))
    error_before = math.hypot(*error)
    if not math.isfinite(predicted_error) or not math.isfinite(error_before):
        return {**result, "usable": False, "reason": "numerical_range_exceeded", "delta_m": None}
    return {**result, "delta_m": delta.tolist(), "step_clipped": norm > max_step_m,
            "error_before_px": error_before,
            "predicted_error_after_px": predicted_error,
            "requires_fresh_observation_and_path_validation": True}


def fit_tool_point(profile, views, *, max_condition=1000., max_error_px=5.,
                   max_offset_m=.2):
    """Fit one rigid tool-frame feature from caller-labeled multi-view pixels.

Each view has id, pixel_uv, T_base_tool (base <- tool_reference), and split
('fit' or 'held_out'). The feature must stay fixed in tool coordinates across
views, e.g. moving-jaw observations must all have the same measured opening.
Fit/held-out errors are camera/FK-relative, not an absolute accuracy estimate.
"""
    import cv2
    k, d, size = _intrinsics(profile)
    frames = profile_frames({"projection": profile.get("projection")})
    if 'base_to_camera' not in frames:
        raise ValueError("Tool-point fitting requires camera extrinsics")
    camera = np.array(frames['base_to_camera']['matrix'])
    if (not all(math.isfinite(v) and v > 0 for v in (max_condition, max_error_px, max_offset_m))
            or max_condition < 1):
        raise ValueError("Finite positive fit thresholds required")
    if len(views) < 3:
        raise ValueError("At least two fit views and one held-out view required")
    matrices, rhs, prepared, ids = [], [], [], set()
    for view in views:
        identity = view['id']
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Unique nonempty observation ids required")
        ids.add(identity)
        split = view['split']
        if split not in ('fit', 'held_out'):
            raise ValueError("View split must be fit or held_out")
        uv = _array(view['pixel_uv'], (2,), 'feature pixel')
        if (uv < 0).any() or (uv >= size).any():
            raise ValueError("Feature pixel outside calibrated image")
        tool = np.array(rigid_transform(view['T_base_tool'], 'tool_reference', 'base_link')['matrix'])
        camera_tool = camera @ tool
        ray = np.r_[cv2.undistortPoints(uv.reshape(1, 1, 2), k, d)[0, 0], 1.]
        direction = camera_tool[:3, :3].T @ ray
        direction /= np.linalg.norm(direction)
        origin = -camera_tool[:3, :3].T @ camera_tool[:3, 3]
        orthogonal = np.eye(3)-np.outer(direction, direction)
        if split == 'fit':
            matrices.append(orthogonal)
            rhs.append(orthogonal @ origin)
        prepared.append((identity, split, uv, camera_tool))
    if len(matrices) < 2 or not any(v[1] == 'held_out' for v in prepared):
        raise ValueError("Two fit views and one independent held-out view required")
    a, b = np.vstack(matrices), np.concatenate(rhs)
    point, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    result = {'available': False, 'offset_from_reference_tool_m': None,
              'rank': int(rank), 'condition': condition, 'motion_commanded': False,
              'absolute_position_uncertainty_m': None,
              'method': 'multi_view_ray_least_squares',
              'assumption': 'Same physical feature fixed in tool coordinates for all views'}
    if rank < 3 or condition is None or not math.isfinite(condition) or condition > max_condition:
        return {**result, 'reason': 'insufficient_view_geometry'}
    if not np.isfinite(point).all() or np.linalg.norm(point) > max_offset_m:
        return {**result, 'reason': 'tool_point_outside_bound'}
    errors = []
    for identity, split, uv, ct in prepared:
        camera_point = ct[:3, :3] @ point+ct[:3, 3]
        if camera_point[2] <= 0:
            return {**result, 'reason': 'feature_behind_camera'}
        projected = cv2.projectPoints(camera_point.reshape(1, 3), np.zeros(3), np.zeros(3), k, d)[0].reshape(2)
        error = float(np.linalg.norm(projected-uv))
        errors.append({'id': identity, 'split': split, 'pixel_error': error,
                       'predicted_uv': projected.tolist()})
    result['observations'] = errors
    if any(not math.isfinite(e['pixel_error']) or e['pixel_error'] > max_error_px for e in errors):
        return {**result, 'reason': 'reprojection_error_exceeded'}
    return {**result, 'available': True, 'reason': None,
            'offset_from_reference_tool_m': point.tolist()}

"""Offline camera-derived local Jacobians for caller-selected image features.

No feature selection, inferred depth, device access, or motion commands.
"""
import math
import numpy as np

try:
    from . import robot_geometry as geometry
except ImportError:
    import robot_geometry as geometry


def camera_jacobian(profile, pixel_uv, plane_z_m, *, axis_names,
                    frame="base_link", pixel_sigma_px=0., height_sigma_m=0.,
                    max_condition=1000.):
    """Return d(raw image pixels)/d(selected base translations), in px/m.

    The caller's base-Z plane locates the observed feature; it is not a fitted
    surface or a measured depth. A selected Z motion axis is a local derivative
    away from that plane. One feature cannot observe arbitrary XYZ translation.
    Output axes/frame/Jacobian match robot_geometry.feature_alignment inputs.
    """
    import cv2
    if frame != "base_link":
        raise ValueError("Only explicit base_link translation axes are supported")
    if (not isinstance(axis_names, (list, tuple)) or not axis_names
            or any(n not in ("x", "y", "z") for n in axis_names)
            or len(set(axis_names)) != len(axis_names)):
        raise ValueError("Select unique axes from x, y, z")
    for name, value in (("plane_z_m", plane_z_m), ("pixel_sigma_px", pixel_sigma_px),
                        ("height_sigma_m", height_sigma_m), ("max_condition", max_condition)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if pixel_sigma_px < 0 or height_sigma_m < 0 or max_condition < 1:
        raise ValueError("Nonnegative noise and condition limit at least one required")
    located = geometry.project_pixel(profile, pixel_uv, plane_z_m, pixel_sigma_px)
    point = np.array(located["xyz_m"])
    frames = geometry.profile_frames({"projection": profile.get("projection")})
    transform = np.array(frames["base_to_camera"]["matrix"])
    k, distortion, _ = geometry._intrinsics(profile)
    camera_point = transform[:3, :3] @ point+transform[:3, 3]
    if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-8:
        raise ValueError("Located feature must have positive finite camera depth")
    projected, derivatives = cv2.projectPoints(camera_point.reshape(1, 3),
                                               np.zeros(3), np.zeros(3), k, distortion)
    # OpenCV translation columns give raw-pixel derivatives wrt camera XYZ,
    # including lens distortion. Rotate base translation vectors into camera.
    full_jacobian = derivatives[:, 3:6] @ transform[:3, :3]
    jacobian = geometry._array(full_jacobian[:, [("x", "y", "z").index(n) for n in axis_names]],
                               (2, len(axis_names)), "pixel Jacobian")
    reprojection_error = float(np.linalg.norm(projected.reshape(2)-np.asarray(pixel_uv)))
    if not math.isfinite(reprojection_error):
        raise ValueError("Nonfinite camera reprojection")
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(singular[0]/singular[-1]) if rank == len(axis_names) and singular[-1] > 0 else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    usable = condition is not None and condition <= max_condition and reprojection_error <= .25
    reason = ("undistortion_roundtrip_error" if reprojection_error > .25 else
              "unobservable_or_ill_conditioned_axes" if not usable else None)
    origin = np.array(frames["camera_to_base"]["matrix"])[:3, 3]
    ray = point-origin
    if abs(ray[2]) <= 1e-12:
        raise ValueError("Plane-height sensitivity is singular")
    height_derivative = ray/ray[2]
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.array(located["pixel_noise_covariance_m2"])+np.outer(
            height_derivative*height_sigma_m, height_derivative*height_sigma_m)
    covariance = geometry._array(covariance, (3, 3), "point covariance")
    return {"usable": usable, "reason": reason, "pixel_uv": [float(v) for v in pixel_uv],
            "point_xyz_m": point.tolist(), "plane_z_m": plane_z_m,
            "plane_height_is_caller_supplied": True, "frame": frame,
            "axis_names": list(axis_names), "jacobian_px_per_m": jacobian.tolist(),
            "rank": rank, "condition": condition, "max_condition": max_condition,
            "singular_values_px_per_m": singular.tolist(),
            "reprojection_error_px": reprojection_error,
            "point_height_derivative_m_per_m": height_derivative.tolist(),
            "point_covariance_m2": covariance.tolist(),
            "pixel_sigma_px": pixel_sigma_px, "height_sigma_m": height_sigma_m,
            "absolute_position_uncertainty_m": None,
            "uncertainty_excludes": ["camera_calibration", "plane_height_bias", "tool_reference", "joint_kinematics",
                                     "feature_correspondence", "unmodeled_surface_shape"],
            "assumptions": "Independent isotropic pixel noise and independent supplied height noise; local translation of the observed feature, fixed camera. Covariance is not a total accuracy bound.",
            "motion_commanded": False, "collision_checked": False,
            "requires_fresh_observation_and_path_validation": True}

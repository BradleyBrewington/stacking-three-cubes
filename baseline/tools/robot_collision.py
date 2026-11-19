"""Explicit capsule approximation for offline collision queries.

Geometry dimensions and exclusions MUST come from a reviewed station model.
No collision geometry is inferred from FK origins or camera fit residuals.
"""
import math


def _point(p):
    if len(p) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                          or not math.isfinite(x) for x in p):
        raise ValueError("expected finite XYZ metres")
    return tuple(float(x) for x in p)


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _closest(p, a, b):
    u = _sub(b, a)
    norm = _dot(u, u)
    t = max(0., min(1., _dot(_sub(p, a), u)/norm)) if norm else 0.
    return tuple(x+t*y for x, y in zip(a, u))


def segment_distance(a, b, c, d):
    """Exact minimum Euclidean distance between two closed 3D segments."""
    a, b, c, d = map(_point, (a, b, c, d))
    pairs = [(a, _closest(a, c, d)), (b, _closest(b, c, d)),
             (c, _closest(c, a, b)), (d, _closest(d, a, b))]
    u, v, w = _sub(b, a), _sub(d, c), _sub(a, c)
    aa, bb, cc, dd, ee = _dot(u, u), _dot(u, v), _dot(v, v), _dot(u, w), _dot(v, w)
    det = aa*cc-bb*bb
    if det > 1e-14*aa*cc:
        s, t = (bb*ee-cc*dd)/det, (aa*ee-bb*dd)/det
        if 0 <= s <= 1 and 0 <= t <= 1:
            pairs.append((tuple(x+s*y for x, y in zip(a, u)),
                          tuple(x+t*y for x, y in zip(c, v))))
    return min(math.sqrt(_dot(_sub(p, q), _sub(p, q))) for p, q in pairs)


def _positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(name + " must be finite and " + ("nonnegative" if zero else "positive"))
    return float(value)


def _box(value):
    if value is None:
        return None
    center = _point(value["center"])
    axes = [_point(a) for a in value["axes"]]
    half = _point(value["half_extents"])
    if len(axes) != 3 or any(h <= 0 for h in half) or any(
            abs(_dot(axes[i], axes[j])-(1. if i == j else 0.)) > 1e-8
            for i in range(3) for j in range(3)):
        raise ValueError("box axes must be orthonormal and extents positive")
    return center, axes, half


def _box_gap(a, b):
    """Separating-axis projected gap: a lower bound on Euclidean clearance."""
    ca, aa, ha = a
    cb, ab, hb = b
    axes = aa+ab
    for u in aa:
        for v in ab:
            cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            norm = math.sqrt(_dot(cross, cross))
            if norm > 1e-10:
                axes.append(tuple(x/norm for x in cross))
    delta = _sub(cb, ca)
    return max(abs(_dot(delta, axis))-sum(h*abs(_dot(u, axis)) for h, u in zip(ha, aa))
               -sum(h*abs(_dot(u, axis)) for h, u in zip(hb, ab)) for axis in axes)


def _box_point_gap(box, point):
    center, axes, half = box
    delta = _sub(point, center)
    return math.sqrt(sum(max(0., abs(_dot(delta, axis))-h)**2 for axis, h in zip(axes, half)))


class CapsuleCollisionModel:
    """Capsules vs horizontal table, sphere obstacles, and each other.

    capsules_at(q) returns [{name,a,b,radius_m}] in base_link. Adjacent links
    generally overlap; explicit excluded_pairs must describe reviewed contact
    pairs. Every other pair is checked. Table exclusions are explicit too.
    margin_m expands environment clearance and pair separation, once per pair.
    Unknown objects, cables, and unmodeled gripper states remain out of scope.
    """
    def __init__(self, capsules_at, *, model_id, table_z_m, margin_m,
                 spheres=(), excluded_pairs=(), table_exclusions=(),
                 tracking_margin_m=0., downstream_reach_m=None, self_pair_tracking_m=None,
                 component_tracking_m=None, table_tracking_m=None):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id is required")
        if isinstance(table_z_m, bool) or not math.isfinite(table_z_m):
            raise ValueError("table height must be finite")
        self.capsules_at = capsules_at
        self.table_z_m = table_z_m
        self.margin_m = _positive(margin_m, "margin", zero=True)
        self.tracking_margin_m = _positive(tracking_margin_m, "tracking margin", zero=True)
        self.self_pair_tracking_m = {frozenset(pair): _positive(v, "pair tracking margin", zero=True)
                                     for pair, v in (self_pair_tracking_m or {}).items()}
        self.component_tracking_m = {n: _positive(v, "component tracking", zero=True) for n, v in (component_tracking_m or {}).items()}
        self.table_tracking_m = {n: _positive(v, "table tracking", zero=True) for n, v in (table_tracking_m or {}).items()}
        reaches = {n: _positive(v, "downstream reach") for n, v in (downstream_reach_m or {}).items()}
        if self.tracking_margin_m and not reaches:
            raise ValueError("tracking margin requires conservative per-joint reach bounds")
        self.spheres = [(_point(s["center"]), _positive(s["radius_m"], "sphere radius")) for s in spheres]
        self.excluded_pairs = set()
        for pair in excluded_pairs:
            if len(pair) != 2 or len(set(pair)) != 2:
                raise ValueError("excluded pair must name two different capsules")
            self.excluded_pairs.add(frozenset(pair))
        self.table_exclusions = set(table_exclusions)
        self.metadata = {"id": model_id,
                         "coverage": "caller-supplied capsules vs horizontal table, sphere obstacles and nonexcluded capsule pairs",
                         "margin_m": self.margin_m, "table_z_m": table_z_m,
                         "tracking_margin_m": self.tracking_margin_m,
                         "tracking_margin_applied": True, "downstream_reach_m": reaches,
                         "spheres": [{"center": list(c), "radius_m": r} for c, r in self.spheres],
                         "excluded_pairs": [sorted(p) for p in self.excluded_pairs],
                         "table_exclusions": sorted(self.table_exclusions),
                         "continuous": False, "geometry_source": "caller-reviewed approximation"}

    def __call__(self, joints):
        return self.inspect(joints)["clear"]

    def inspect(self, joints):
        capsules, names = [], set()
        for c in self.capsules_at(dict(joints)):
            name = c["name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("capsule names must be unique nonempty strings")
            names.add(name)
            capsules.append((name, _point(c["a"]), _point(c["b"]), _positive(c["radius_m"], "capsule radius"), _box(c.get("box"))))
        if not capsules or not self.table_exclusions <= names or any(not p <= names for p in self.excluded_pairs):
            raise ValueError("empty geometry or exclusions name absent capsules")
        minimum = math.inf
        closest = None
        boxes = {}
        for name, a, b, radius, box in capsules:
            lo = [min(a[k], b[k])-radius for k in range(3)]
            hi = [max(a[k], b[k])+radius for k in range(3)]
            if box:
                extents = [sum(abs(axis[k])*h for axis, h in zip(box[1], box[2])) for k in range(3)]
                lo = [max(lo[k], box[0][k]-extents[k]) for k in range(3)]
                hi = [min(hi[k], box[0][k]+extents[k]) for k in range(3)]
            boxes[name] = lo, hi

        def record(clearance, pair):
            nonlocal minimum, closest
            if clearance < minimum:
                minimum, closest = clearance, pair

        for i, (name, a, b, radius, box) in enumerate(capsules):
            if name not in self.table_exclusions:
                lower_z = min(a[2], b[2])-radius
                if box:
                    lower_z = max(lower_z, box[0][2]-sum(abs(axis[2])*h for axis, h in zip(box[1], box[2])))
                table_margin = self.table_tracking_m.get(name, self.tracking_margin_m)
                record(lower_z-self.table_z_m-self.margin_m-table_margin, [name, "table"])
            for j, (center, r) in enumerate(self.spheres):
                gap = segment_distance(a, b, center, center)-radius
                if box:
                    gap = max(gap, _box_point_gap(box, center))
                record(gap-r-self.margin_m-self.component_tracking_m.get(name, self.tracking_margin_m), [name, "sphere:"+str(j)])
            for other, c, d, r, other_box in capsules[i+1:]:
                if frozenset((name, other)) not in self.excluded_pairs:
                    pair_margin = self.self_pair_tracking_m.get(frozenset((name, other)), 2*self.tracking_margin_m)
                    la, ua = boxes[name]
                    lb, ub = boxes[other]
                    broad_gap = max(max(lb[k]-ua[k], la[k]-ub[k]) for k in range(3))
                    if broad_gap-self.margin_m-pair_margin >= minimum:
                        continue
                    gap = segment_distance(a, b, c, d)-radius-r
                    if box and other_box:
                        gap = max(gap, _box_gap(box, other_box))
                    record(gap-self.margin_m-pair_margin, [name, other])
        if minimum == math.inf:
            raise ValueError("model excludes every collision check")
        return {"clear": minimum > 0., "minimum_clearance_m": minimum,
                "closest_pair": closest, "model_id": self.metadata["id"]}


def resource_capsules(resources, radii_m, tool_capsules):
    """Return a callable deriving capsules from existing URDF FK chain.

    radii_m must name every chain segment (joint names from resources.chain()).
    Each capsule joins the preceding origin to this joint origin. Reviewed radii
    must enclose actual links; origins alone do not supply physical dimensions.
    tool_capsules are {name,a,b,radius_m} in nominal gripper_frame_link, covering
    the entire jaw opening range the caller intends to use. They are mandatory.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    chain = resources.chain()
    if set(radii_m) != {name for name, _, _ in chain} or not tool_capsules:
        raise ValueError("all chain radii and explicit gripper envelope required")
    radii = {name: _positive(radius, "link radius") for name, radius in radii_m.items()}
    envelope = [{"name": c["name"], "a": _point(c["a"]), "b": _point(c["b"]),
                 "radius_m": _positive(c["radius_m"], "tool radius")} for c in tool_capsules]

    def capsules(q):
        T, previous, result = np.eye(4), np.zeros(3), []
        for name, fixed, axis in chain:
            T = T @ fixed
            current = T[:3, 3].copy()
            result.append({"name": name, "a": previous.tolist(), "b": current.tolist(), "radius_m": radii[name]})
            if axis is not None:
                angle = float(q[name])
                if not math.isfinite(angle):
                    raise ValueError("nonfinite joint angle")
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(axis*math.radians(angle)).as_matrix()
                T = T @ rotation
            previous = current
        for c in envelope:
            result.append({**c, "a": (T[:3, 3]+T[:3, :3]@c["a"]).tolist(),
                           "b": (T[:3, 3]+T[:3, :3]@c["b"]).tolist()})
        return result

    return capsules


def resource_sweep_bound(resources, radii_m, tool_capsules):
    """Conservative travel bound for any modeled point during joint interpolation.

    Whole-chain reach bounds each point's distance from every rotation axis.
    Sum of |joint angle change| times reach bounds surface travel, even when
    the tool point is stationary. This bound is deliberately loose. It assumes
    a rigid chain and a gripper envelope enclosing every intended opening.
    """
    # Reuse dimensional/completeness validation without running any FK.
    resource_capsules(resources, radii_m, tool_capsules)
    chain = resources.chain()
    reach = sum(math.sqrt(sum(float(x)**2 for x in fixed[:3, 3])) for _, fixed, _ in chain)
    reach += max([float(r) for r in radii_m.values()] +
                 [math.sqrt(sum(x*x for x in _point(c[k])))+c["radius_m"]
                  for c in tool_capsules for k in ("a", "b")])
    names = [name for name, _, axis in chain if axis is not None]

    def bound(a, b):
        return reach*sum(abs(math.radians(float(b[n])-float(a[n]))) for n in names)

    bound.downstream_reach_m = {n: reach for n in names}
    return bound


def cad_model(resources, geometry, *, table_z_m, uncertainty_margin_m,
              tracking_margin_m, spheres=(), scope="full",
              gripper_phase_deg=0., gripper_phase_uncertainty_deg=0.,
              jaw_angle_range_rad=None, joint_tracking_tolerance_deg=None):
    """Instantiate a derived CAD model against the current control FK chain.

    Nominal CAD envelopes are usable offline; physical dimensions, mounting,
    table uncertainty, and excluded contacts still need station acceptance.
    Jaw geometry encloses the entire URDF joint range, with angular-sampling
    inflation, so no percent-to-angle mapping is assumed. Mechanical range
    containment in this CAD range must be accepted during station review.
    """
    import hashlib
    import json
    import numpy as np
    from scipy.spatial.transform import Rotation
    geometry = json.loads(json.dumps(geometry, allow_nan=False))
    if scope not in ("full", "gripper_assembly_external"):
        raise ValueError("unknown collision scope")
    if isinstance(gripper_phase_deg, bool) or not math.isfinite(gripper_phase_deg):
        raise ValueError("invalid gripper phase")
    uncertainty_angle = _positive(gripper_phase_uncertainty_deg, "phase uncertainty", zero=True)
    if uncertainty_angle > 180:
        raise ValueError("phase uncertainty must not exceed 180 degrees")
    if geometry.get("schema_version") != 1 or geometry.get("jaw_mode") != "urdf_range_swept_envelope_no_percent_to_angle_assumption":
        raise ValueError("unsupported CAD collision schema or jaw convention")
    joints = geometry["joints"]
    by_name = {j["name"]: j for j in joints}
    for name, fixed, axis in resources.chain():
        j = by_name[name]
        if not np.allclose(fixed, j["origin"], atol=1e-10, rtol=0):
            raise ValueError("CAD/control origin mismatch: "+name)
        if axis is not None and not np.allclose(axis, j["axis"], atol=1e-10, rtol=0):
            raise ValueError("CAD/control joint axis mismatch: "+name)
    nominal = geometry["capsules"]
    if not nominal or not any(c.get("jaw_swept_envelope") for c in nominal):
        raise ValueError("missing CAD jaw envelope")
    if jaw_angle_range_rad is not None:
        jaw_angles = sorted({c["angular_sample_rad"] for c in nominal if c.get("jaw_swept_envelope")})
        if len(jaw_angle_range_rad) != 2:
            raise ValueError("jaw interval requires two angles")
        lo, hi = jaw_angle_range_rad
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (lo, hi)) or not jaw_angles[0] <= lo <= hi <= jaw_angles[-1]:
            raise ValueError("jaw interval must lie within source mechanical range")
        half_step = min(b-a for a, b in zip(jaw_angles, jaw_angles[1:]))/2
        nominal = [c for c in nominal if not c.get("jaw_swept_envelope") or
                   lo-half_step-1e-12 <= c["angular_sample_rad"] <= hi+half_step+1e-12]
    names = [name for name, _, axis in resources.chain() if axis is not None]
    gripper_links = {"gripper_link", "moving_jaw_so101_v1_link"}
    assembly_links = gripper_links | {"wrist_link"}
    jaw_origin = np.asarray(by_name["gripper"]["origin"])
    # Per-component radial bound avoids inflating a fixed fingertip by the
    # reach of a different, fully opened jaw. Both enclosing shapes bound it.
    import itertools
    phase_inflations = {}
    for c in nominal:
        if c["link"] not in gripper_links:
            phase_inflations[c["name"]] = 0.
            continue
        def to_gripper(point):
            return jaw_origin[:3, 3]+jaw_origin[:3, :3]@point if c["link"] == "moving_jaw_so101_v1_link" else np.asarray(point)
        cap_reach = max(float(np.linalg.norm(to_gripper(c[k])[:2]))+c["radius_m"] for k in ("a", "b"))
        corners = itertools.product(*zip(c["box_min"], c["box_max"]))
        box_reach = max(float(np.linalg.norm(to_gripper(np.asarray(p))[:2])) for p in corners)
        phase_inflations[c["name"]] = min(cap_reach, box_reach)*math.radians(uncertainty_angle)

    def capsules(q):
        if set(q) != set(names):
            raise ValueError("CAD model requires all five arm joints in degrees")
        frames, remaining = {"base_link": np.eye(4)}, list(joints)
        while remaining:
            ready = [j for j in remaining if j["parent"] in frames]
            if not ready:
                raise ValueError("invalid CAD joint topology")
            for j in ready:
                T = frames[j["parent"]] @ np.asarray(j["origin"], float)
                if j["type"] != "fixed" and j["name"] != "gripper":
                    value = float(q[j["name"]])
                    if not math.isfinite(value):
                        raise ValueError("nonfinite CAD joint")
                    if j["name"] == "wrist_roll":
                        value += gripper_phase_deg
                    turn = np.eye(4)
                    turn[:3, :3] = Rotation.from_rotvec(np.asarray(j["axis"])*math.radians(value)).as_matrix()
                    T = T @ turn
                frames[j["child"]] = T
                remaining.remove(j)
        return [{"name": c["name"], "radius_m": c["radius_m"]+phase_inflations[c["name"]],
                 "box": {"center": (frames[c["link"]][:3, 3]+frames[c["link"]][:3, :3]@((np.asarray(c["box_min"])+c["box_max"])/2)).tolist(),
                         "axes": frames[c["link"]][:3, :3].T.tolist(),
                         "half_extents": ((np.asarray(c["box_max"])-c["box_min"])/2+phase_inflations[c["name"]]).tolist()},
                 "a": (frames[c["link"]][:3, 3]+frames[c["link"]][:3, :3]@c["a"]).tolist(),
                 "b": (frames[c["link"]][:3, 3]+frames[c["link"]][:3, :3]@c["b"]).tolist()}
                for c in nominal]

    parent_joint = {j["child"]: j for j in joints}
    reaches = {}
    component_reaches = {c["name"]: {} for c in nominal}
    for n in names:
        candidates = []
        root_link = by_name[n]["child"]
        for c in nominal:
            link = c["link"]
            capsule_reach = max(float(np.linalg.norm(c[k]))+c["radius_m"] for k in ("a", "b"))
            box_reach = max(float(np.linalg.norm(p)) for p in itertools.product(*zip(c["box_min"], c["box_max"])))
            length = max(capsule_reach, box_reach)+math.sqrt(3.)*phase_inflations[c["name"]]
            while link != root_link and link in parent_joint:
                step = parent_joint[link]
                length += float(np.linalg.norm(np.asarray(step["origin"])[:3, 3]))
                link = step["parent"]
            if link == root_link:
                candidates.append(length)
                component_reaches[c["name"]][n] = length
        if not candidates:
            raise ValueError("joint has no downstream collision geometry")
        reaches[n] = max(candidates)

    def sweep(a, b):
        return sum(reaches[n]*abs(math.radians(b[n]-a[n])) for n in names)

    sweep.downstream_reach_m = reaches
    pair_tracking = {}
    component_tracking, table_tracking = {}, {}
    if joint_tracking_tolerance_deg is not None:
        if set(joint_tracking_tolerance_deg) != set(names):
            raise ValueError("tracking tolerances must name all arm joints")
        tolerances = {n: _positive(v, "joint tracking tolerance", zero=True)
                      for n, v in joint_tracking_tolerance_deg.items()}
        if sum(reaches[n]*math.radians(tolerances[n]) for n in names) > tracking_margin_m:
            raise ValueError("tracking margin insufficient for joint tolerance budget")
        for c in nominal:
            cr = component_reaches[c["name"]]
            component_tracking[c["name"]] = sum(r*math.radians(tolerances[n]) for n, r in cr.items())
            table_tracking[c["name"]] = 0.
            for n, r in cr.items():
                factor = 1.
                joint = by_name[n]
                if joint["parent"] == "base_link":
                    world_axis = np.asarray(joint["origin"])[:3, :3]@joint["axis"]
                    factor = min(1., float(np.linalg.norm(world_axis[:2])))
                table_tracking[c["name"]] += factor*r*math.radians(tolerances[n])
        for i, a in enumerate(nominal):
            ar = component_reaches[a["name"]]
            for b in nominal[i+1:]:
                br = component_reaches[b["name"]]
                unique = set(ar) ^ set(br)
                pair_tracking[frozenset((a["name"], b["name"]))] = sum(
                    (ar.get(n, 0.)+br.get(n, 0.))*math.radians(tolerances[n]) for n in unique)
    identity = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    active_names = {c["name"] for c in nominal}
    exclusions = [p for p in geometry["excluded_pairs"] if set(p) <= active_names]
    if scope == "gripper_assembly_external":
        exclusions.extend([a["name"], b["name"]] for i, a in enumerate(nominal) for b in nominal[i+1:]
                          if a["link"] in assembly_links and b["link"] in assembly_links)
    model = CapsuleCollisionModel(capsules, model_id=identity, table_z_m=table_z_m,
                                  margin_m=uncertainty_margin_m, tracking_margin_m=tracking_margin_m,
                                  downstream_reach_m=reaches, spheres=spheres,
                                  self_pair_tracking_m=pair_tracking,
                                  component_tracking_m=component_tracking, table_tracking_m=table_tracking,
                                  excluded_pairs=exclusions,
                                  table_exclusions=geometry["table_exclusions"])
    model.metadata.update(geometry_source="hashed SO101 CAD vertex-enclosing capsules and oriented boxes",
                          physical_validation=geometry["physical_validation"],
                          jaw_mode=geometry["jaw_mode"],
                          asset_hashes=geometry["assets_sha256"],
                          exclusions_basis=geometry["exclusion_basis"])
    model.metadata.update(scope=scope, internal_gripper_assembly_checked=scope == "full",
                          gripper_phase_deg=gripper_phase_deg,
                          gripper_phase_uncertainty_deg=uncertainty_angle,
                          jaw_angle_range_rad=jaw_angle_range_rad,
                          gripper_phase_inflation_m=max(phase_inflations.values()),
                          gripper_component_phase_inflation_m=phase_inflations)
    model.metadata.update(joint_tracking_tolerance_deg=joint_tracking_tolerance_deg,
                          self_tracking_policy="relative_kinematic_bound" if pair_tracking else "twice_global_margin")
    return model, sweep

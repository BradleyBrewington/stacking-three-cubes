"""Trusted station boundary for generic plan and execute commands."""
import hashlib
import json
import math
import os
from pathlib import Path
import socket


def lease_valid():
    token = os.environ.get('ARMFARM_WRITER_LEASE_TOKEN')
    if not token:
        return False
    path = os.environ.get('ARMFARM_RECORDING_SOCKET', '/run/armfarm-recording-armfarm/control.sock')
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(.1)
            client.connect(path)
            client.sendall((json.dumps({'op': 'lease_check', 'token': token})+'\n').encode())
            data = bytearray()
            while b'\n' not in data:
                chunk = client.recv(4096)
                if not chunk or len(data)+len(chunk) > 16384:
                    return False
                data.extend(chunk)
        result = json.loads(data.split(b'\n', 1)[0])
        return result.get('ok') is True and result.get('valid') is True
    except (OSError, ValueError):
        return False


def resource_identity(root):
    root = Path(root)
    names = {p.name for p in root.glob('robot_*.py')}
    names.update(('trajectory_execution.py', 'motion_commands.py', 'robot_geometry.json',
                  'robot_collision.json', 'so101_collision_geometry.json',
                  'calibrate_workspace.py', 'fk.py', 'recording.py'))
    # Runtime model assets are explicit siblings, not caller-selected host paths.
    names.update(p.name for p in root.glob('robot_collision_*.json'))
    result = {}
    for name in sorted(names):
        path = root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or redirected runtime dependency: '+name)
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _context(resources):
    root = Path(resources.__file__).resolve().parent
    runtime = Path(os.environ.get('ARMFARM_RECORDING_SOCKET', '/run/armfarm-recording-armfarm/control.sock')).parent
    token = os.environ.get('ARMFARM_WRITER_LEASE_TOKEN')
    if not token or not lease_valid():
        raise ValueError('An active enforced writer lease is required')
    return root, runtime/'motion-plans', token


def plan_command(resources, bus, args):
    if not Path(resources.__file__).with_name('robot_collision.json').exists():
        return {'status': 'unavailable', 'reason': 'station_collision_model_uncommissioned',
                'collision_checked': False, 'motion_commanded': False}
    import robot_motion
    from robot_plan_store import save_plan
    from trajectory_execution import prepare_execution
    root, store, token = _context(resources)
    hashes = resource_identity(root)
    config = json.loads((root/'robot_collision.json').read_text())
    speed = config.get('max_joint_speed_deg_s', 12.)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0 < speed <= 12:
        raise ValueError('Installed max_joint_speed_deg_s must be finite and in (0,12]')
    observed = resources.state(bus)
    plan = robot_motion.plan_to(resources, observed, args.xyz, args.approach, args.clearance,
                                collision_config=config, tilts_deg=args.tilts,
                                roll_deg=args.roll, offset_m=args.offset)
    if plan.get('status') != 'ok':
        return plan
    names = tuple(plan['waypoints'][0])
    bounds = {j: observed['limits'][j] for j in names}
    envelope = prepare_execution(plan, hashes['robot_geometry.json'],
                                 max_speed={j: float(speed) for j in names}, bounds=bounds)
    if not lease_valid() or resource_identity(root) != hashes:
        raise ValueError('Lease or resources changed during planning')
    handle = save_plan(store, envelope, resource_hashes=hashes, lease_identity=token)
    summary = {key: value for key, value in plan.items() if key != 'waypoints'}
    return {**summary, 'waypoint_count': len(plan['waypoints']), 'execution_handle': handle,
            'planned_duration_s': envelope['payload']['waypoints'][-1]['time_s'],
            'max_joint_speed_deg_s': float(speed),
            'motion_commanded': False}


def execute_command(resources, bus, query):
    from robot_plan_store import consume_plan
    from trajectory_execution import execute_station_plan
    root, store, token = _context(resources)
    hashes = resource_identity(root)
    if not isinstance(query, dict):
        raise ValueError('Expected plan result or execution handle object')
    handle = query.get('execution_handle', query)
    envelope = consume_plan(store, handle, resource_hashes=hashes, lease_identity=token)
    budget = envelope['payload']['plan'].get('execution_tolerance_deg')
    if not isinstance(budget, dict) or not budget:
        raise ValueError('Plan lacks collision-reserved execution tolerance')
    def authorized():
        return lease_valid() and resource_identity(root) == hashes
    result = execute_station_plan(bus, envelope, lease_valid=authorized,
                                  profile_path=root/'robot_geometry.json',
                                  tracking_tolerance={j: min(9., value) for j, value in budget.items()},
                                  final_tolerance={j: min(.4, value) for j, value in budget.items()},
                                  timeout_s=65.)
    images = []
    for role in ('top', 'wrist'):
        try:
            images.append(resources.camera(role))
        except Exception as exc:
            result.setdefault('image_errors', {})[role] = str(exc)
    return {'images': images, **result}

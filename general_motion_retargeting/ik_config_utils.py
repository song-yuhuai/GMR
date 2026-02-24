import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_rot_offset(rot_offset):
    """
    Parse IK rotation offset from config into scipy Rotation.

    Supported formats:
    - Quaternion (wxyz): [w, x, y, z]
    - Euler XYZ in degrees: [rx_deg, ry_deg, rz_deg]
    - Dict forms:
      {"quat": [w, x, y, z]}
      {"euler_deg": [rx, ry, rz]}
      {"euler_rad": [rx, ry, rz]}
      {"euler": [rx, ry, rz]}  # treated as degrees
    """
    if isinstance(rot_offset, dict):
        if "quat" in rot_offset:
            q = np.asarray(rot_offset["quat"], dtype=float)
            if q.shape != (4,):
                raise ValueError(f"Invalid quat shape for rot_offset: {q.shape}")
            return R.from_quat(q, scalar_first=True)
        if "euler_deg" in rot_offset:
            e = np.asarray(rot_offset["euler_deg"], dtype=float)
            if e.shape != (3,):
                raise ValueError(f"Invalid euler_deg shape for rot_offset: {e.shape}")
            return R.from_euler("xyz", e, degrees=True)
        if "euler_rad" in rot_offset:
            e = np.asarray(rot_offset["euler_rad"], dtype=float)
            if e.shape != (3,):
                raise ValueError(f"Invalid euler_rad shape for rot_offset: {e.shape}")
            return R.from_euler("xyz", e, degrees=False)
        if "euler" in rot_offset:
            e = np.asarray(rot_offset["euler"], dtype=float)
            if e.shape != (3,):
                raise ValueError(f"Invalid euler shape for rot_offset: {e.shape}")
            return R.from_euler("xyz", e, degrees=True)
        raise ValueError(f"Unsupported rot_offset dict keys: {list(rot_offset.keys())}")

    arr = np.asarray(rot_offset, dtype=float)
    if arr.shape == (4,):
        return R.from_quat(arr, scalar_first=True)
    if arr.shape == (3,):
        return R.from_euler("xyz", arr, degrees=True)

    raise ValueError(
        f"Unsupported rot_offset format with shape {arr.shape}. "
        "Use quat [w,x,y,z] or euler_deg [rx,ry,rz]."
    )


def format_rot_offset_like(base_rot_offset, rot: R):
    """
    Format rotation offset `rot` using the same representation as `base_rot_offset`.
    Falls back to quaternion (wxyz) when unknown.
    """
    if isinstance(base_rot_offset, dict):
        if "quat" in base_rot_offset:
            return {"quat": rot.as_quat(scalar_first=True).tolist()}
        if "euler_deg" in base_rot_offset:
            return {"euler_deg": rot.as_euler("xyz", degrees=True).tolist()}
        if "euler_rad" in base_rot_offset:
            return {"euler_rad": rot.as_euler("xyz", degrees=False).tolist()}
        if "euler" in base_rot_offset:
            return {"euler": rot.as_euler("xyz", degrees=True).tolist()}

    arr = np.asarray(base_rot_offset, dtype=float)
    if arr.shape == (3,):
        return rot.as_euler("xyz", degrees=True).tolist()
    return rot.as_quat(scalar_first=True).tolist()

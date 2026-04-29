"""
step1_pose_estimation_v2.py
========================
Pose estimation of the Device (T-probe) using PnP algorithm.

Problem
-------
Given:
  - 3D coordinates of LED markers 1-10 and central target M in Device frame {D}
  - 2D pixel coordinates of visible LED markers in a camera image
  - 3D world coordinate of M from laser tracker: p_M_world = (xm, ym, zm)
Find:
  - Rotation R (3x3) and translation t (3,): Device frame -> World/Camera frame
  - 3D world coordinates of all 10 LED points and M

World coordinate system {W} coincides with Camera coordinate system {C}:
  Camera optical centre is at world origin (0,0,0).

Key equation:
  p_world = R @ p_device + t
  [u, v, 1]^T  ~  K @ p_world        (pinhole projection)

See Description.md for full mathematical derivation.

Usage
-----
  python step1_pose_estimation.py

Configuration switches are in the CONFIGURATION section below.
"""

import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # non-interactive backend, works on Windows without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (needed for 3d projection)
from scipy.spatial.transform import Rotation as SciRot
from typing import Optional, Tuple

# ==============================================================
# CONFIGURATION
# ==============================================================

# Output directory for saved figures.
# By default: the directory containing this script (e.g. D:\GitHub\drill_rotation).
# Change to any absolute path if needed, e.g.:
#   OUTPUT_DIR = Path(r'C:\my\output\folder')
OUTPUT_DIR: Path = Path(__file__).parent

# How to obtain the 2D image coordinate of point M:
#   True  -> (u_M, v_M) is given as a direct measurement (independent source)
#   False -> compute from 3D world coord p_M_world and camera matrix K
USE_GIVEN_M2D: bool = False

# Include M as an additional correspondence in the PnP solve:
USE_M_IN_PNP: bool = True

# Camera intrinsic matrix K  (from kabsch_example.py calibration)
K = np.array([
    [2931.000112931028,    0.0,              532.1829789852895],
    [   0.0,           2907.325320259926,    359.9799850555673],
    [   0.0,              0.0,                1.0             ]
], dtype=np.float64)

# Lens distortion coefficients  (zero = ideal pinhole; replace with real values)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

# Image dimensions (pixels) — derived from principal point
IMG_W = int(round(2.0 * K[0, 2]))   # ≈ 1064
IMG_H = int(round(2.0 * K[1, 2]))   # ≈ 720

# ==============================================================
# DEVICE GEOMETRY  (mm, Device frame {D})
# ==============================================================
# Axis convention (from documentation):
#   X_D: direction P1 -> P2
#   Y_D: direction P9 -> P1
#   Z_D: orthogonal to X_D Y_D, pointing AWAY from camera (into device body)
#        Z_D > 0 = farther from camera; Z_D < 0 = toward camera (e.g. P5, P6)
#        System is right-handed: X_D x Y_D = +Z_D  (confirmed in world frame)
#
# v2 change: M is the origin of the Device coordinate system.
# Raw coordinates from documentation are stored first; then P_M_RAW
# is subtracted from every point so that M = (0, 0, 0).
# Consequence: t (translation) in the PnP solution equals the
# world position of M, which is measured directly by the laser tracker.

# Raw coordinates from device documentation (mm)
_P_M_RAW = np.array([39.825, 114.750, 34.449], dtype=np.float64)

_DEVICE_POINTS_RAW: dict = {
    'M': np.array([ 39.825, 114.750, 34.449], dtype=np.float64),
    1:   np.array([  0.000, 226.575,  0.434], dtype=np.float64),
    2:   np.array([ 78.300, 226.575,  1.478], dtype=np.float64),
    3:   np.array([ 39.825, 186.300, 22.448], dtype=np.float64),
    4:   np.array([ 39.825, 157.050, 31.958], dtype=np.float64),
    5:   np.array([-11.475, 114.750, 57.395], dtype=np.float64),
    6:   np.array([ 88.425, 114.750, 56.180], dtype=np.float64),
    7:   np.array([ 39.825,  70.650, 32.060], dtype=np.float64),
    8:   np.array([ 39.825,  40.950, 22.928], dtype=np.float64),
    9:   np.array([  0.000,   0.000,  0.000], dtype=np.float64),
    10:  np.array([ 78.300,   0.000,  1.384], dtype=np.float64),
}

# M-centred Device coordinates: subtract P_M_RAW from every point,
# then negate Z so that the system is right-handed.
#
# Handedness analysis (world frame, OpenCV convention X=right, Y=down, Z=into scene):
#   X_D × Y_D  =  right × down  =  (0, 0, +1)  [into scene, same as world +Z]
#   Z_D must equal (0, 0, +1) for right-handed system.
#   → Z_D must point AWAY from camera (into device body).
#   → Negate the raw Z component (which pointed TOWARD the camera).
#
# Physical meaning after negation:
#   Z_D > 0  ⟹  point is deeper in the device body (farther from camera)
#   Z_D < 0  ⟹  point protrudes toward the camera (e.g. P5, P6)
DEVICE_POINTS: dict = {
    lbl: np.array([pt[0] - _P_M_RAW[0],
                   pt[1] - _P_M_RAW[1],
                 -(pt[2] - _P_M_RAW[2])])   # negate Z → right-handed
    for lbl, pt in _DEVICE_POINTS_RAW.items()
}

# P_M_BODY is now the zero vector (M is the Device origin)
P_M_BODY = DEVICE_POINTS['M']   # np.array([0., 0., 0.])

# Structural edges for visualisation (approximate device skeleton)
DEVICE_EDGES = [
    (1, 2), (1, 3), (2, 3),        # top pair -> upper spine
    (3, 4), (4, 5), (4, 6),        # spine -> lateral arms
    (5, 'M'), (6, 'M'),            # arms -> center
    ('M', 7), ('M', 8),            # center -> lower spine
    (7, 8), (8, 9), (8, 10),       # lower spine -> bottom pair
    (9, 10),
]

# ==============================================================
# HELPER: PROJECTION
# ==============================================================

def project_to_image(device_pts: dict, R: np.ndarray, t: np.ndarray,
                     K: np.ndarray, dist: np.ndarray) -> tuple[dict, dict]:
    """
    Project Device-frame points to 2D image coordinates.

    Parameters
    ----------
    device_pts : dict {label: (3,) array}
    R          : (3,3) rotation matrix Device->World
    t          : (3,)  translation Device->World  (mm)
    K          : (3,3) camera intrinsic matrix
    dist       : (5,1) distortion coefficients

    Returns
    -------
    proj2d     : dict {label: (2,) pixel array}
    world3d    : dict {label: (3,) world-coord array}
    """
    labels = list(device_pts.keys())
    pts3d = np.array([device_pts[l] for l in labels], dtype=np.float64)

    rvec, _ = cv2.Rodrigues(R)
    pts2d_cv, _ = cv2.projectPoints(pts3d, rvec, t.reshape(3, 1), K, dist)
    pts2d_cv = pts2d_cv.squeeze()   # (N, 2)

    proj2d  = {l: pts2d_cv[i]             for i, l in enumerate(labels)}
    world3d = {l: R @ device_pts[l] + t   for l in labels}
    return proj2d, world3d


def filter_visible(proj2d: dict, world3d: dict,
                   w: int, h: int) -> dict:
    """
    Keep only points that lie in front of the camera AND inside the image frame.

    Returns dict of visible projected points.
    """
    visible = {}
    for label, p2d in proj2d.items():
        pw = world3d[label]
        u, v = p2d
        if pw[2] > 0 and (0 <= u < w) and (0 <= v < h):
            visible[label] = p2d
    return visible


def compute_m2d(p_m_world: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute 2D pixel coordinate of M from its 3D world position and camera K.

    u = fx * X/Z + cx,   v = fy * Y/Z + cy
    """
    X, Y, Z = p_m_world
    u = K[0, 0] * X / Z + K[0, 2]
    v = K[1, 1] * Y / Z + K[1, 2]
    return np.array([u, v], dtype=np.float64)


# ==============================================================
# HELPER: PnP SOLVE
# ==============================================================

def solve_pnp(visible_led_2d: dict,
              device_pts: dict,
              K: np.ndarray,
              dist: np.ndarray,
              use_m_in_pnp: bool = True,
              m_2d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Recover pose (R, t) from 2D–3D correspondences via PnP.

    Only LED points 1-10 are used by default; M is added when use_m_in_pnp=True.

    Parameters
    ----------
    visible_led_2d : dict {label: (2,) pixel}  — visible LED points (1-10)
    device_pts     : dict {label: (3,) Device-frame coords}
    K, dist        : camera parameters
    use_m_in_pnp   : add M as extra PnP correspondence
    m_2d           : 2D pixel coord of M (required when use_m_in_pnp=True)

    Returns
    -------
    R_calc  : (3,3) rotation matrix
    t_calc  : (3,)  translation vector
    success : bool
    """
    obj_pts = []
    img_pts = []

    for label in visible_led_2d:
        if label == 'M':
            continue
        obj_pts.append(device_pts[label])
        img_pts.append(visible_led_2d[label])

    if use_m_in_pnp and m_2d is not None:
        obj_pts.append(device_pts['M'])
        img_pts.append(m_2d)

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    n = len(obj_pts)
    if n < 4:
        raise ValueError(f"PnP requires ≥4 correspondences, got {n}.")

    # Step 1: EPnP — good closed-form for n>4
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                   flags=cv2.SOLVEPNP_EPNP)
    # Step 2: Levenberg-Marquardt refinement (warm-started from EPnP)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                   rvec, tvec,
                                   useExtrinsicGuess=True,
                                   flags=cv2.SOLVEPNP_ITERATIVE)

    R_calc, _ = cv2.Rodrigues(rvec)
    t_calc    = tvec.flatten()
    return R_calc, t_calc, bool(ok)


# ==============================================================
# HELPER: ERRORS AND VERIFICATION
# ==============================================================

def pose_errors(R_gt: np.ndarray, t_gt: np.ndarray,
                R_calc: np.ndarray, t_calc: np.ndarray) -> tuple[float, float]:
    """
    Return (angular_error_deg, translation_error_mm).

    Angular error = rotation angle of  R_calc @ R_gt^T  (geodesic distance on SO(3)).
    """
    R_rel = R_calc @ R_gt.T
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    ang_err = float(np.degrees(np.linalg.norm(rvec_rel)))
    t_err   = float(np.linalg.norm(t_calc - t_gt))
    return ang_err, t_err


def reprojection_errors(visible_2d: dict, device_pts: dict,
                        R: np.ndarray, t: np.ndarray,
                        K: np.ndarray, dist: np.ndarray) -> dict:
    """
    Compute per-point reprojection error (pixels) for a solved pose.
    """
    labels  = list(visible_2d.keys())
    pts3d   = np.array([device_pts[l] for l in labels], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    reproj, _ = cv2.projectPoints(pts3d, rvec, t.reshape(3, 1), K, dist)
    reproj = reproj.squeeze()

    errors = {}
    for i, l in enumerate(labels):
        obs = visible_2d[l]
        errors[l] = float(np.linalg.norm(obs - reproj[i]))
    return errors


def recover_world_coords(device_pts: dict,
                         R: np.ndarray, t: np.ndarray) -> dict:
    """Apply R, t to all Device-frame points -> World coords."""
    return {l: R @ p + t for l, p in device_pts.items()}


# ==============================================================
# VISUALISATION: CAMERA IMAGE (2D)
# ==============================================================

def visualize_image(visible_2d: dict, all_proj2d: dict,
                    R_gt, t_gt, R_calc, t_calc,
                    img_w: int, img_h: int,
                    title: str = "Simulated Camera Image") -> plt.Figure:
    """
    Draw the simulated camera image with all projected Device points.

    - Green circles  : LED markers used in PnP
    - Red star       : M (central target sphere)
    - Gray circles   : points outside the frame (shown at clipped position, dimmed)
    - Dashed lines   : device skeleton
    """
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    # Image boundary
    ax.add_patch(plt.Rectangle((0, 0), img_w, img_h,
                                fill=False, edgecolor='#888888', lw=1.5))

    # ---- skeleton (draw first, below markers) ----
    for a, b in DEVICE_EDGES:
        if a in visible_2d and b in visible_2d:
            pa, pb = visible_2d[a], visible_2d[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                    color='#223344', lw=0.9, zorder=1)

    # ---- markers ----
    for label, p2d in all_proj2d.items():
        u, v = p2d
        in_frame = label in visible_2d

        if label == 'M':
            color  = '#ff4455'
            marker = '*'
            msize  = 14
            zorder = 6
        else:
            color  = '#33ee88' if in_frame else '#445555'
            marker = 'o'
            msize  = 9
            zorder = 5 if in_frame else 3

        ax.plot(u, v, marker, color=color, markersize=msize,
                markeredgecolor='white', markeredgewidth=0.5, zorder=zorder)

        if in_frame:
            # label
            ax.annotate(str(label), (u, v), xytext=(7, 5),
                        textcoords='offset points',
                        color='white', fontsize=9, fontweight='bold', zorder=7)
            # pixel coords in small text
            ax.annotate(f'({u:.0f}, {v:.0f})', (u, v), xytext=(7, -10),
                        textcoords='offset points',
                        color='#aaaaaa', fontsize=7, zorder=7)

    # ---- axes ----
    ax.set_xlim(-30, img_w + 30)
    ax.set_ylim(img_h + 30, -30)   # invert Y (image convention)
    ax.set_xlabel('u  (pixels)', color='white', fontsize=11)
    ax.set_ylabel('v  (pixels)', color='white', fontsize=11)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    # ---- error info box ----
    ang_err, t_err = pose_errors(R_gt, t_gt, R_calc, t_calc)
    info = (f"Visible points : {len(visible_2d)}\n"
            f"Image size     : {img_w}×{img_h} px\n"
            f"Angular error  : {ang_err:.5f}°\n"
            f"Translation err: {t_err:.5f} mm")
    ax.text(0.01, 0.01, info, transform=ax.transAxes,
            color='white', fontsize=9, va='bottom', family='monospace',
            bbox=dict(boxstyle='round', fc='#111133', alpha=0.85))

    # ---- legend ----
    legend_handles = [
        mpatches.Patch(color='#33ee88', label='LED (visible, in PnP)'),
        mpatches.Patch(color='#ff4455', label='M — central target sphere'),
        mpatches.Patch(color='#445555', label='Point outside frame'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              facecolor='#111133', edgecolor='#555555',
              labelcolor='white', fontsize=9)

    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=8)
    plt.tight_layout()
    return fig


# ==============================================================
# VISUALISATION: 3D SCENE
# ==============================================================

def _draw_device_3d(ax, device_pts: dict, R: np.ndarray, t: np.ndarray,
                    color: str, alpha: float = 1.0, label_pts: bool = True):
    """Draw Device points and local coordinate axes in a 3D axes object."""
    world = {l: R @ p + t for l, p in device_pts.items()}

    for label, pw in world.items():
        if label == 'M':
            ax.scatter(*pw.reshape(1, -1).T, c=color, s=60, marker='*',
                       alpha=alpha, zorder=5)
        else:
            ax.scatter(*pw.reshape(1, -1).T, c=color, s=18, marker='o',
                       alpha=alpha)
        if label_pts and alpha > 0.4:
            ax.text(pw[0], pw[1], pw[2], f' {label}',
                    fontsize=6, color=color, alpha=alpha)

    # Device coordinate axes (length = 50 mm)
    axis_len = 50.0
    for i, col in enumerate(['red', 'lime', 'dodgerblue']):
        d = R[:, i] * axis_len
        ax.quiver(*t, *d, color=col, alpha=alpha * 0.85,
                  arrow_length_ratio=0.25, linewidth=1.5)


def visualize_3d(device_pts: dict,
                 R_gt: np.ndarray, t_gt: np.ndarray,
                 R_calc: Optional[np.ndarray] = None,
                 t_calc: Optional[np.ndarray] = None) -> plt.Figure:
    """
    3D overview: camera + Device in World coordinate system.
    Ground-truth pose in green; calculated pose in orange (if provided).
    """
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor('#0d0d1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d0d1a')

    # Ground-truth device
    _draw_device_3d(ax, device_pts, R_gt, t_gt, '#33ee88', alpha=1.0)

    # Calculated device
    if R_calc is not None and t_calc is not None:
        _draw_device_3d(ax, device_pts, R_calc, t_calc,
                        '#ffaa33', alpha=0.6, label_pts=False)

    # Camera at origin
    ax.scatter([0], [0], [0], c='white', s=120, marker='P', zorder=10)
    ax.text(0, 0, 0, '  Camera\n  (World origin)', color='white', fontsize=8)

    # World axes at camera
    cam_len = 120.0
    for i, (col, lbl) in enumerate(zip(['red','lime','dodgerblue'],
                                        ['X_W', 'Y_W', 'Z_W'])):
        d = np.zeros(3); d[i] = cam_len
        ax.quiver(0, 0, 0, *d, color=col, arrow_length_ratio=0.25, linewidth=2)
        ax.text(*(d * 1.12), lbl, color=col, fontsize=9)

    # Labels and style
    ax.set_xlabel('X (mm)', color='white', labelpad=6)
    ax.set_ylabel('Y (mm)', color='white', labelpad=6)
    ax.set_zlabel('Z (mm)', color='white', labelpad=6)
    ax.tick_params(colors='#888888', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    legend_handles = [
        mpatches.Patch(color='#33ee88', label='Ground-truth pose'),
    ]
    if R_calc is not None:
        legend_handles.append(
            mpatches.Patch(color='#ffaa33', label='Calculated pose'))
    ax.legend(handles=legend_handles, facecolor='#111133',
              edgecolor='#555555', labelcolor='white', fontsize=9)

    ax.set_title('3D Scene — Camera + Device', color='white',
                 fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


# ==============================================================
# PRINT HELPERS
# ==============================================================

def print_section(title: str):
    print()
    print('=' * 62)
    print(f'  {title}')
    print('=' * 62)


def print_matrix(M: np.ndarray, name: str, indent: int = 4):
    pad = ' ' * indent
    print(f'{pad}{name}:')
    for row in M:
        print(pad + '  [' + '  '.join(f'{v:10.5f}' for v in row) + ']')


# ==============================================================
# MAIN TEST
# ==============================================================

def main():
    print_section('STEP 1 — POSE ESTIMATION TEST')

    # ----------------------------------------------------------
    # 1. Ground-truth pose
    # ----------------------------------------------------------
    # Device rotated by (pitch=15°, yaw=20°, roll=5°) in XYZ Euler convention.
    # t_gt is now the WORLD POSITION OF M (Device origin in v2),
    # measured by the laser tracker in a real scenario.
    angles_gt_deg = np.array([15.0, 20.0, 5.0])   # pitch, yaw, roll (X,Y,Z)
    R_gt = SciRot.from_euler('xyz', np.radians(angles_gt_deg)).as_matrix()
    t_gt = np.array([50.0, 100.0, 2000.0], dtype=np.float64)   # mm — world pos of M

    print(f'\nGround-truth rotation  (XYZ Euler): {angles_gt_deg} deg')
    print(f'Ground-truth translation           : {t_gt} mm')
    print_matrix(R_gt, 'R_gt')

    # 2. Simulate camera image (forward projection)
    # ----------------------------------------------------------
    proj2d_all, world3d_gt = project_to_image(DEVICE_POINTS, R_gt, t_gt,
                                              K, DIST_COEFFS)
    visible_2d = filter_visible(proj2d_all, world3d_gt, IMG_W, IMG_H)
    print(f'\nImage resolution: {IMG_W} × {IMG_H} px')

    def sort_key(k):
        return (0, 0) if k == 'M' else (1, k)

    vis_sorted = sorted(visible_2d.keys(), key=sort_key)
    print(f'Visible points  : {vis_sorted}  '
          f'({len(visible_2d)} of {len(DEVICE_POINTS)})')
    print(f'\n{"Point":>6} {"u (px)":>9} {"v (px)":>9}'
          f'  {"X_world":>9} {"Y_world":>9} {"Z_world":>9}')
    print('-' * 62)
    for label in ['M'] + list(range(1, 11)):
        flag = '✓' if label in visible_2d else ' '
        p2d = proj2d_all[label]
        pw = world3d_gt[label]
        print(f'  {flag} {str(label):>4}  {p2d[0]:9.1f} {p2d[1]:9.1f}'
              f'  {pw[0]:9.2f} {pw[1]:9.2f} {pw[2]:9.2f}')
    # ----------------------------------------------------------
    # 3. Prepare M 2D coordinate
    # ----------------------------------------------------------
    # In v2: t_gt IS the world position of M (since M = Device origin).
    # In a real scenario this comes from the laser tracker.
    p_m_world = world3d_gt['M']  # == R_gt @ (0,0,0) + t_gt == t_gt
    if USE_GIVEN_M2D:
        # In a real scenario, this is measured independently
        if 'M' in visible_2d:
            m_2d = visible_2d['M'].copy()
        else:
            # M not visible — cannot proceed in this branch
            print('\nWARNING: M not visible in image; USE_GIVEN_M2D=True requires'
                  ' a visible M.  Falling back to computed.')
            m_2d = compute_m2d(p_m_world, K)
    else:
        # Compute from 3D world coord (from laser tracker) + camera K
        m_2d = compute_m2d(p_m_world, K)
    print(f'\nM 2D coordinate ({"given" if USE_GIVEN_M2D else "computed from tracker"}):'
          f'  ({m_2d[0]:.2f}, {m_2d[1]:.2f}) px')
    print(f'M world 3D (from tracker): ({p_m_world[0]:.3f}, '
          f'{p_m_world[1]:.3f}, {p_m_world[2]:.3f}) mm')

    # ----------------------------------------------------------
    # 4. Solve PnP
    # ----------------------------------------------------------
    # Exclude M from visible_led_2d (it is handled separately via use_m_in_pnp)
    visible_led_2d = {l: v for l, v in visible_2d.items() if l != 'M'}
    n_pnp = len(visible_led_2d) + (1 if USE_M_IN_PNP else 0)

    print(f'\nSolving PnP with {n_pnp} correspondences'
          f' (LEDs={len(visible_led_2d)}'
          f' + M={1 if USE_M_IN_PNP else 0})...')

    R_calc, t_calc, pnp_ok = solve_pnp(
        visible_led_2d, DEVICE_POINTS, K, DIST_COEFFS,
        use_m_in_pnp=USE_M_IN_PNP,
        m_2d=m_2d
    )
    print(f'PnP converged: {pnp_ok}')
    print_matrix(R_calc, 'R_calc')
    print(f'    t_calc: {t_calc}')

    # ----------------------------------------------------------
    # 5. Comparison and verification
    # ----------------------------------------------------------
    ang_err, t_err = pose_errors(R_gt, t_gt, R_calc, t_calc)

    euler_gt   = SciRot.from_matrix(R_gt).as_euler('xyz', degrees=True)
    euler_calc = SciRot.from_matrix(R_calc).as_euler('xyz', degrees=True)

    print_section('RESULTS')
    print(f'  GT   Euler XYZ (deg): {euler_gt.round(4)}')
    print(f'  Calc Euler XYZ (deg): {euler_calc.round(4)}')
    print(f'  GT   translation (mm): {t_gt}')
    print(f'  Calc translation (mm): {t_calc.round(4)}')
    print(f'\n  Angular error     : {ang_err:.6f} deg')
    print(f'  Translation error : {t_err:.6f} mm')

    # Rotation matrix validity
    ortho_err = np.linalg.norm(R_calc @ R_calc.T - np.eye(3))
    det_err   = abs(np.linalg.det(R_calc) - 1.0)
    print(f'\n  ||R^T R - I||  : {ortho_err:.2e}   (should be ~0)')
    print(f'  |det(R) - 1|   : {det_err:.2e}    (should be ~0)')

    # M consistency check
    m_reconstructed = R_calc @ P_M_BODY + t_calc
    m_consistency   = np.linalg.norm(m_reconstructed - p_m_world)
    print(f'\n  M reconstruction error: {m_consistency:.6f} mm'
          f'  (should be < 0.1 mm)')

    # Reprojection errors
    reproj_errs = reprojection_errors(
        {l: v for l, v in visible_2d.items()},
        DEVICE_POINTS, R_calc, t_calc, K, DIST_COEFFS
    )
    max_repr  = max(reproj_errs.values())
    mean_repr = np.mean(list(reproj_errs.values()))
    print(f'\n  Reprojection error — mean: {mean_repr:.4f} px,  '
          f'max: {max_repr:.4f} px')

    # ----------------------------------------------------------
    # 6. Recover all world coordinates
    # ----------------------------------------------------------
    world3d_calc = recover_world_coords(DEVICE_POINTS, R_calc, t_calc)

    print_section('RECOVERED 3D WORLD COORDINATES')
    hdr = f'{"Label":>6}  {"GT X":>9} {"GT Y":>9} {"GT Z":>9}  ' \
          f'{"Calc X":>9} {"Calc Y":>9} {"Calc Z":>9}  {"Err mm":>8}'
    print('  ' + hdr)
    print('  ' + '-' * len(hdr))
    for label in ['M'] + list(range(1, 11)):
        pg = world3d_gt[label]
        pc = world3d_calc[label]
        e  = np.linalg.norm(pg - pc)
        print(f'  {str(label):>6}'
              f'  {pg[0]:9.3f} {pg[1]:9.3f} {pg[2]:9.3f}'
              f'  {pc[0]:9.3f} {pc[1]:9.3f} {pc[2]:9.3f}'
              f'  {e:8.5f}')

    # ----------------------------------------------------------
    # 7. Visualise
    # ----------------------------------------------------------
    print_section('SAVING VISUALISATIONS')

    fig_img = visualize_image(
        visible_2d, proj2d_all,
        R_gt, t_gt, R_calc, t_calc,
        IMG_W, IMG_H,
        title=(f'Simulated Camera Image  '
               f'[ang_err={ang_err:.4f}°, t_err={t_err:.4f} mm]')
    )
    out_img = OUTPUT_DIR / 'camera_image.png'
    fig_img.savefig(out_img, dpi=150,
                    bbox_inches='tight', facecolor=fig_img.get_facecolor())
    print(f'  Saved: {out_img}')

    fig_3d = visualize_3d(DEVICE_POINTS, R_gt, t_gt, R_calc, t_calc)
    out_3d = OUTPUT_DIR / 'scene_3d.png'
    fig_3d.savefig(out_3d, dpi=150,
                   bbox_inches='tight', facecolor=fig_3d.get_facecolor())
    print(f'  Saved: {out_3d}')

    plt.close('all')
    print('\nDone.')


if __name__ == '__main__':
    main()

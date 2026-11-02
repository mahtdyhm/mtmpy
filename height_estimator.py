#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
height_estimator.py - Monocular true-height estimation (single RGB camera)

Pipeline
  1. Intrinsics K + distortion  -> undistort (barrel / fisheye-ish wide lens)
  2. Detect reference (A4 sheet or ISO/IEC 7810 ID-1 card) lying on the FLOOR
     -> solvePnP (IPPE) -> floor plane in camera coordinates
  3. Pose (MediaPipe, 33 landmarks + metric-ish world landmarks)
  4. Per-keypoint metric depth:  Z_i = Z_floor + (zw_i - zw_feet)
     3D point P_i = Z_i * K^-1 [u v 1]^T   (this is exactly the foreshortening fix)
  5. Kinematic chain:  H = (head + torso + femur + tibia) / (1 - 0.039)
  6. Robust fusion over frames (median / MAD), annotated output

Usage
  python height_estimator.py calibrate --images "calib/*.jpg" --pattern 9x6 --square 0.025 --out camera.npz
  python height_estimator.py measure --input person.jpg --camera camera.npz --ref a4
  python height_estimator.py measure --input walk.mp4 --focal35 26 --ref card --frame-step 3
  python height_estimator.py fit --csv subjects.csv --out correction.json
  python height_estimator.py selftest

Requirements: pip install opencv-python numpy mediapipe==0.10.14
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
REFS = {"a4": (0.297, 0.210), "card": (0.0856, 0.05398)}  # metres (long, short)

ANKLE_HEIGHT_RATIO = 0.039      # ankle-joint height / stature (Drillis & Contini)
EYE_TO_VERTEX_M = 0.115         # eye centre -> top of head (adult mean, +-1 cm). Tune per population.
# Height of each foot landmark above the floor (m). The scale error this causes is ~ dh / camera_height, so
# keep the phone high (>= 1.5 m) and pitched down 15-25 deg. Tune on your own data.
FOOT_LANDMARK_HEIGHT = {29: 0.02, 30: 0.02, 31: 0.01, 32: 0.01, 27: 0.07, 28: 0.07}
DEPTH_SIGN = 1.0               # MediaPipe world z: +1 = larger z is farther from camera. Flip if needed.

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
L_KNEE, R_KNEE, L_ANK, R_ANK = 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32
L_EYE, R_EYE, MOUTH_L, MOUTH_R = 2, 5, 9, 10

EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (27, 31),
         (28, 30), (30, 32), (28, 32), (0, 2), (0, 5), (2, 5), (9, 10)]


# --------------------------------------------------------------------------
# 1. Camera intrinsics and lens distortion
# --------------------------------------------------------------------------
@dataclass
class Camera:
    K: np.ndarray
    dist: np.ndarray
    source: str


def camera_from_file(path: str, size: tuple[int, int]) -> Camera:
    d = np.load(path)
    K, dist = d["K"].astype(np.float64), d["dist"].astype(np.float64).ravel()
    w0, h0 = [int(v) for v in d["size"]]
    if (w0, h0) != tuple(size):          # same FOV, different resolution
        s = size[0] / w0
        K = K.copy()
        K[0, :] *= s
        K[1, :] *= s
    return Camera(K, dist, f"file:{path}")


def camera_from_focal35(f35: float, size: tuple[int, int]) -> Camera:
    """Approximate K from the 35 mm-equivalent focal length (diagonal-based). Uncalibrated: expect 2-5 % scale error."""
    w, h = size
    f = f35 * math.hypot(w, h) / 43.267
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    return Camera(K, np.zeros(5), f"focal35={f35}")


class Undistorter:
    def __init__(self, cam: Camera, size: tuple[int, int]):
        w, h = size
        if np.allclose(cam.dist, 0):
            self.K, self.maps = cam.K.copy(), None
            return
        newK, _ = cv2.getOptimalNewCameraMatrix(cam.K, cam.dist, (w, h), 0, (w, h))
        self.K = newK
        self.maps = cv2.initUndistortRectifyMap(cam.K, cam.dist, None, newK, (w, h), cv2.CV_16SC2)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img if self.maps is None else cv2.remap(img, self.maps[0], self.maps[1], cv2.INTER_LINEAR)


def calibrate_chessboard(pattern_glob: str, pattern=(9, 6), square_m=0.025, out="camera.npz") -> None:
    """Zhang calibration. Take 15-30 photos of a printed chessboard at varied tilts/positions, same phone & zoom."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    objpts, imgpts, size = [], [], None
    for p in sorted(glob.glob(pattern_glob)):
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            objpts.append(objp)
            imgpts.append(corners.reshape(-1, 1, 2))
    if len(objpts) < 8:
        sys.exit(f"Only {len(objpts)} usable calibration images (need >= 8, ideally 20+).")
    rms, K, dist, _, _ = cv2.calibrateCamera(objpts, imgpts, size, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    np.savez(out, K=K, dist=dist, size=np.array(size), rms=rms)
    print(f"Calibration done: RMS reprojection error = {rms:.3f} px (aim < 0.5), K=\n{K}\nsaved -> {out}")


# --------------------------------------------------------------------------
# 2. Reference object detection -> floor plane
# --------------------------------------------------------------------------
@dataclass
class Reference:
    kind: str
    corners: np.ndarray      # 4x2 image points
    rvec: np.ndarray
    tvec: np.ndarray
    err: float               # RMS reprojection error, px
    area: float

    @property
    def plane(self) -> tuple[np.ndarray, np.ndarray]:
        R, _ = cv2.Rodrigues(self.rvec)
        return R[:, 2].copy(), self.tvec.reshape(3).copy()   # normal, point (camera coords)


def order_quad(p: np.ndarray) -> np.ndarray:
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], np.float32)  # TL TR BR BL


def _quad_candidates(gray: np.ndarray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 40, 120), np.ones((3, 3), np.uint8))
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ada = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5)
    h, w = gray.shape
    lo, hi = 4e-4 * w * h, 0.5 * w * h
    for m in (edges, otsu, 255 - otsu, ada):
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < lo or a > hi:
                continue
            ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                yield ap.reshape(4, 2).astype(np.float32)


def _looks_like_paper(gray: np.ndarray, quad: np.ndarray) -> bool:
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    if inner.sum() == 0:
        return False
    m, s = cv2.meanStdDev(gray, mask=inner)
    # آستانه منعطف‌تر برای روشنایی و سایه‌های محیطی در عکس‌های گوشی
    return m[0, 0] > 70 and s[0, 0] < 90


def _obj_pts(L: float, S: float, orient: int) -> np.ndarray:
    if orient == 0:
        return np.array([[0, 0, 0], [L, 0, 0], [L, S, 0], [0, S, 0]], np.float64)
    return np.array([[0, 0, 0], [S, 0, 0], [S, L, 0], [0, L, 0]], np.float64)


def detect_reference(bgr: np.ndarray, K: np.ndarray, kind: str = "auto", roi=None,
                     max_reproj: float = 3.0) -> Optional[Reference]:
    """Find the A4 sheet / ID-1 card (image must already be undistorted). roi=(x,y,w,h) strongly recommended."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    x0 = y0 = 0
    sub = gray
    if roi:
        x0, y0, rw, rh = roi
        sub = gray[y0:y0 + rh, x0:x0 + rw]
    kinds = list(REFS) if kind == "auto" else [kind]
    found: list[Reference] = []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    for q in _quad_candidates(sub):
        q = order_quad(q + np.array([x0, y0], np.float32))
        q = cv2.cornerSubPix(gray, q.reshape(-1, 1, 2), (7, 7), (-1, -1), crit).reshape(4, 2)
        area = cv2.contourArea(q)
        for k in kinds:
            if k == "a4" and not _looks_like_paper(gray, q):
                continue
            L, S = REFS[k]
            for orient in (0, 1):
                obj = _obj_pts(L, S, orient)
                try:
                    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, q.astype(np.float64), K, None,
                                                             flags=cv2.SOLVEPNP_IPPE)
                except cv2.error:
                    continue
                for rv, tv in zip(rvecs, tvecs):
                    if tv[2, 0] <= 0:
                        continue
                    proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
                    err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - q) ** 2, axis=1))))
                    if err < max_reproj:
                        found.append(Reference(k, q, rv, tv, err, area))
    if not found:
        return None
    emin = min(r.err for r in found)
    good = [r for r in found if r.err <= emin + 0.5]
    return max(good, key=lambda r: r.area)   # nested quads (inner print frames) fit too -> take the outer one


def floor_up_normal(n: np.ndarray) -> np.ndarray:
    """Floor normal pointing 'up'. Camera y axis points down, so up has a negative y component
    (assumes the phone is roughly upright / landscape, not pointing at the floor or ceiling)."""
    return n if n[1] < 0 else -n


def ray_depth_on_plane(uv, K, n, p0, offset: float = 0.0) -> Optional[float]:
    """Camera-frame depth Z where the pixel ray meets the plane lying `offset` metres ABOVE the floor.
    offset accounts for landmarks (heel, toe, ankle) that are not exactly on the floor."""
    nu = floor_up_normal(n)
    d = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
    den = float(nu @ d)
    if abs(den) < 1e-6:
        return None
    lam = (float(nu @ p0) + offset) / den
    return lam if lam > 0 else None        # d_z = 1  ->  lam is the camera-frame depth Z


# --------------------------------------------------------------------------
# 3. Pose backend
# --------------------------------------------------------------------------
@dataclass
class PoseResult:
    px: np.ndarray      # (33,2) pixels
    vis: np.ndarray     # (33,)
    world: np.ndarray   # (33,3) metres, hip-centred


class MediaPipeBackend:
    def __init__(self, complexity: int = 1, prefer_yolo: bool = False):
        self.backend = None
        self.yolo_model = None

        if not prefer_yolo:
            try:
                import mediapipe as mp
                try:
                    self.backend = mp.solutions.pose.Pose(
                        static_image_mode=True, model_complexity=complexity, min_detection_confidence=0.5
                    )
                except AttributeError:
                    from mediapipe.python.solutions import pose as mp_pose
                    self.backend = mp_pose.Pose(
                        static_image_mode=True, model_complexity=complexity, min_detection_confidence=0.5
                    )
            except Exception:
                self.backend = None

        if self.backend is None:
            try:
                from ultralytics import YOLO
                m_path = Path("yolo11n-pose.pt")
                if not m_path.exists():
                    m_path = Path(__file__).resolve().parent / "yolo11n-pose.pt"
                self.yolo_model = YOLO(str(m_path))
                print("[info] MediaPipe not available: using YOLO11-Pose backend.")
            except Exception as e:
                print(f"[warn] YOLO fallback failed: {e}")

    def infer(self, bgr: np.ndarray) -> Optional[PoseResult]:
        h, w = bgr.shape[:2]

        # ۱. تلاش با مدیاپایپ در صورت در دسترس بودن
        if self.backend is not None:
            try:
                res = self.backend.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks and res.pose_world_landmarks:
                    lm = res.pose_landmarks.landmark
                    px = np.array([[l.x * w, l.y * h] for l in lm])
                    vis = np.array([l.visibility for l in lm])
                    world = np.array([[l.x, l.y, l.z] for l in res.pose_world_landmarks.landmark])
                    return PoseResult(px, vis, world)
            except Exception:
                pass

        # ۲. فال‌بک خودکار به YOLO11-Pose
        if self.yolo_model is not None:
            results = self.yolo_model(bgr, verbose=False)
            if not results or len(results[0].keypoints) == 0:
                return None
            kpts = results[0].keypoints[0]
            xy = kpts.xy.cpu().numpy()[0]  # (17, 2)
            conf = kpts.conf.cpu().numpy()[0] if kpts.conf is not None else np.ones(17)

            if len(xy) < 17:
                return None

            # ساخت مپ ۳۳ تایی استاندارد برای سازگاری با زنجیره کینماتیک
            px = np.zeros((33, 2), dtype=np.float32)
            vis = np.zeros(33, dtype=np.float32)
            world = np.zeros((33, 3), dtype=np.float32)

            coco_to_mp = {
                0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
                5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
                11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28
            }
            for c_idx, mp_idx in coco_to_mp.items():
                px[mp_idx] = xy[c_idx]
                vis[mp_idx] = float(conf[c_idx])

            # تخمین موقعیت دهان بین بینی و شانه‌ها
            px[9] = px[0] + [-10, 15]
            px[10] = px[0] + [10, 15]
            vis[9] = vis[10] = vis[0]

            # تخمین پاشنه و پنجه پا دقیقا زیر مچ پا
            px[29] = px[27] + [0, 6]
            px[31] = px[27] + [0, 10]
            vis[29] = vis[31] = vis[27]

            px[30] = px[28] + [0, 6]
            px[32] = px[28] + [0, 10]
            vis[30] = vis[32] = vis[28]

            return PoseResult(px, vis, world)

        return None


# --------------------------------------------------------------------------
# 4/5. 3D lifting + kinematic chain
# --------------------------------------------------------------------------
@dataclass
class Measurement:
    height_m: float
    segments: dict          # raw lengths (m): head, torso, femur, tibia
    pts: dict               # named 3D points (camera frame)
    P: np.ndarray           # (33,3)
    z_floor: float


def load_correction(path: Optional[str]) -> dict:
    base = {"gains": {"head": 1.0, "torso": 1.0, "femur": 1.0, "tibia": 1.0}, "bias_m": 0.0}
    if path:
        base.update(json.load(open(path)))
    return base


def lift_to_3d(pose: PoseResult, K, n, p0, min_vis=0.5, depth_scale=1.0):
    feet = [i for i in (L_HEEL, R_HEEL, L_TOE, R_TOE) if pose.vis[i] >= min_vis]
    if not feet:
        feet = [i for i in (L_ANK, R_ANK) if pose.vis[i] >= min_vis]
    if not feet:
        return None
    zs = [ray_depth_on_plane(pose.px[i], K, n, p0, FOOT_LANDMARK_HEIGHT[i]) for i in feet]
    zs = [z for z in zs if z is not None]
    if not zs:
        return None
    z_floor = float(np.mean(zs))
    zw_feet = float(np.mean(pose.world[feet, 2]))
    Z = z_floor + DEPTH_SIGN * depth_scale * (pose.world[:, 2] - zw_feet)
    if np.any(Z <= 0.2):
        return None
    P = np.c_[(pose.px[:, 0] - K[0, 2]) / K[0, 0] * Z, (pose.px[:, 1] - K[1, 2]) / K[1, 1] * Z, Z]
    return P, z_floor


def measure(pose: PoseResult, K, ref: Reference, corr: dict, min_vis=0.5, depth_scale=1.0) -> Optional[Measurement]:
    n, p0 = ref.plane
    lifted = lift_to_3d(pose, K, n, p0, min_vis, depth_scale)
    if lifted is None:
        return None
    P, z_floor = lifted
    v = pose.vis
    need = [L_SH, R_SH, L_HIP, R_HIP, L_EYE, R_EYE, MOUTH_L, MOUTH_R]
    if any(v[i] < min_vis for i in need):
        return None

    mid_sh, mid_hip = (P[L_SH] + P[R_SH]) / 2, (P[L_HIP] + P[R_HIP]) / 2
    eyes, mouth = (P[L_EYE] + P[R_EYE]) / 2, (P[MOUTH_L] + P[MOUTH_R]) / 2
    up = eyes - mouth
    up /= np.linalg.norm(up) + 1e-9
    vertex = eyes + up * EYE_TO_VERTEX_M

    def side_mean(pairs):
        ls = [np.linalg.norm(P[a] - P[b]) for a, b in pairs if v[a] >= min_vis and v[b] >= min_vis]
        return float(np.mean(ls)) if ls else None

    fem = side_mean([(L_HIP, L_KNEE), (R_HIP, R_KNEE)])
    tib = side_mean([(L_KNEE, L_ANK), (R_KNEE, R_ANK)])
    if fem is None or tib is None:
        return None
    seg = {"head": float(np.linalg.norm(vertex - mid_sh)),
           "torso": float(np.linalg.norm(mid_sh - mid_hip)), "femur": fem, "tibia": tib}
    g = corr["gains"]
    S = sum(g[k] * seg[k] for k in seg) + corr["bias_m"]
    H = S / (1.0 - ANKLE_HEIGHT_RATIO)
    return Measurement(H, seg, {"vertex": vertex, "mid_sh": mid_sh, "mid_hip": mid_hip}, P, z_floor)


def fuse(heights: list[float]) -> dict:
    a = np.array(heights)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {"height_m": med, "robust_sigma_m": 1.4826 * mad, "n_frames": int(len(a))}


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def _proj(p, K):
    return int(K[0, 0] * p[0] / p[2] + K[0, 2]), int(K[1, 0 + 1] * p[1] / p[2] + K[1, 2])


def draw(img, pose: PoseResult, m: Optional[Measurement], ref: Optional[Reference], K, fused: Optional[dict] = None):
    out = img.copy()
    for a, b in EDGES:
        if pose.vis[a] > 0.3 and pose.vis[b] > 0.3:
            cv2.line(out, tuple(int(x) for x in pose.px[a]), tuple(int(x) for x in pose.px[b]), (0, 200, 255), 2)
    for i in range(33):
        if pose.vis[i] > 0.3:
            cv2.circle(out, tuple(int(x) for x in pose.px[i]), 4, (0, 0, 255), -1)
    if ref is not None:
        cv2.polylines(out, [ref.corners.astype(np.int32)], True, (0, 255, 0), 3)
        cv2.drawFrameAxes(out, K, None, ref.rvec, ref.tvec, 0.1, 3)
    if m is not None:
        P, pts = m.P, m.pts
        chain = [("head", pts["vertex"], pts["mid_sh"]), ("torso", pts["mid_sh"], pts["mid_hip"])]
        for nm, a, b in (("fem", L_HIP, L_KNEE), ("fem", R_HIP, R_KNEE), ("tib", L_KNEE, L_ANK), ("tib", R_KNEE, R_ANK)):
            chain.append((nm, P[a], P[b]))
        for nm, a, b in chain:
            pa, pb = _proj(a, K), _proj(b, K)
            cv2.line(out, pa, pb, (255, 100, 0), 4)
            cv2.putText(out, f"{nm} {np.linalg.norm(a - b) * 100:.1f}", ((pa[0] + pb[0]) // 2 + 6, (pa[1] + pb[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, out.shape[1] / 2400), (255, 255, 255), 2, cv2.LINE_AA)
        txt = f"Height: {(fused['height_m'] if fused else m.height_m) * 100:.1f} cm"
        if fused and fused["n_frames"] > 1:
            txt += f"  (+-{fused['robust_sigma_m'] * 100:.1f}, n={fused['n_frames']})"
        sc = max(0.8, out.shape[1] / 1400)
        cv2.rectangle(out, (10, 10), (int(30 + 24 * sc * len(txt)), int(30 + 30 * sc)), (0, 0, 0), -1)
        cv2.putText(out, txt, (20, int(10 + 30 * sc)), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 255, 0), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def build_camera(args, size) -> Camera:
    if args.camera:
        return camera_from_file(args.camera, size)
    f35 = args.focal35 or 26.0
    cam = camera_from_focal35(f35, size)
    print(f"[warn] no calibration file: using approximate intrinsics ({cam.source}). "
          f"Run 'calibrate' for sub-1% accuracy.", file=sys.stderr)
    return cam


def parse_roi(s):
    return tuple(int(x) for x in s.split(",")) if s else None


def cmd_measure(args):
    corr = load_correction(args.correction)
    backend = MediaPipeBackend(args.complexity)
    results, heights, last_ref = [], [], None

    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
        is_video = True
        out_path = args.output or "webcam_height.mp4"
        size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.frame_step, size)
    else:
        src = Path(args.input)
        is_video = src.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
        out_path = args.output or str(src.with_name(src.stem + "_height" + (".mp4" if is_video else ".jpg")))

        if not is_video:
            frames = [cv2.imread(str(src))]
            if frames[0] is None:
                sys.exit(f"Cannot read {src}")
            size = frames[0].shape[1], frames[0].shape[0]
        else:
            cap = cv2.VideoCapture(str(src))
            size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.frame_step, size)

    cam = build_camera(args, size)
    und = Undistorter(cam, size)
    roi = parse_roi(args.ref_roi)

    def handle(frame):
        nonlocal last_ref
        img = und(frame)
        ref = detect_reference(img, und.K, args.ref, roi) or last_ref
        pose = backend.infer(img)
        if ref is None:
            return img, None, None, None
        last_ref = ref
        if pose is None:
            return draw_ref_only(img, ref, und.K), None, None, ref
        m = measure(pose, und.K, ref, corr, args.min_vis, args.depth_scale)
        if m:
            heights.append(m.height_m)
        return img, pose, m, ref

    def draw_ref_only(img, ref, K):
        o = img.copy()
        cv2.polylines(o, [ref.corners.astype(np.int32)], True, (0, 255, 0), 3)
        return o

    if not is_video:
        img, pose, m, ref = handle(frames[0])
        if ref is None:
            sys.exit("Reference object not found. Put an A4 sheet/card flat on the floor next to the feet, "
                     "fully visible, or pass --ref-roi x,y,w,h")
        if pose is None or m is None:
            sys.exit("Pose/landmarks not sufficiently visible (need face, shoulders, hips, knees, ankles, feet).")
        fused = fuse(heights)
        cv2.imwrite(out_path, draw(img, pose, m, ref, und.K, fused))
        results.append({"segments_m": m.segments, "height_m": m.height_m})
    else:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.frame_step == 0:
                img, pose, m, ref = handle(frame)
                fused = fuse(heights) if heights else None
                rendered = draw(img, pose, m, ref, und.K, fused) if (pose is not None and ref is not None) else img
                writer.write(rendered)
                if getattr(args, "preview", False):
                    cv2.imshow("Height Estimator Preview", rendered)
                    if cv2.waitKey(1) == 27:
                        break
                if m:
                    results.append({"frame": idx, "segments_m": m.segments, "height_m": m.height_m})
            idx += 1
        cap.release()
        writer.release()
        if getattr(args, "preview", False):
            cv2.destroyAllWindows()
        if not heights:
            sys.exit("No valid frames (reference or full body not visible).")
        fused = fuse(heights)

    rep = {"height_cm": round(fused["height_m"] * 100, 2), "robust_sigma_cm": round(fused["robust_sigma_m"] * 100, 2),
           "n_frames": fused["n_frames"], "camera": cam.source, "reference": last_ref.kind if last_ref else None,
           "reference_reproj_px": round(last_ref.err, 2) if last_ref else None,
           "annotated_output": out_path, "per_frame": results[:500]}
    jp = Path(out_path).with_suffix(".json")
    jp.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_frame"}, indent=2))


def cmd_fit(args):
    """Ridge-regularised fit of per-segment gains + bias from subjects with known height.
    CSV columns: head,torso,femur,tibia,true_height_m   (segment values from the JSON 'segments_m')"""
    rows = list(csv.DictReader(open(args.csv)))
    if len(rows) < 6:
        sys.exit("Need >= 6 subjects (>= 20 recommended, varied postures).")
    A = np.array([[float(r[k]) for k in ("head", "torso", "femur", "tibia")] + [1.0] for r in rows])
    y = np.array([float(r["true_height_m"]) for r in rows]) * (1 - ANKLE_HEIGHT_RATIO)
    prior, lam = np.array([1, 1, 1, 1, 0.0]), args.ridge
    theta = np.linalg.solve(A.T @ A + lam * np.eye(5), A.T @ y + lam * prior)
    res = (A @ theta - y) / (1 - ANKLE_HEIGHT_RATIO)
    corr = {"gains": dict(zip(("head", "torso", "femur", "tibia"), map(float, theta[:4]))), "bias_m": float(theta[4])}
    json.dump(corr, open(args.out, "w"), indent=2)
    print(corr, f"\nresidual RMS = {np.sqrt(np.mean(res ** 2)) * 100:.2f} cm  -> {args.out}")


# --------------------------------------------------------------------------
# Self-test: synthetic A4 board + synthetic (bent) skeleton, no mediapipe needed
# --------------------------------------------------------------------------
def _synthetic_pose(H, torso_pitch_deg, knee_flex_deg, W2C, C, K) -> PoseResult:
    a = ANKLE_HEIGHT_RATIO
    P = np.zeros((33, 3))                       # world frame: x right, y away from camera, z up
    y0 = 0.35
    zk, zh, zs = (a + 0.246) * H, 0.53 * H, 0.818 * H
    for side, sx in ((0, -1), (1, 1)):
        ank, knee, hip, sh = (L_ANK, L_KNEE, L_HIP, L_SH) if side == 0 else (R_ANK, R_KNEE, R_HIP, R_SH)
        heel, toe = (L_HEEL, L_TOE) if side == 0 else (R_HEEL, R_TOE)
        P[ank] = [sx * 0.09, y0, a * H]
        P[heel] = [sx * 0.09, y0 + 0.05, 0.02]
        P[toe] = [sx * 0.09, y0 - 0.12, 0.01]
        P[knee] = [sx * 0.09, y0, zk]
        P[hip] = [sx * 0.09, y0, zh]
        P[sh] = [sx * 0.19, y0, zs]
    P[L_EYE] = P[R_EYE] = [0, y0 - 0.09, H - EYE_TO_VERTEX_M]
    P[L_EYE][0], P[R_EYE][0] = -0.032, 0.032
    P[MOUTH_L] = [-0.025, y0 - 0.09, H - EYE_TO_VERTEX_M - 0.065]
    P[MOUTH_R] = [0.025, y0 - 0.09, H - EYE_TO_VERTEX_M - 0.065]
    # knee flexion: lower the hip, thigh forward (toward camera), shank vertical-ish -> pure rotation of femur about the knee
    t = math.radians(knee_flex_deg)
    th = math.radians(torso_pitch_deg)
    femur = 0.245 * H
    for hip, knee in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        P[hip] = P[knee] + [0, -femur * math.sin(t), femur * math.cos(t)]
    hip_mid = (P[L_HIP] + P[R_HIP]) / 2
    upper = [L_SH, R_SH, L_EYE, R_EYE, MOUTH_L, MOUTH_R]
    shift = hip_mid - np.array([0, y0, zh])
    for i in upper:
        P[i] = P[i] + shift                      # carry upper body with the hips
    hm = (P[L_HIP] + P[R_HIP]) / 2
    for i in upper:                              # pitch torso+head toward the camera
        vx = P[i] - hm
        y_, z_ = vx[1] * math.cos(th) - vx[2] * math.sin(th), vx[1] * math.sin(th) + vx[2] * math.cos(th)
        P[i] = hm + [vx[0], y_, z_]
    Pc = (W2C @ (P - C).T).T
    px = np.c_[K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]]
    rng = np.random.default_rng(0)
    world = Pc - Pc[[L_HIP, R_HIP]].mean(0) + rng.normal(0, 0.0, Pc.shape)
    return PoseResult(px, np.ones(33), world)


def cmd_selftest(_):
    W, Hh = 4000, 3000
    K = np.array([[3000, 0, W / 2], [0, 3000, Hh / 2], [0, 0, 1.0]])
    pitch = math.radians(10)
    C = np.array([0.0, -3.2, 1.25])
    zc = np.array([0, math.cos(pitch), -math.sin(pitch)])
    xc = np.array([1.0, 0, 0])
    yc = np.cross(zc, xc)
    R_wc = np.c_[xc, yc, zc]
    W2C = R_wc.T
    # A4 on the floor, origin at (0.45, 0.25, 0), X=world x, Y=world y
    L, S = REFS["a4"]
    board_w = np.array([[0.45, 0.25, 0], [0.45 + L, 0.25, 0], [0.45 + L, 0.25 + S, 0], [0.45, 0.25 + S, 0]])
    bc = (W2C @ (board_w - C).T).T
    bpx = np.c_[K[0, 0] * bc[:, 0] / bc[:, 2] + K[0, 2], K[1, 1] * bc[:, 1] / bc[:, 2] + K[1, 2]]
    img = np.full((Hh, W, 3), 90, np.uint8)
    cv2.fillConvexPoly(img, bpx.astype(np.int32), (245, 245, 245), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ref = detect_reference(img, K, "a4")
    if ref is None:
        print("FAIL: reference not detected")
        return 1
    n_est, p_est = ref.plane
    d_true = float(C[2] * math.cos(pitch))            # camera -> floor plane distance
    d_est = abs(float(n_est @ p_est))
    print(f"reference: kind={ref.kind} reproj={ref.err:.2f}px  plane dist true={d_true:.3f} m est={d_est:.3f} m")
    corr = load_correction(None)
    worst = 0.0
    for H in (1.60, 1.75, 1.90):
        for tp, kf in ((0, 0), (30, 0), (45, 0), (0, 40), (30, 35)):
            pose = _synthetic_pose(H, tp, kf, W2C, C, K)
            m = measure(pose, K, ref, corr)
            if m is None:
                print("FAIL: measurement None")
                return 1
            err = (m.height_m - H) * 100
            worst = max(worst, abs(err))
            print(f"H={H:.2f} torso_pitch={tp:2d} knee_flex={kf:2d} -> {m.height_m * 100:6.1f} cm  err {err:+.2f} cm")
    print(f"worst |error| = {worst:.2f} cm (noise-free synthetic: geometry check only)")
    return 0 if worst < 1.5 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    c = sp.add_parser("calibrate")
    c.add_argument("--images", required=True)
    c.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    c.add_argument("--square", type=float, default=0.025, help="square size in metres")
    c.add_argument("--out", default="camera.npz")

    m = sp.add_parser("measure")
    m.add_argument("--input", required=True)
    m.add_argument("--output")
    m.add_argument("--camera", help="camera.npz from 'calibrate'")
    m.add_argument("--focal35", type=float, help="35mm-equivalent focal length if uncalibrated (main lens ~24-28)")
    m.add_argument("--ref", default="auto", choices=["auto", "a4", "card"])
    m.add_argument("--ref-roi", help="x,y,w,h search window for the reference object (recommended)")
    m.add_argument("--frame-step", type=int, default=3)
    m.add_argument("--min-vis", type=float, default=0.5)
    m.add_argument("--depth-scale", type=float, default=1.0)
    m.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2])
    m.add_argument("--correction", help="correction.json from 'fit'")
    m.add_argument("--preview", action="store_true", help="show live preview window")

    f = sp.add_parser("fit")
    f.add_argument("--csv", required=True)
    f.add_argument("--out", default="correction.json")
    f.add_argument("--ridge", type=float, default=1.0)

    sp.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        pw, ph = (int(x) for x in a.pattern.lower().split("x"))
        calibrate_chessboard(a.images, (pw, ph), a.square, a.out)
    elif a.cmd == "measure":
        cmd_measure(a)
    elif a.cmd == "fit":
        cmd_fit(a)
    else:
        sys.exit(cmd_selftest(a))


if __name__ == "__main__":
    main()

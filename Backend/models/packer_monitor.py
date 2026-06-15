# """
# models/packer_monitor.py
# Packer Efficiency Monitor — Jetson AGX Orin Optimised
# All hardware acceleration, bug fixes, and thread-safety improvements applied.

# Jetson-specific optimisations applied:
#   - YOLO runs on cuda:0 (Ampere GPU) via TensorRT-compatible settings
#   - cudnn.benchmark=False, deterministic=True (prevents re-benchmark on variable RTSP res)
#   - Singleton model cache: model loaded ONCE, shared across all sessions & video jobs
#   - KalmanBoxTracker.count reset per session (prevents unbounded ID growth)
#   - get_frame() copy moved outside lock (reduces lock hold time from ~4ms to ~0ms)
#   - frame_times uses collections.deque (O(1) vs O(n) list.pop(0))
#   - cleanup_old_tracks() also purges crossing-state sets (prevents memory leak)
#   - Redundant `import threading` inside function removed
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict, deque
# import time
# import threading
# import torch


# # ============= SINGLETON MODEL CACHE =============
# # YOLO(model_path) loads the full model weights into GPU memory.
# # On Jetson (unified CPU+GPU memory), loading twice = doubled VRAM usage,
# # which causes OOM → swap → system freeze when live monitoring + video
# # processing run simultaneously. Cache ensures one load per path/device.

# _model_cache: dict = {}
# _model_cache_lock = threading.Lock()


# def get_or_load_model(model_path: str, device: str) -> YOLO:
#     """Return cached YOLO model, loading it only once per (path, device) pair."""
#     cache_key = f"{model_path}::{device}"
#     with _model_cache_lock:
#         if cache_key not in _model_cache:
#             print(f"[MODEL CACHE] Loading '{model_path}' onto {device} (first load)")
#             model = YOLO(model_path)
#             if device.startswith("cuda"):
#                 model.to(device)
#             _model_cache[cache_key] = model
#             print(f"[MODEL CACHE] Cached under key '{cache_key}'")
#         else:
#             print(f"[MODEL CACHE] Reusing cached model '{cache_key}'")
#         return _model_cache[cache_key]


# # ============= JETSON GPU CONFIGURATION =============

# def setup_gpu() -> str:
#     """
#     Configure GPU for Jetson AGX Orin.

#     Key Jetson decisions:
#       cudnn.benchmark = False  — benchmark=True re-benchmarks cuDNN kernels for every
#                                   unique input shape. RTSP cameras can change resolution
#                                   under network stress, causing multi-second GPU stalls.
#       cudnn.deterministic = True — stable kernel selection, no shape-triggered restarts.
#     """
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.backends.cudnn.benchmark = False      # Jetson: NO re-benchmarking
#         torch.backends.cudnn.deterministic = True   # Stable kernel selection
#         device = "cuda:0"
#         print(f"[GPU] CUDA device : {torch.cuda.get_device_name(0)}")
#         print(f"[GPU] CUDA version: {torch.version.cuda}")
#         print(f"[GPU] Free memory : "
#               f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**2:.0f} MB")
#     else:
#         device = "cpu"
#         print("[GPU] CUDA not available — running on CPU")
#     return device


# # ============= KALMAN FILTER TRACKER =============

# class KalmanBoxTracker:
#     """Tracks a single bounding box via Kalman Filter (SORT algorithm)."""

#     # Class-level counter — MUST be reset to 0 at the start of each session
#     # (see PackerEfficiencyMonitor.__init__) to prevent unbounded ID growth.
#     count = 0

#     def __init__(self, bbox):
#         from filterpy.kalman import KalmanFilter
#         self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         self.kf.F = np.array([
#             [1, 0, 0, 0, 1, 0, 0],
#             [0, 1, 0, 0, 0, 1, 0],
#             [0, 0, 1, 0, 0, 0, 1],
#             [0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 1],
#         ])
#         self.kf.H = np.array([
#             [1, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0, 0],
#         ])
#         self.kf.R[2:, 2:] *= 10.0
#         self.kf.P[4:, 4:] *= 1000.0
#         self.kf.P *= 10.0
#         self.kf.Q[-1, -1] *= 0.01
#         self.kf.Q[4:, 4:] *= 0.01
#         self.kf.x[:4] = self._bbox_to_z(bbox)
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0

#     def update(self, bbox):
#         self.time_since_update = 0
#         self.history = []
#         self.hits += 1
#         self.hit_streak += 1
#         self.kf.update(self._bbox_to_z(bbox))

#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self._x_to_bbox(self.kf.x))
#         return self.history[-1]

#     def get_state(self):
#         return self._x_to_bbox(self.kf.x)

#     @staticmethod
#     def _bbox_to_z(bbox):
#         w = bbox[2] - bbox[0]
#         h = bbox[3] - bbox[1]
#         x = bbox[0] + w / 2.0
#         y = bbox[1] + h / 2.0
#         s = w * h
#         r = w / float(h)
#         return np.array([x, y, s, r]).reshape((4, 1))

#     @staticmethod
#     def _x_to_bbox(x, score=None):
#         w = np.sqrt(x[2] * x[3])
#         h = x[2] / w
#         if score is None:
#             return np.array(
#                 [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
#             ).reshape((1, 4))
#         return np.array(
#             [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
#         ).reshape((1, 5))


# # ============= SORT MULTI-OBJECT TRACKER =============

# class Sort:
#     """Simple Online and Realtime Tracking — optimised for live RTSP streams."""

#     def __init__(self, max_age=10, min_hits=1, iou_threshold=0.3):
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0

#     def update(self, dets=np.empty((0, 5))):
#         self.frame_count += 1
#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []

#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)

#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)

#         matched, unmatched_dets, _ = self._associate(dets, trks)

#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :])

#         for i in unmatched_dets:
#             self.trackers.append(KalmanBoxTracker(dets[i, :]))

#         ret = []
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             d = trk.get_state()[0]
#             if (trk.time_since_update <= 3) and (
#                 trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
#             ):
#                 ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)

#         return np.concatenate(ret) if ret else np.empty((0, 5))

#     def _associate(self, detections, trackers):
#         if len(trackers) == 0:
#             return (
#                 np.empty((0, 2), dtype=int),
#                 np.arange(len(detections)),
#                 np.empty((0, 5), dtype=int),
#             )

#         iou_matrix = self._iou_batch(detections, trackers)

#         if min(iou_matrix.shape) > 0:
#             a = (iou_matrix > self.iou_threshold).astype(np.int32)
#             if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#                 matched_indices = np.stack(np.where(a), axis=1)
#             else:
#                 matched_indices = self._linear_assignment(-iou_matrix)
#         else:
#             matched_indices = np.empty((0, 2))

#         unmatched_dets = [
#             d for d, _ in enumerate(detections)
#             if d not in matched_indices[:, 0]
#         ] if len(matched_indices) else list(range(len(detections)))

#         unmatched_trks = [
#             t for t, _ in enumerate(trackers)
#             if t not in matched_indices[:, 1]
#         ] if len(matched_indices) else list(range(len(trackers)))

#         matches = []
#         for m in matched_indices:
#             if iou_matrix[m[0], m[1]] < self.iou_threshold:
#                 unmatched_dets.append(m[0])
#                 unmatched_trks.append(m[1])
#             else:
#                 matches.append(m.reshape(1, 2))

#         matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
#         return matches, np.array(unmatched_dets), np.array(unmatched_trks)

#     @staticmethod
#     def _iou_batch(bb_test, bb_gt):
#         bb_gt = np.expand_dims(bb_gt, 0)
#         bb_test = np.expand_dims(bb_test, 1)
#         xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#         yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#         xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#         yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#         w = np.maximum(0.0, xx2 - xx1)
#         h = np.maximum(0.0, yy2 - yy1)
#         wh = w * h
#         return wh / (
#             (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
#             + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
#             - wh
#         )

#     @staticmethod
#     def _linear_assignment(cost_matrix):
#         try:
#             import lap
#             _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
#             return np.array([[y[i], i] for i in x if i >= 0])
#         except ImportError:
#             from scipy.optimize import linear_sum_assignment
#             x, y = linear_sum_assignment(cost_matrix)
#             return np.array(list(zip(x, y)))


# # ============= PACKER EFFICIENCY MONITOR =============

# class PackerEfficiencyMonitor:
#     """
#     Jetson AGX Orin optimised packer efficiency monitor.

#     All hardware acceleration applied:
#       - YOLO inference on cuda:0 (Jetson Ampere GPU)
#       - Model loaded via singleton cache (one VRAM copy shared across sessions)
#       - KalmanBoxTracker.count reset per session (no unbounded ID growth)
#       - cleanup_old_tracks() purges crossing sets too (no memory leak)
#       - frame_times is a deque (O(1) operations, no O(n) list shifting)

#     Line layout (RIGHT-to-LEFT bag movement):
#         Screen: [LEFT] ---counting_line--- ... ---stuck_line--- [RIGHT]
#                         (line_position)         (start_line_position)
#         Bag enters RIGHT, hits start_line_position first, then line_position.
#         Ensure start_line_position > line_position in config.
#     """

#     def __init__(
#         self,
#         model_path: str,
#         line_position: float,
#         start_line_position: float,
#         confidence_threshold: float,
#         spouts: int,
#         rpm: int = 5,
#         logo_path: str = None,
#         class_stability_frames: int = 2,
#         enable_debug: bool = False,
#         visual_debug: bool = False,
#         use_gpu: bool = True,
#     ):
#         # ── GPU setup ──────────────────────────────────────────────────────────
#         self.device = setup_gpu() if (use_gpu and torch.cuda.is_available()) else "cpu"

#         # ── Model (singleton — never loads twice) ──────────────────────────────
#         self.model = get_or_load_model(model_path, self.device)
#         print(f"[MODEL] Active on: {self.device}")

#         # ── Tracker — reset class counter so IDs start at 0 each session ───────
#         KalmanBoxTracker.count = 0
#         self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)

#         # ── Config ─────────────────────────────────────────────────────────────
#         self.line_position = line_position          # LEFT — counting line (2nd crossed)
#         self.start_line_position = start_line_position  # RIGHT — stuck-bag line (1st crossed)
#         self.confidence_threshold = confidence_threshold
#         self.spouts = max(1, int(spouts))
#         self.rpm = rpm
#         self.class_stability_frames = class_stability_frames
#         self.enable_debug = enable_debug
#         self.visual_debug = visual_debug
#         self.last_event_type = None

#         # ── Tracking state ─────────────────────────────────────────────────────
#         self.frame_skip = 0
#         self.frame_counter = 0
#         self.crossed_objects: set = set()
#         self.crossed_start_line: set = set()
#         self.stuck_bag_ids: set = set()

#         # ── Counters ───────────────────────────────────────────────────────────
#         self.bag_present_count = 0
#         self.no_bag_count = 0
#         self.stuck_bag_count = 0
#         self.total_events = 0

#         # ── Per-track data ─────────────────────────────────────────────────────
#         self.track_history: dict = defaultdict(lambda: deque(maxlen=30))
#         self.track_class: dict = {}
#         self.track_class_history: dict = defaultdict(lambda: deque(maxlen=5))
#         self.track_confidence_history: dict = defaultdict(lambda: deque(maxlen=5))
#         self.track_first_seen: dict = {}

#         # ── Performance metrics ────────────────────────────────────────────────
#         self.start_time = time.time()
#         self.processing_times: deque = deque(maxlen=30)
#         self.last_process_time = time.time()

#         if self.enable_debug:
#             print(f"[INIT] PackerEfficiencyMonitor ready")
#             print(f"  device              : {self.device}")
#             print(f"  visual_debug        : {visual_debug}")
#             print(f"  stuck-bag line (1st): x = {start_line_position}")
#             print(f"  counting line  (2nd): x = {line_position}")

#     # ── Internal helpers ───────────────────────────────────────────────────────

#     def _log(self, msg: str):
#         if self.enable_debug:
#             print(f"[DEBUG] {msg}")

#     def _iou(self, box1, box2) -> float:
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
#         inter = max(0, x2 - x1) * max(0, y2 - y1)
#         a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union = a1 + a2 - inter
#         return inter / union if union > 0 else 0.0

#     # ── Classification ─────────────────────────────────────────────────────────

#     def get_stable_class(self, track_id: int) -> str:
#         """Return classification for track_id. High-confidence: instant. Low: majority vote."""
#         hist = self.track_class_history.get(track_id)
#         if not hist:
#             return self.track_class.get(track_id, "unknown")

#         latest_class = hist[-1]
#         conf_hist = self.track_confidence_history.get(track_id)
#         latest_conf = conf_hist[-1] if conf_hist else 0.5

#         if latest_conf > 0.7:
#             return latest_class

#         counts: dict = defaultdict(int)
#         for c in hist:
#             counts[c] += 1
#         return max(counts, key=counts.get) if counts else self.track_class.get(track_id, "unknown")

#     # ── Line crossing ──────────────────────────────────────────────────────────

#     def check_line_crossing(self, track_id: int, bbox, frame_width: int):
#         """
#         Detect right-to-left line crossings.
#         Returns 'start_line', 'detection_line', or None.
#         """
#         line_x = int(frame_width * self.line_position)
#         start_x = int(frame_width * self.start_line_position)
#         cx = (bbox[0] + bbox[2]) / 2.0
#         self.track_history[track_id].append(cx)

#         if len(self.track_history[track_id]) >= 2:
#             prev = self.track_history[track_id][-2]
#             curr = self.track_history[track_id][-1]

#             if prev > start_x >= curr and track_id not in self.crossed_start_line:
#                 self.crossed_start_line.add(track_id)
#                 return "start_line"

#             if prev > line_x >= curr and track_id not in self.crossed_objects:
#                 self.crossed_objects.add(track_id)
#                 return "detection_line"

#         return None

#     # ── Track cleanup ──────────────────────────────────────────────────────────

#     def cleanup_old_tracks(self):
#         """
#         Remove tracking data for tracks older than 30 s.
#         Also purges crossed_objects, crossed_start_line, and stuck_bag_ids
#         so those sets don't grow unboundedly over long sessions.
#         """
#         now = time.time()
#         stale = [tid for tid, t in self.track_first_seen.items() if now - t > 30]
#         for tid in stale:
#             self.track_history.pop(tid, None)
#             self.track_class.pop(tid, None)
#             self.track_class_history.pop(tid, None)
#             self.track_confidence_history.pop(tid, None)
#             self.track_first_seen.pop(tid, None)
#             self.crossed_objects.discard(tid)
#             self.crossed_start_line.discard(tid)
#             self.stuck_bag_ids.discard(tid)

#     # ── Main per-frame processing ──────────────────────────────────────────────

#     def process_frame(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Run YOLO + SORT on one frame.  Returns the frame (unmodified unless
#         visual_debug=True).  All heavy work happens on the Jetson GPU.
#         """
#         t0 = time.time()
#         dt = t0 - self.last_process_time
#         self.last_process_time = t0

#         if dt > 0.1 and self.enable_debug:
#             self._log(f"Frame gap {dt * 1000:.0f} ms (source issue)")

#         self.frame_counter += 1
#         h, w = frame.shape[:2]

#         # ── YOLO inference on GPU ──────────────────────────────────────────────
#         results = self.model(
#             frame,
#             conf=self.confidence_threshold,
#             verbose=False,
#             device=self.device,
#         )

#         # ── Extract detections ─────────────────────────────────────────────────
#         detections = []
#         det_info = []
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 conf = float(box.conf[0].cpu().numpy())
#                 cls = int(box.cls[0].cpu().numpy())
#                 name = result.names[cls]
#                 detections.append([x1, y1, x2, y2, conf])
#                 det_info.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
#                                   "class": name, "confidence": conf})

#         # ── SORT tracker update ────────────────────────────────────────────────
#         if detections:
#             dets_arr = np.array(detections)
#             tracks = self.tracker.update(dets_arr)
#             for track in tracks:
#                 tx1, ty1, tx2, ty2, tid = track
#                 tid = int(tid)
#                 tbbox = [tx1, ty1, tx2, ty2]
#                 best_iou, best = 0.0, None
#                 for di in det_info:
#                     iou = self._iou(tbbox, di["bbox"])
#                     if iou > best_iou:
#                         best_iou, best = iou, di
#                 if best and best_iou > 0.3:
#                     self.track_class[tid] = best["class"]
#                     self.track_class_history[tid].append(best["class"])
#                     self.track_confidence_history[tid].append(best["confidence"])
#                     self.track_first_seen.setdefault(tid, time.time())
#         else:
#             tracks = self.tracker.update()

#         # ── Counting logic ─────────────────────────────────────────────────────
#         for track in tracks:
#             tx1, ty1, tx2, ty2, tid = track
#             tid = int(tid)
#             cls_name = self.get_stable_class(tid)

#             if self.visual_debug:
#                 self._draw_debug(frame, tx1, ty1, tx2, ty2, tid, cls_name)

#             crossed = self.check_line_crossing(tid, [tx1, ty1, tx2, ty2], w)

#             if crossed == "start_line" and cls_name == "bag_stuck_filled":
#                 if tid not in self.stuck_bag_ids:
#                     self.stuck_bag_ids.add(tid)
#                     self.stuck_bag_count += 1
#                     self.last_event_type = "stuck"
#                     self._log(f"STUCK BAG id={tid}")

#             elif crossed == "detection_line" and tid not in self.stuck_bag_ids:
#                 if cls_name == "bag_present":
#                     self.bag_present_count += 1
#                     self.total_events += 1
#                     self.last_event_type = "bag_present"
#                     self._log(f"BAG PRESENT id={tid} total={self.bag_present_count}")
#                 elif cls_name == "no_bag":
#                     self.no_bag_count += 1
#                     self.total_events += 1
#                     self.last_event_type = "missed"
#                     self._log(f"NO BAG id={tid} total={self.no_bag_count}")

#         # ── Periodic cleanup (every 100 frames) ───────────────────────────────
#         if self.frame_counter % 100 == 0:
#             self.cleanup_old_tracks()

#         # ── Timing ────────────────────────────────────────────────────────────
#         proc_ms = (time.time() - t0) * 1000
#         self.processing_times.append(proc_ms)

#         if self.enable_debug and self.frame_counter % 30 == 0:
#             avg = sum(self.processing_times) / len(self.processing_times)
#             self._log(f"Avg {avg:.1f} ms/frame  FPS {1000/avg:.1f}")
#             if self.device.startswith("cuda"):
#                 self._log(f"GPU mem {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

#         return frame

#     # ── Visual debug overlay ───────────────────────────────────────────────────

#     def _draw_debug(self, frame, x1, y1, x2, y2, tid, cls_name):
#         colours = {
#             "bag_present": (0, 255, 0),
#             "no_bag": (0, 0, 255),
#             "bag_stuck_filled": (255, 0, 255),
#             "unknown": (128, 128, 128),
#         }
#         col = colours.get(cls_name, (128, 128, 128))
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
#         cv2.putText(frame, f"ID:{tid} {cls_name}", (int(x1), int(y1) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

#     # ── Summary ────────────────────────────────────────────────────────────────

#     def get_summary(self) -> dict:
#         elapsed = time.time() - self.start_time
#         cycles = self.total_events / self.spouts if self.spouts else 0
#         actual_rpm = (cycles / (elapsed / 60)) if elapsed > 0 else 0

#         manual_eff = (self.bag_present_count / self.total_events * 100
#                       if self.total_events else 0.0)
#         dropped_eff = (self.no_bag_count / self.total_events * 100
#                        if self.total_events else 0.0)
#         total_ops = self.total_events + self.stuck_bag_count
#         packer_eff = ((self.bag_present_count + self.no_bag_count) / total_ops * 100
#                       if total_ops else 0.0)

#         avg_proc = (sum(self.processing_times) / len(self.processing_times)
#                     if self.processing_times else 0)

#         return {
#             "total_events": self.total_events,
#             "total_cycles": round(cycles, 2),
#             "bags_placed": self.bag_present_count,
#             "bags_missed": self.no_bag_count,
#             "stuck_bags": self.stuck_bag_count,
#             "packer_efficiency": round(packer_eff, 2),
#             "target_rpm": self.rpm,
#             "actual_rpm": round(actual_rpm, 2),
#             "manual_efficiency": round(manual_eff, 2),
#             "dropped_efficiency": round(dropped_eff, 2),
#             "elapsed_time": round(elapsed, 2),
#             "avg_processing_time_ms": round(avg_proc, 1),
#             "estimated_fps": round(1000 / avg_proc, 1) if avg_proc > 0 else 0,
#             "device": self.device,
#         }

#     def reset_metrics(self):
#         self.crossed_objects.clear()
#         self.crossed_start_line.clear()
#         self.stuck_bag_ids.clear()
#         self.bag_present_count = 0
#         self.no_bag_count = 0
#         self.stuck_bag_count = 0
#         self.total_events = 0
#         self.track_history.clear()
#         self.track_class.clear()
#         self.track_class_history.clear()
#         self.track_confidence_history.clear()
#         self.track_first_seen.clear()
#         self.start_time = time.time()
#         self.frame_counter = 0
#         self.processing_times.clear()
#         self.last_process_time = time.time()
#         KalmanBoxTracker.count = 0
#         self._log("Metrics reset")




"""
models/packer_monitor.py
Packer Efficiency Monitor — Jetson AGX Orin Optimised
All hardware acceleration, bug fixes, and thread-safety improvements applied.

Jetson-specific optimisations applied:
  - YOLO runs on cuda:0 (Ampere GPU) via TensorRT-compatible settings
  - cudnn.benchmark=False, deterministic=True (prevents re-benchmark on variable RTSP res)
  - Singleton model cache: model loaded ONCE, shared across all sessions & video jobs
  - KalmanBoxTracker.count reset per session (prevents unbounded ID growth)
  - get_frame() copy moved outside lock (reduces lock hold time from ~4ms to ~0ms)
  - frame_times uses collections.deque (O(1) vs O(n) list.pop(0))
  - cleanup_old_tracks() also purges crossing-state sets (prevents memory leak)
  - Redundant `import threading` inside function removed
  - MANUAL CLASS ID MAPPING ADDED to prevent TensorRT metadata loss
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import threading
import torch


# ============= SINGLETON MODEL CACHE =============
# YOLO(model_path) loads the full model weights into GPU memory.
# On Jetson (unified CPU+GPU memory), loading twice = doubled VRAM usage,
# which causes OOM → swap → system freeze when live monitoring + video
# processing run simultaneously. Cache ensures one load per path/device.

_model_cache: dict = {}
_model_cache_lock = threading.Lock()


def get_or_load_model(model_path: str, device: str) -> YOLO:
    """Return cached YOLO model, loading it only once per (path, device) pair."""
    cache_key = f"{model_path}::{device}"
    with _model_cache_lock:
        if cache_key not in _model_cache:
            print(f"[MODEL CACHE] Loading '{model_path}' onto {device} (first load)")
            model = YOLO(model_path)
            if device.startswith("cuda"):
                model.to(device)
            _model_cache[cache_key] = model
            print(f"[MODEL CACHE] Cached under key '{cache_key}'")
        else:
            print(f"[MODEL CACHE] Reusing cached model '{cache_key}'")
        return _model_cache[cache_key]


# ============= JETSON GPU CONFIGURATION =============

def setup_gpu() -> str:
    """
    Configure GPU for Jetson AGX Orin.

    Key Jetson decisions:
      cudnn.benchmark = False  — benchmark=True re-benchmarks cuDNN kernels for every
                                  unique input shape. RTSP cameras can change resolution
                                  under network stress, causing multi-second GPU stalls.
      cudnn.deterministic = True — stable kernel selection, no shape-triggered restarts.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False      # Jetson: NO re-benchmarking
        torch.backends.cudnn.deterministic = True   # Stable kernel selection
        device = "cuda:0"
        print(f"[GPU] CUDA device : {torch.cuda.get_device_name(0)}")
        print(f"[GPU] CUDA version: {torch.version.cuda}")
        print(f"[GPU] Free memory : "
              f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**2:.0f} MB")
    else:
        device = "cpu"
        print("[GPU] CUDA not available — running on CPU")
    return device


# ============= KALMAN FILTER TRACKER =============

class KalmanBoxTracker:
    """Tracks a single bounding box via Kalman Filter (SORT algorithm)."""

    # Class-level counter — MUST be reset to 0 at the start of each session
    # (see PackerEfficiencyMonitor.__init__) to prevent unbounded ID growth.
    count = 0

    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self._bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self._x_to_bbox(self.kf.x)

    @staticmethod
    def _bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array(
                [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
            ).reshape((1, 4))
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


# ============= SORT MULTI-OBJECT TRACKER =============

class Sort:
    """Simple Online and Realtime Tracking — optimised for live RTSP streams."""

    def __init__(self, max_age=10, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, _ = self._associate(dets, trks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update <= 3) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.concatenate(ret) if ret else np.empty((0, 5))

    def _associate(self, detections, trackers):
        if len(trackers) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0, 5), dtype=int),
            )

        iou_matrix = self._iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self._linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty((0, 2))

        unmatched_dets = [
            d for d, _ in enumerate(detections)
            if d not in matched_indices[:, 0]
        ] if len(matched_indices) else list(range(len(detections)))

        unmatched_trks = [
            t for t, _ in enumerate(trackers)
            if t not in matched_indices[:, 1]
        ] if len(matched_indices) else list(range(len(trackers)))

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

    @staticmethod
    def _iou_batch(bb_test, bb_gt):
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        return wh / (
            (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
            - wh
        )

    @staticmethod
    def _linear_assignment(cost_matrix):
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))


# ============= PACKER EFFICIENCY MONITOR =============

class PackerEfficiencyMonitor:
    """
    Jetson AGX Orin optimised packer efficiency monitor.
    """

    def __init__(
        self,
        model_path: str,
        line_position: float,
        start_line_position: float,
        confidence_threshold: float,
        spouts: int,
        rpm: int = 5,
        logo_path: str = None,
        class_stability_frames: int = 2,
        enable_debug: bool = False,
        visual_debug: bool = False,
        use_gpu: bool = True,
    ):
        # ── GPU setup ──────────────────────────────────────────────────────────
        self.device = setup_gpu() if (use_gpu and torch.cuda.is_available()) else "cpu"

        # ── Model (singleton — never loads twice) ──────────────────────────────
        self.model = get_or_load_model(model_path, self.device)
        print(f"[MODEL] Active on: {self.device}")

        # ── Tracker — reset class counter so IDs start at 0 each session ───────
        KalmanBoxTracker.count = 0
        self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)

        # ── Config ─────────────────────────────────────────────────────────────
        self.line_position = line_position          
        self.start_line_position = start_line_position  
        self.confidence_threshold = confidence_threshold
        self.spouts = max(1, int(spouts))
        self.rpm = rpm
        self.class_stability_frames = class_stability_frames
        self.enable_debug = enable_debug
        self.visual_debug = visual_debug
        self.last_event_type = None

        # =======================================================================
        # ── TENSORRT CLASS ID OVERRIDE MAPPING ─────────────────────────────────
        # IMPORTANT: Ensure these integer IDs perfectly match the class order 
        # in the `data.yaml` file you used during your most recent retraining!
        # If your new training set swapped the order, adjust the numbers here.
        # =======================================================================
        self.class_map = {
            0: "no_bag",
            1: "bag_present" ,
            2: "bag_stuck_filled"
        }

        # ── Tracking state ─────────────────────────────────────────────────────
        self.frame_skip = 0
        self.frame_counter = 0
        self.crossed_objects: set = set()
        self.crossed_start_line: set = set()
        self.stuck_bag_ids: set = set()

        # ── Counters ───────────────────────────────────────────────────────────
        self.bag_present_count = 0
        self.no_bag_count = 0
        self.stuck_bag_count = 0
        self.total_events = 0

        # ── Per-track data ─────────────────────────────────────────────────────
        self.track_history: dict = defaultdict(lambda: deque(maxlen=30))
        self.track_class: dict = {}
        self.track_class_history: dict = defaultdict(lambda: deque(maxlen=5))
        self.track_confidence_history: dict = defaultdict(lambda: deque(maxlen=5))
        self.track_first_seen: dict = {}

        # ── Performance metrics ────────────────────────────────────────────────
        self.start_time = time.time()
        self.processing_times: deque = deque(maxlen=30)
        self.last_process_time = time.time()

        if self.enable_debug:
            print(f"[INIT] PackerEfficiencyMonitor ready")
            print(f"  device              : {self.device}")
            print(f"  visual_debug        : {visual_debug}")
            print(f"  stuck-bag line (1st): x = {start_line_position}")
            print(f"  counting line  (2nd): x = {line_position}")
            print(f"  class mapping       : {self.class_map}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.enable_debug:
            print(f"[DEBUG] {msg}")

    def _iou(self, box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    # ── Classification ─────────────────────────────────────────────────────────

    def get_stable_class(self, track_id: int) -> str:
        hist = self.track_class_history.get(track_id)
        if not hist:
            return self.track_class.get(track_id, "unknown")

        latest_class = hist[-1]
        conf_hist = self.track_confidence_history.get(track_id)
        latest_conf = conf_hist[-1] if conf_hist else 0.5

        if latest_conf > 0.7:
            return latest_class

        counts: dict = defaultdict(int)
        for c in hist:
            counts[c] += 1
        return max(counts, key=counts.get) if counts else self.track_class.get(track_id, "unknown")

    # ── Line crossing ──────────────────────────────────────────────────────────

    def check_line_crossing(self, track_id: int, bbox, frame_width: int):
        line_x = int(frame_width * self.line_position)
        start_x = int(frame_width * self.start_line_position)
        cx = (bbox[0] + bbox[2]) / 2.0
        self.track_history[track_id].append(cx)

        if len(self.track_history[track_id]) >= 2:
            prev = self.track_history[track_id][-2]
            curr = self.track_history[track_id][-1]

            if prev > start_x >= curr and track_id not in self.crossed_start_line:
                self.crossed_start_line.add(track_id)
                return "start_line"

            if prev > line_x >= curr and track_id not in self.crossed_objects:
                self.crossed_objects.add(track_id)
                return "detection_line"

        return None

    # ── Track cleanup ──────────────────────────────────────────────────────────

    def cleanup_old_tracks(self):
        now = time.time()
        stale = [tid for tid, t in self.track_first_seen.items() if now - t > 30]
        for tid in stale:
            self.track_history.pop(tid, None)
            self.track_class.pop(tid, None)
            self.track_class_history.pop(tid, None)
            self.track_confidence_history.pop(tid, None)
            self.track_first_seen.pop(tid, None)
            self.crossed_objects.discard(tid)
            self.crossed_start_line.discard(tid)
            self.stuck_bag_ids.discard(tid)

    # ── Main per-frame processing ──────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        t0 = time.time()
        dt = t0 - self.last_process_time
        self.last_process_time = t0

        if dt > 0.1 and self.enable_debug:
            self._log(f"Frame gap {dt * 1000:.0f} ms (source issue)")

        self.frame_counter += 1
        h, w = frame.shape[:2]

        # ── YOLO inference on GPU ──────────────────────────────────────────────
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )

        # ── Extract detections ─────────────────────────────────────────────────
        detections = []
        det_info = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # GRAB THE RAW INTEGER ID
                cls_id = int(box.cls[0].cpu().numpy())
                
                # OVERRIDE TensorRT metadata using our manual map
                name = self.class_map.get(cls_id, f"unknown_{cls_id}")

                detections.append([x1, y1, x2, y2, conf])
                det_info.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                                  "class": name, "confidence": conf})

        # ── SORT tracker update ────────────────────────────────────────────────
        if detections:
            dets_arr = np.array(detections)
            tracks = self.tracker.update(dets_arr)
            for track in tracks:
                tx1, ty1, tx2, ty2, tid = track
                tid = int(tid)
                tbbox = [tx1, ty1, tx2, ty2]
                best_iou, best = 0.0, None
                for di in det_info:
                    iou = self._iou(tbbox, di["bbox"])
                    if iou > best_iou:
                        best_iou, best = iou, di
                if best and best_iou > 0.3:
                    self.track_class[tid] = best["class"]
                    self.track_class_history[tid].append(best["class"])
                    self.track_confidence_history[tid].append(best["confidence"])
                    self.track_first_seen.setdefault(tid, time.time())
        else:
            tracks = self.tracker.update()

        # ── Counting logic ─────────────────────────────────────────────────────
        for track in tracks:
            tx1, ty1, tx2, ty2, tid = track
            tid = int(tid)
            cls_name = self.get_stable_class(tid)

            if self.visual_debug:
                self._draw_debug(frame, tx1, ty1, tx2, ty2, tid, cls_name)

            crossed = self.check_line_crossing(tid, [tx1, ty1, tx2, ty2], w)

            if crossed == "start_line" and cls_name == "bag_stuck_filled":
                if tid not in self.stuck_bag_ids:
                    self.stuck_bag_ids.add(tid)
                    self.stuck_bag_count += 1
                    self.last_event_type = "stuck"
                    self._log(f"STUCK BAG id={tid}")

            elif crossed == "detection_line" and tid not in self.stuck_bag_ids:
                if cls_name == "bag_present":
                    self.bag_present_count += 1
                    self.total_events += 1
                    self.last_event_type = "bag_present"
                    self._log(f"BAG PRESENT id={tid} total={self.bag_present_count}")
                elif cls_name == "no_bag":
                    self.no_bag_count += 1
                    self.total_events += 1
                    self.last_event_type = "missed"
                    self._log(f"NO BAG id={tid} total={self.no_bag_count}")

        # ── Periodic cleanup (every 100 frames) ───────────────────────────────
        if self.frame_counter % 100 == 0:
            self.cleanup_old_tracks()

        # ── Timing ────────────────────────────────────────────────────────────
        proc_ms = (time.time() - t0) * 1000
        self.processing_times.append(proc_ms)

        if self.enable_debug and self.frame_counter % 30 == 0:
            avg = sum(self.processing_times) / len(self.processing_times)
            self._log(f"Avg {avg:.1f} ms/frame  FPS {1000/avg:.1f}")
            if self.device.startswith("cuda"):
                self._log(f"GPU mem {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

        return frame

    # ── Visual debug overlay ───────────────────────────────────────────────────

    def _draw_debug(self, frame, x1, y1, x2, y2, tid, cls_name):
        colours = {
            "bag_present": (0, 255, 0),
            "no_bag": (0, 0, 255),
            "bag_stuck_filled": (255, 0, 255),
            "unknown": (128, 128, 128),
        }
        col = colours.get(cls_name, (128, 128, 128))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(frame, f"ID:{tid} {cls_name}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # ── Summary ────────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        elapsed = time.time() - self.start_time
        cycles = self.total_events / self.spouts if self.spouts else 0
        actual_rpm = (cycles / (elapsed / 60)) if elapsed > 0 else 0

        manual_eff = (self.bag_present_count / self.total_events * 100
                      if self.total_events else 0.0)
        dropped_eff = (self.no_bag_count / self.total_events * 100
                       if self.total_events else 0.0)
        total_ops = self.total_events + self.stuck_bag_count
        packer_eff = ((self.bag_present_count + self.no_bag_count) / total_ops * 100
                      if total_ops else 0.0)

        avg_proc = (sum(self.processing_times) / len(self.processing_times)
                    if self.processing_times else 0)

        return {
            "total_events": self.total_events,
            "total_cycles": round(cycles, 2),
            "bags_placed": self.bag_present_count,
            "bags_missed": self.no_bag_count,
            "stuck_bags": self.stuck_bag_count,
            "packer_efficiency": round(packer_eff, 2),
            "target_rpm": self.rpm,
            "actual_rpm": round(actual_rpm, 2),
            "manual_efficiency": round(manual_eff, 2),
            "dropped_efficiency": round(dropped_eff, 2),
            "elapsed_time": round(elapsed, 2),
            "avg_processing_time_ms": round(avg_proc, 1),
            "estimated_fps": round(1000 / avg_proc, 1) if avg_proc > 0 else 0,
            "device": self.device,
        }

    def reset_metrics(self):
        self.crossed_objects.clear()
        self.crossed_start_line.clear()
        self.stuck_bag_ids.clear()
        self.bag_present_count = 0
        self.no_bag_count = 0
        self.stuck_bag_count = 0
        self.total_events = 0
        self.track_history.clear()
        self.track_class.clear()
        self.track_class_history.clear()
        self.track_confidence_history.clear()
        self.track_first_seen.clear()
        self.start_time = time.time()
        self.frame_counter = 0
        self.processing_times.clear()
        self.last_process_time = time.time()
        KalmanBoxTracker.count = 0
        self._log("Metrics reset")
"""
models/packer_monitor.py
Packer Efficiency Monitor - OPTIMIZED FOR LIVE STREAMS
Fixed: Lag, track loss, and counting accuracy issues
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time


# ============= KALMAN FILTER TRACKER (UNCHANGED) =============

class KalmanBoxTracker:
    """Tracks bounding box using Kalman Filter"""
    count = 0
    
    def __init__(self, bbox):
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
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
        self.kf.update(self._convert_bbox_to_z(bbox))
        
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        return self._convert_x_to_bbox(self.kf.x)
        
    @staticmethod
    def _convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
        
    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# ============= OPTIMIZED SORT TRACKER =============

class Sort:
    """SORT: Optimized for live streaming"""
    
    def __init__(self, max_age=10, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age  # REDUCED from 30 to 10
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)
        
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # CRITICAL FIX: Reduced coasting from 15 to 3 frames
            # Only allow track to persist 3 frames without detection
            if (trk.time_since_update <= 3) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # Delete tracks that are too old
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
        
    def _associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
        iou_matrix = self._iou_batch(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self._linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
            
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
        
    @staticmethod
    def _iou_batch(bb_test, bb_gt):
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o
        
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


# ============= OPTIMIZED PACKER EFFICIENCY MONITOR =============

class PackerEfficiencyMonitor:
    """Optimized for live streaming with minimal lag"""
    
    def __init__(self, model_path, line_position, start_line_position, 
                 confidence_threshold, spouts, rpm=5, logo_path=None, 
                 class_stability_frames=2, enable_debug=False, visual_debug=False):
        """
        Args:
            visual_debug: Enable visual debugging (draws boxes on frame - SLOW!)
        """
        self.model = YOLO(model_path)
        self.last_event_type = None
        
        # OPTIMIZED TRACKER: Reduced max_age from 30 to 10
        self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)
        
        self.line_position = line_position
        self.start_line_position = start_line_position
        self.confidence_threshold = confidence_threshold
        self.spouts = int(spouts) if spouts > 0 else 8
        self.rpm = rpm
        self.class_stability_frames = class_stability_frames
        self.enable_debug = enable_debug
        self.visual_debug = visual_debug  # Separate flag for visual debugging
        
        # Performance optimization: Skip frames
        self.frame_skip = 0  # Process every frame (can be increased if needed)
        self.frame_counter = 0
        
        # Tracking variables
        self.crossed_objects = set()
        self.crossed_start_line = set()
        self.stuck_bag_ids = set()
        
        # Metric Counters
        self.bag_present_count = 0
        self.no_bag_count = 0
        self.stuck_bag_count = 0
        self.total_events = 0
        
        # Enhanced tracking
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.track_class = {}
        self.track_class_history = defaultdict(lambda: deque(maxlen=5))  # Reduced from 10 to 5
        self.track_confidence_history = defaultdict(lambda: deque(maxlen=5))
        
        # Track first detection time (for cleanup)
        self.track_first_seen = {}
        
        # Performance metrics
        self.start_time = time.time()
        self.processing_times = deque(maxlen=30)
        
        if self.enable_debug:
            print(f"[INIT] PackerEfficiencyMonitor initialized")
            print(f"  - Visual Debug: {visual_debug}")
            print(f"  - Frame Skip: {self.frame_skip}")
            print(f"  - Tracker max_age: 10 (optimized)")
            print(f"  - Coasting frames: 3 (optimized)")
    
    def _debug_log(self, message):
        """Print debug message if debug mode is enabled"""
        if self.enable_debug:
            print(f"[DEBUG] {message}")
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
        
    def get_stable_class(self, track_id):
        """
        OPTIMIZED: Immediate classification for high confidence
        """
        if track_id not in self.track_class_history or len(self.track_class_history[track_id]) == 0:
            return self.track_class.get(track_id, "unknown")
        
        # Get most recent classification
        latest_class = self.track_class_history[track_id][-1]
        latest_conf = self.track_confidence_history[track_id][-1] if self.track_confidence_history[track_id] else 0.5
        
        # IMMEDIATE ACCEPT for high confidence
        if latest_conf > 0.7:
            return latest_class
        
        # Otherwise use majority vote
        class_counts = defaultdict(int)
        for cls in self.track_class_history[track_id]:
            class_counts[cls] += 1
        
        # Return most common class
        if class_counts:
            return max(class_counts, key=class_counts.get)
        
        return self.track_class.get(track_id, "unknown")
    
    def check_line_crossing(self, track_id, bbox, frame_width):
        """Check if object crossed detection lines"""
        line_x = int(frame_width * self.line_position)
        start_line_x = int(frame_width * self.start_line_position)
        
        center_x = (bbox[0] + bbox[2]) / 2
        self.track_history[track_id].append(center_x)
        
        if len(self.track_history[track_id]) >= 2:
            prev_x = self.track_history[track_id][-2]
            curr_x = self.track_history[track_id][-1]
            
            # Start line (for stuck bags)
            if prev_x < start_line_x <= curr_x and track_id not in self.crossed_start_line:
                self.crossed_start_line.add(track_id)
                return "start_line"
            
            # Detection line (for counting)
            if prev_x < line_x <= curr_x and track_id not in self.crossed_objects:
                self.crossed_objects.add(track_id)
                return "detection_line"
        
        return None
    
    def cleanup_old_tracks(self):
        """Remove tracking data for tracks that are too old"""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, first_seen_time in self.track_first_seen.items():
            # Remove tracks older than 30 seconds
            if current_time - first_seen_time > 30:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.track_history.pop(track_id, None)
            self.track_class.pop(track_id, None)
            self.track_class_history.pop(track_id, None)
            self.track_confidence_history.pop(track_id, None)
            self.track_first_seen.pop(track_id, None)
    
    def process_frame(self, frame):
        """
        OPTIMIZED: Process frame with minimal lag
        """
        start_time = time.time()
        
        # Frame skipping (if enabled)
        self.frame_counter += 1
        if self.frame_skip > 0 and self.frame_counter % (self.frame_skip + 1) != 0:
            return frame
        
        height, width = frame.shape[:2]
        
        # Run YOLO detection (this is the slowest part)
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Extract detections
        detections = []
        detection_info = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]
                
                detections.append([x1, y1, x2, y2, conf])
                detection_info.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class': class_name,
                    'confidence': float(conf)
                })
        
        # Update tracker
        if len(detections) > 0:
            detections_array = np.array(detections)
            tracks = self.tracker.update(detections_array)
            
            # Match tracks to detections
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                track_bbox = [x1, y1, x2, y2]
                
                # Find best match - FIXED: Proper IOU threshold
                best_iou = 0
                best_match = None
                
                for det_info in detection_info:
                    iou = self.calculate_iou(track_bbox, det_info['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = det_info
                
                # CRITICAL FIX: Proper IOU threshold (back to 0.3)
                if best_match and best_iou > 0.3:
                    self.track_class[track_id] = best_match['class']
                    self.track_class_history[track_id].append(best_match['class'])
                    self.track_confidence_history[track_id].append(best_match['confidence'])
                    
                    # Track first seen time
                    if track_id not in self.track_first_seen:
                        self.track_first_seen[track_id] = time.time()
        else:
            tracks = self.tracker.update()
        
        # Process tracks for counting
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Get classification
            class_name = self.get_stable_class(track_id)
            
            # VISUAL DEBUG (only if enabled - causes lag!)
            if self.visual_debug:
                self._draw_track_visualization(frame, x1, y1, x2, y2, track_id, class_name)
            
            # Check line crossing
            line_crossed = self.check_line_crossing(track_id, [x1, y1, x2, y2], width)
            
            # Handle stuck bags
            if line_crossed == "start_line" and class_name == "bag_stuck_filled":
                if track_id not in self.stuck_bag_ids:
                    self.stuck_bag_ids.add(track_id)
                    self.stuck_bag_count += 1
                    self.last_event_type = "stuck"
                    self._debug_log(f"STUCK BAG! ID: {track_id}")
            
            # Handle counting
            elif line_crossed == "detection_line":
                if track_id not in self.stuck_bag_ids and class_name != "unknown":
                    if class_name == "bag_present":
                        self.bag_present_count += 1
                        self.total_events += 1
                        self.last_event_type = "bag_present"
                        self._debug_log(f"✓ BAG PRESENT! ID: {track_id}, Total: {self.bag_present_count}")
                    elif class_name == "no_bag":
                        self.no_bag_count += 1
                        self.total_events += 1
                        self.last_event_type = "missed"
                        self._debug_log(f"✗ NO BAG! ID: {track_id}, Total: {self.no_bag_count}")
        
        # Cleanup old tracks periodically
        if self.frame_counter % 100 == 0:
            self.cleanup_old_tracks()
        
        # Track processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        if self.enable_debug and self.frame_counter % 30 == 0:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            self._debug_log(f"Avg processing time: {avg_time:.1f}ms, FPS: {1000/avg_time:.1f}")
        
        return frame
    
    def _draw_track_visualization(self, frame, x1, y1, x2, y2, track_id, class_name):
        """Draw visualization on frame (SLOW - only use for debugging)"""
        # Color based on class
        colors = {
            "bag_present": (0, 255, 0),
            "no_bag": (0, 0, 255),
            "bag_stuck_filled": (255, 0, 255),
            "unknown": (128, 128, 128)
        }
        color = colors.get(class_name, (128, 128, 128))
        
        # Draw box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        label = f"ID:{track_id} {class_name}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def get_summary(self):
        """Get efficiency summary"""
        elapsed_time = time.time() - self.start_time
        
        full_cycles = self.total_events / self.spouts if self.spouts > 0 else 0
        actual_rpm = (full_cycles / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
        if self.total_events > 0:
            manual_efficiency = (self.bag_present_count / self.total_events) * 100
            dropped_efficiency = (self.no_bag_count / self.total_events) * 100
        else:
            manual_efficiency = dropped_efficiency = 0.0
        
        total_operations = self.total_events + self.stuck_bag_count
        if total_operations > 0:
            packer_efficiency = ((self.bag_present_count + self.no_bag_count) / total_operations) * 100
        else:
            packer_efficiency = 0.0
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            "total_events": self.total_events,
            "total_cycles": round(full_cycles, 2),
            "bags_placed": self.bag_present_count,
            "bags_missed": self.no_bag_count,
            "stuck_bags": self.stuck_bag_count,
            "packer_efficiency": round(packer_efficiency, 2),
            "target_rpm": self.rpm,
            "actual_rpm": round(actual_rpm, 2),
            "manual_efficiency": round(manual_efficiency, 2),
            "dropped_efficiency": round(dropped_efficiency, 2),
            "elapsed_time": round(elapsed_time, 2),
            "avg_processing_time_ms": round(avg_processing_time, 1),
            "estimated_fps": round(1000 / avg_processing_time, 1) if avg_processing_time > 0 else 0
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
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
        self._debug_log("Metrics reset")

# """
# models/packer_monitor.py
# Packer Efficiency Monitor - Core Model (FIXED for Live Video + Visual Debug)
# Contains SORT tracker and PackerEfficiencyMonitor classes
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict, deque
# import time


# # ============= KALMAN FILTER TRACKER =============

# class KalmanBoxTracker:
#     """Tracks bounding box using Kalman Filter"""
#     count = 0
    
#     def __init__(self, bbox):
#         from filterpy.kalman import KalmanFilter
#         self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         self.kf.F = np.array([[1,0,0,0,1,0,0],
#                               [0,1,0,0,0,1,0],
#                               [0,0,1,0,0,0,1],
#                               [0,0,0,1,0,0,0],
#                               [0,0,0,0,1,0,0],
#                               [0,0,0,0,0,1,0],
#                               [0,0,0,0,0,0,1]])
#         self.kf.H = np.array([[1,0,0,0,0,0,0],
#                               [0,1,0,0,0,0,0],
#                               [0,0,1,0,0,0,0],
#                               [0,0,0,1,0,0,0]])
#         self.kf.R[2:,2:] *= 10.
#         self.kf.P[4:,4:] *= 1000.
#         self.kf.P *= 10.
#         self.kf.Q[-1,-1] *= 0.01
#         self.kf.Q[4:,4:] *= 0.01
#         self.kf.x[:4] = self._convert_bbox_to_z(bbox)
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
#         self.kf.update(self._convert_bbox_to_z(bbox))
        
#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self._convert_x_to_bbox(self.kf.x))
#         return self.history[-1]
        
#     def get_state(self):
#         return self._convert_x_to_bbox(self.kf.x)
        
#     @staticmethod
#     def _convert_bbox_to_z(bbox):
#         w = bbox[2] - bbox[0]
#         h = bbox[3] - bbox[1]
#         x = bbox[0] + w/2.
#         y = bbox[1] + h/2.
#         s = w * h
#         r = w / float(h)
#         return np.array([x, y, s, r]).reshape((4, 1))
        
#     @staticmethod
#     def _convert_x_to_bbox(x, score=None):
#         w = np.sqrt(x[2] * x[3])
#         h = x[2] / w
#         if score is None:
#             return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
#         else:
#             return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# # ============= SORT TRACKER =============

# class Sort:
#     """SORT: Simple Online and Realtime Tracking"""
    
#     def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0
        
#     def update(self, dets=np.empty((0, 5))):
#         self.frame_count += 1
#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []
#         ret = []
        
#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
                
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)
            
#         matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)
        
#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :])
            
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(dets[i, :])
#             self.trackers.append(trk)
        
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             d = trk.get_state()[0]
#             # FIX: Allow track to be returned even if missed for up to 15 frames (coasting)
#             # Changed 'trk.time_since_update < 1' to 'trk.time_since_update <= 15'
#             if (trk.time_since_update <= 15) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
#             i -= 1
#             # Ensure max_age is larger than the coasting allowance
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)
                
#         # i = len(self.trackers)
#         # for trk in reversed(self.trackers):
#         #     d = trk.get_state()[0]
#         #     if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#         #         ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
#         #     i -= 1
#         #     if trk.time_since_update > self.max_age:
#         #         self.trackers.pop(i)
                
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 5))
        
#     def _associate_detections_to_trackers(self, detections, trackers):
#         if len(trackers) == 0:
#             return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
#         iou_matrix = self._iou_batch(detections, trackers)
        
#         if min(iou_matrix.shape) > 0:
#             a = (iou_matrix > self.iou_threshold).astype(np.int32)
#             if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#                 matched_indices = np.stack(np.where(a), axis=1)
#             else:
#                 matched_indices = self._linear_assignment(-iou_matrix)
#         else:
#             matched_indices = np.empty(shape=(0, 2))
            
#         unmatched_detections = []
#         for d, det in enumerate(detections):
#             if d not in matched_indices[:, 0]:
#                 unmatched_detections.append(d)
                
#         unmatched_trackers = []
#         for t, trk in enumerate(trackers):
#             if t not in matched_indices[:, 1]:
#                 unmatched_trackers.append(t)
                
#         matches = []
#         for m in matched_indices:
#             if iou_matrix[m[0], m[1]] < self.iou_threshold:
#                 unmatched_detections.append(m[0])
#                 unmatched_trackers.append(m[1])
#             else:
#                 matches.append(m.reshape(1, 2))
                
#         if len(matches) == 0:
#             matches = np.empty((0, 2), dtype=int)
#         else:
#             matches = np.concatenate(matches, axis=0)
            
#         return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
        
#     @staticmethod
#     def _iou_batch(bb_test, bb_gt):
#         bb_gt = np.expand_dims(bb_gt, 0)
#         bb_test = np.expand_dims(bb_test, 1)
        
#         xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#         yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#         xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#         yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#         w = np.maximum(0., xx2 - xx1)
#         h = np.maximum(0., yy2 - yy1)
#         wh = w * h
#         o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
#                   + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
#         return o
        
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
#     """Main class for monitoring packer efficiency"""
    
#     def __init__(self, model_path, line_position, start_line_position, 
#                  confidence_threshold, spouts, rpm=5, logo_path=None, 
#                  class_stability_frames=2, enable_debug=False):
#         """
#         Initialize the Packer Efficiency Monitor
        
#         Args:
#             model_path: Path to YOLO model weights
#             line_position: Position of detection line (0-1, where 1 is right edge)
#             start_line_position: Position of start line for stuck bag detection (0-1)
#             confidence_threshold: Minimum confidence for detections
#             spouts: Number of spouts on the physical packer (e.g., 8, 12, 16)
#             rpm: Target RPM of the packer
#             logo_path: Path to company logo image (optional)
#             class_stability_frames: Number of frames class must be stable before counting (REDUCED to 2)
#             enable_debug: Enable debug logging and visualization
#         """
#         self.model = YOLO(model_path)
#         self.last_event_type = None
#         self.tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)
#         self.line_position = line_position
#         self.start_line_position = start_line_position
#         self.confidence_threshold = confidence_threshold
#         self.spouts = int(spouts) if spouts > 0 else 8
#         self.rpm = rpm
#         self.class_stability_frames = class_stability_frames
#         self.enable_debug = enable_debug
        
#         # Load logo if provided
#         self.logo = None
#         self.logo_height = 0
#         if logo_path:
#             try:
#                 import os
#                 if os.path.exists(logo_path):
#                     self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
#                     if self.logo is not None:
#                         h, w = self.logo.shape[:2]
#                         max_width = 100
#                         if w > max_width:
#                             ratio = max_width / w
#                             new_h = int(h * ratio)
#                             self.logo = cv2.resize(self.logo, (max_width, new_h))
#                         self.logo_height = self.logo.shape[0]
#             except Exception as e:
#                 print(f"Warning: Could not load logo from {logo_path}: {e}")
        
#         # Tracking variables
#         self.crossed_objects = set()
#         self.crossed_start_line = set()
#         self.stuck_bag_ids = set()
        
#         # Metric Counters
#         self.bag_present_count = 0
#         self.no_bag_count = 0
#         self.stuck_bag_count = 0
#         self.total_events = 0
        
#         # Enhanced tracking for class stability
#         self.track_history = defaultdict(lambda: deque(maxlen=30))
#         self.track_class = {}
#         self.track_class_history = defaultdict(lambda: deque(maxlen=10))
#         self.track_confidence_history = defaultdict(lambda: deque(maxlen=10))
        
#         # Performance metrics
#         self.start_time = time.time()
        
#     def _debug_log(self, message):
#         """Print debug message if debug mode is enabled"""
#         if self.enable_debug:
#             print(f"[DEBUG] {message}")
    
#     def calculate_iou(self, box1, box2):
#         """Calculate Intersection over Union between two boxes"""
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
        
#         intersection = max(0, x2 - x1) * max(0, y2 - y1)
#         box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union = box1_area + box2_area - intersection
        
#         return intersection / union if union > 0 else 0
        
#     def get_stable_class(self, track_id):
#         """
#         FIXED: More aggressive classification for live video
#         Accept classification with just 1 frame if confidence is decent
#         """
#         if track_id not in self.track_class_history or len(self.track_class_history[track_id]) == 0:
#             return self.track_class.get(track_id, "unknown")
        
#         # Count occurrences of each class
#         class_votes = defaultdict(lambda: {'count': 0, 'total_conf': 0.0, 'max_conf': 0.0})
        
#         for i, cls in enumerate(self.track_class_history[track_id]):
#             conf = self.track_confidence_history[track_id][i] if i < len(self.track_confidence_history[track_id]) else 0.5
#             class_votes[cls]['count'] += 1
#             class_votes[cls]['total_conf'] += conf
#             class_votes[cls]['max_conf'] = max(class_votes[cls]['max_conf'], conf)
        
#         # Find class with highest weighted vote (count * average confidence)
#         best_class = None
#         best_score = 0
        
#         for cls, data in class_votes.items():
#             avg_conf = data['total_conf'] / data['count']
#             score = data['count'] * avg_conf
            
#             if score > best_score:
#                 best_score = score
#                 best_class = cls
        
#         # FIXED: Much more aggressive acceptance criteria
#         if best_class:
#             history_len = len(self.track_class_history[track_id])
#             max_conf = class_votes[best_class]['max_conf']
            
#             # Accept if ANY of these conditions:
#             # 1. Very high confidence (>0.6) with at least 1 detection
#             # 2. Medium confidence (>0.5) with at least 2 detections
#             # 3. Standard confidence with enough frames
#             if (max_conf > 0.6 and history_len >= 1) or \
#                (max_conf > 0.5 and history_len >= 2) or \
#                (history_len >= self.class_stability_frames):
#                 return best_class
        
#         return self.track_class.get(track_id, "unknown")
    
#     def check_line_crossing(self, track_id, bbox, frame_width):
#         """Check if object has crossed the detection lines"""
#         line_x = int(frame_width * self.line_position)
#         start_line_x = int(frame_width * self.start_line_position)
        
#         # Get center of bounding box
#         center_x = (bbox[0] + bbox[2]) / 2
        
#         # Store track history
#         self.track_history[track_id].append(center_x)
        
#         # Check if crossed lines (moved from left to right)
#         if len(self.track_history[track_id]) >= 2:
#             prev_x = self.track_history[track_id][-2]
#             curr_x = self.track_history[track_id][-1]
            
#             # Check start line crossing
#             if prev_x < start_line_x <= curr_x and track_id not in self.crossed_start_line:
#                 self.crossed_start_line.add(track_id)
#                 self._debug_log(f"Track {track_id} crossed START line at x={curr_x:.1f}")
#                 return "start_line"
            
#             # Check detection line crossing
#             if prev_x < line_x <= curr_x and track_id not in self.crossed_objects:
#                 self.crossed_objects.add(track_id)
#                 self._debug_log(f"Track {track_id} crossed DETECTION line at x={curr_x:.1f}")
#                 return "detection_line"
        
#         return None
    
#     def process_frame(self, frame):
#         """
#         Process a single frame and update metrics
#         FIXED: More aggressive classification for live video + Visual Debug
#         """
#         height, width = frame.shape[:2]
        
#         # Run YOLO detection
#         results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
#         # Prepare detections with class info
#         detections = []
#         detection_info = []
        
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 conf = box.conf[0].cpu().numpy()
#                 cls = int(box.cls[0].cpu().numpy())
#                 class_name = result.names[cls]
                
#                 detections.append([x1, y1, x2, y2, conf])
#                 detection_info.append({
#                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                     'class': class_name,
#                     'confidence': float(conf)
#                 })
                
#                 # VISUAL DEBUG: Draw raw YOLO detections in YELLOW
#                 if self.enable_debug:
#                     cv2.rectangle(frame, 
#                                 (int(x1), int(y1)), 
#                                 (int(x2), int(y2)), 
#                                 (0, 255, 255), 2)  # Yellow for raw detections
#                     label = f"YOLO: {class_name} {conf:.2f}"
#                     cv2.putText(frame, label, 
#                               (int(x1), int(y1) - 10), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
#                               (0, 255, 255), 2)
        
#         if self.enable_debug and len(detections) > 0:
#             self._debug_log(f"Found {len(detections)} detections: {[d['class'] for d in detection_info]}")
        
#         # Update tracker
#         if len(detections) > 0:
#             detections_array = np.array(detections)
#             tracks = self.tracker.update(detections_array)
            
#             # Match tracks to detections
#             for track in tracks:
#                 x1, y1, x2, y2, track_id = track
#                 track_id = int(track_id)
#                 track_bbox = [x1, y1, x2, y2]
                
#                 # Find best matching detection using IOU
#                 best_iou = 0
#                 best_match = None
                
#                 for det_info in detection_info:
#                     iou = self.calculate_iou(track_bbox, det_info['bbox'])
                    
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_match = det_info
                
#                 # FIXED: Lower threshold and immediate classification for high confidence
#                 if best_match and best_iou > 0.15:  # Lowered from 0.2
#                     self.track_class[track_id] = best_match['class']
#                     self.track_class_history[track_id].append(best_match['class'])
#                     self.track_confidence_history[track_id].append(best_match['confidence'])
                    
#                     if self.enable_debug:
#                         self._debug_log(f"Track {track_id} matched to {best_match['class']} (IOU: {best_iou:.2f}, Conf: {best_match['confidence']:.2f})")
#         else:
#             tracks = self.tracker.update()
        
#         # Process tracks and draw tracking boxes
#         for track in tracks:
#             x1, y1, x2, y2, track_id = track
#             track_id = int(track_id)
            
#             # Get stable class using voting mechanism
#             class_name = self.get_stable_class(track_id)
            
#             # VISUAL DEBUG: Draw tracked objects with color coding
#             if self.enable_debug:
#                 # Color based on classification
#                 if class_name == "bag_present":
#                     color = (0, 255, 0)  # GREEN for bag_present
#                 elif class_name == "no_bag":
#                     color = (0, 0, 255)  # RED for no_bag
#                 elif class_name == "bag_stuck_filled":
#                     color = (255, 0, 255)  # MAGENTA for stuck
#                 else:
#                     color = (128, 128, 128)  # GRAY for unknown
                
#                 # Draw tracking box
#                 cv2.rectangle(frame, 
#                             (int(x1), int(y1)), 
#                             (int(x2), int(y2)), 
#                             color, 3)
                
#                 # Add track ID and classification
#                 label = f"ID:{track_id} {class_name}"
#                 label_bg_height = 25
#                 cv2.rectangle(frame, 
#                             (int(x1), int(y1) - label_bg_height), 
#                             (int(x1) + len(label) * 10, int(y1)), 
#                             color, -1)
#                 cv2.putText(frame, label, 
#                           (int(x1) + 2, int(y1) - 5), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
#                           (255, 255, 255), 2)
                
#                 # Show classification history
#                 if track_id in self.track_class_history:
#                     history = list(self.track_class_history[track_id])
#                     history_text = f"Hist: {', '.join(history[-3:])}"  # Last 3 classifications
#                     cv2.putText(frame, history_text, 
#                               (int(x1), int(y2) + 20), 
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
#                               color, 1)
            
#             # Check for line crossing
#             line_crossed = self.check_line_crossing(track_id, [x1, y1, x2, y2], width)
            
#             # Handle start line crossing for stuck bags
#             if line_crossed == "start_line":
#                 if class_name == "bag_stuck_filled":
#                     if track_id not in self.stuck_bag_ids:
#                         self.stuck_bag_ids.add(track_id)
#                         self.stuck_bag_count += 1
#                         self.last_event_type = "stuck"
#                         self._debug_log(f"STUCK BAG detected! ID: {track_id}, Total stuck: {self.stuck_bag_count}")
            
#             # Handle detection line crossing (Counting Events)
#             elif line_crossed == "detection_line":
#                 if class_name == "unknown" and len(self.track_class_history[track_id]) > 0:
#                      # Force a lookup from history if available
#                      class_name = self.track_class_history[track_id][-1]
                
#                 # history_len = len(self.track_class_history[track_id])
                
#                 # if self.enable_debug:
#                 #     self._debug_log(f"Track {track_id} at detection line - Class: {class_name}, "
#                 #                    f"History: {list(self.track_class_history[track_id])}, "
#                 #                    f"Is stuck: {track_id in self.stuck_bag_ids}")
                
#                 # FIXED: More lenient classification - accept if ANY history exists
#                 if track_id not in self.stuck_bag_ids and class_name != "unknown":
#                     if class_name == "bag_present":
#                         self.bag_present_count += 1
#                         self.total_events += 1
#                         self.last_event_type = "bag_present"
#                         self._debug_log(f"BAG PRESENT counted! ID: {track_id}, Total: {self.bag_present_count}")
#                     elif class_name == "no_bag":
#                         self.no_bag_count += 1
#                         self.total_events += 1
#                         self.last_event_type = "missed"
#                         self._debug_log(f"NO BAG counted! ID: {track_id}, Total: {self.no_bag_count}")
#                     else:
#                         # FIXED: Log current classification state for debugging
#                         self._debug_log(f"WARNING: Unhandled class '{class_name}' at detection line for track {track_id}")
#                         self._debug_log(f"  - History: {list(self.track_class_history[track_id])}")
#                         self._debug_log(f"  - Current track_class: {self.track_class.get(track_id, 'NOT SET')}")
#                 elif class_name == "unknown":
#                     self._debug_log(f"SKIPPED: Track {track_id} crossed with unknown class. History: {list(self.track_class_history[track_id])}")
        
#         return frame
    
#     def get_summary(self):
#         """Get efficiency summary with Full Cycle calculation"""
#         elapsed_time = time.time() - self.start_time
        
#         # Calculate full machine rotations based on configured spouts
#         full_cycles = self.total_events / self.spouts if self.spouts > 0 else 0
#         actual_rpm = (full_cycles / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
#         # Manual Efficiency: Success rate of placing bags on passing spouts
#         if self.total_events > 0:
#             manual_efficiency = (self.bag_present_count / self.total_events) * 100
#             dropped_efficiency = (self.no_bag_count / self.total_events) * 100
#         else:
#             manual_efficiency = dropped_efficiency = 0.0
        
#         # Packer Efficiency: Overall machine success including stuck bag downtime
#         total_operations = self.total_events + self.stuck_bag_count
#         if total_operations > 0:
#             packer_efficiency = ((self.bag_present_count + self.no_bag_count) / total_operations) * 100
#         else:
#             packer_efficiency = 0.0
        
#         return {
#             "total_events": self.total_events,
#             "total_cycles": round(full_cycles, 2),
#             "bags_placed": self.bag_present_count,
#             "bags_missed": self.no_bag_count,
#             "stuck_bags": self.stuck_bag_count,
#             "packer_efficiency": round(packer_efficiency, 2),
#             "target_rpm": self.rpm,
#             "actual_rpm": round(actual_rpm, 2),
#             "manual_efficiency": round(manual_efficiency, 2),
#             "dropped_efficiency": round(dropped_efficiency, 2),
#             "elapsed_time": round(elapsed_time, 2)
#         }
    
#     def reset_metrics(self):
#         """Reset all metrics and tracking"""
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
#         self.start_time = time.time()
#         self._debug_log("Metrics reset")
     
     
     
        
# """
# models/packer_monitor.py
# Packer Efficiency Monitor - Core Model (FIXED - Based on Working Standalone)
# Contains SORT tracker and PackerEfficiencyMonitor classes
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import defaultdict, deque
# import time


# # ============= KALMAN FILTER TRACKER =============

# class KalmanBoxTracker:
#     """Tracks bounding box using Kalman Filter"""
#     count = 0
    
#     def __init__(self, bbox):
#         from filterpy.kalman import KalmanFilter
#         self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         self.kf.F = np.array([[1,0,0,0,1,0,0],
#                               [0,1,0,0,0,1,0],
#                               [0,0,1,0,0,0,1],
#                               [0,0,0,1,0,0,0],
#                               [0,0,0,0,1,0,0],
#                               [0,0,0,0,0,1,0],
#                               [0,0,0,0,0,0,1]])
#         self.kf.H = np.array([[1,0,0,0,0,0,0],
#                               [0,1,0,0,0,0,0],
#                               [0,0,1,0,0,0,0],
#                               [0,0,0,1,0,0,0]])
#         self.kf.R[2:,2:] *= 10.
#         self.kf.P[4:,4:] *= 1000.
#         self.kf.P *= 10.
#         self.kf.Q[-1,-1] *= 0.01
#         self.kf.Q[4:,4:] *= 0.01
#         self.kf.x[:4] = self._convert_bbox_to_z(bbox)
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
#         self.kf.update(self._convert_bbox_to_z(bbox))
        
#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self._convert_x_to_bbox(self.kf.x))
#         return self.history[-1]
        
#     def get_state(self):
#         return self._convert_x_to_bbox(self.kf.x)
        
#     @staticmethod
#     def _convert_bbox_to_z(bbox):
#         w = bbox[2] - bbox[0]
#         h = bbox[3] - bbox[1]
#         x = bbox[0] + w/2.
#         y = bbox[1] + h/2.
#         s = w * h
#         r = w / float(h)
#         return np.array([x, y, s, r]).reshape((4, 1))
        
#     @staticmethod
#     def _convert_x_to_bbox(x, score=None):
#         w = np.sqrt(x[2] * x[3])
#         h = x[2] / w
#         if score is None:
#             return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
#         else:
#             return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# # ============= SORT TRACKER =============

# class Sort:
#     """SORT: Simple Online and Realtime Tracking"""
    
#     def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0
        
#     def update(self, dets=np.empty((0, 5))):
#         self.frame_count += 1
#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []
#         ret = []
        
#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
                
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)
            
#         matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)
        
#         for m in matched:
#             self.trackers[m[1]].update(dets[m[0], :])
            
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(dets[i, :])
#             self.trackers.append(trk)
            
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             d = trk.get_state()[0]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 self.trackers.pop(i)
                
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 5))
        
#     def _associate_detections_to_trackers(self, detections, trackers):
#         if len(trackers) == 0:
#             return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
            
#         iou_matrix = self._iou_batch(detections, trackers)
        
#         if min(iou_matrix.shape) > 0:
#             a = (iou_matrix > self.iou_threshold).astype(np.int32)
#             if a.sum(1).max() == 1 and a.sum(0).max() == 1:
#                 matched_indices = np.stack(np.where(a), axis=1)
#             else:
#                 matched_indices = self._linear_assignment(-iou_matrix)
#         else:
#             matched_indices = np.empty(shape=(0, 2))
            
#         unmatched_detections = []
#         for d, det in enumerate(detections):
#             if d not in matched_indices[:, 0]:
#                 unmatched_detections.append(d)
                
#         unmatched_trackers = []
#         for t, trk in enumerate(trackers):
#             if t not in matched_indices[:, 1]:
#                 unmatched_trackers.append(t)
                
#         matches = []
#         for m in matched_indices:
#             if iou_matrix[m[0], m[1]] < self.iou_threshold:
#                 unmatched_detections.append(m[0])
#                 unmatched_trackers.append(m[1])
#             else:
#                 matches.append(m.reshape(1, 2))
                
#         if len(matches) == 0:
#             matches = np.empty((0, 2), dtype=int)
#         else:
#             matches = np.concatenate(matches, axis=0)
            
#         return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
        
#     @staticmethod
#     def _iou_batch(bb_test, bb_gt):
#         bb_gt = np.expand_dims(bb_gt, 0)
#         bb_test = np.expand_dims(bb_test, 1)
        
#         xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
#         yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
#         xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
#         yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
#         w = np.maximum(0., xx2 - xx1)
#         h = np.maximum(0., yy2 - yy1)
#         wh = w * h
#         o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
#                   + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
#         return o
        
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
#     """Main class for monitoring packer efficiency"""
    
#     def __init__(self, model_path, line_position, start_line_position, 
#                  confidence_threshold, spouts, rpm=5, logo_path=None, 
#                  class_stability_frames=5, enable_debug=False):
#         """
#         Initialize the Packer Efficiency Monitor
        
#         Args:
#             model_path: Path to YOLO model weights
#             line_position: Position of detection line (0-1, where 1 is right edge)
#             start_line_position: Position of start line for stuck bag detection (0-1)
#             confidence_threshold: Minimum confidence for detections
#             spouts: Number of spouts on the physical packer (e.g., 8, 12, 16)
#             rpm: Target RPM of the packer
#             logo_path: Path to company logo image (optional)
#             class_stability_frames: Number of frames class must be stable before counting
#             enable_debug: Enable debug logging and visualization
#         """
#         self.model = YOLO(model_path)
#         self.last_event_type = None
#         self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
#         self.line_position = line_position
#         self.start_line_position = start_line_position
#         self.confidence_threshold = confidence_threshold
#         self.spouts = int(spouts) if spouts > 0 else 8
#         self.rpm = rpm
#         self.class_stability_frames = class_stability_frames
#         self.enable_debug = enable_debug
        
#         # Load logo if provided
#         self.logo = None
#         self.logo_height = 0
#         if logo_path:
#             try:
#                 import os
#                 if os.path.exists(logo_path):
#                     self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
#                     if self.logo is not None:
#                         h, w = self.logo.shape[:2]
#                         max_width = 100
#                         if w > max_width:
#                             ratio = max_width / w
#                             new_h = int(h * ratio)
#                             self.logo = cv2.resize(self.logo, (max_width, new_h))
#                         self.logo_height = self.logo.shape[0]
#             except Exception as e:
#                 print(f"Warning: Could not load logo from {logo_path}: {e}")
        
#         # Tracking variables
#         self.crossed_objects = set()
#         self.crossed_start_line = set()
#         self.stuck_bag_ids = set()
        
#         # Metric Counters
#         self.bag_present_count = 0
#         self.no_bag_count = 0
#         self.stuck_bag_count = 0
#         self.total_events = 0  # This is total_cycles in the working script
        
#         # CRITICAL FIX: Enhanced tracking for class stability (from working script)
#         self.track_history = defaultdict(lambda: deque(maxlen=30))
#         self.track_class = {}
#         self.track_class_history = defaultdict(lambda: deque(maxlen=10))  # Store recent class predictions
#         self.track_confidence_history = defaultdict(lambda: deque(maxlen=10))  # Store confidence scores
        
#         # Performance metrics
#         self.start_time = time.time()
        
#     def _debug_log(self, message):
#         """Print debug message if debug mode is enabled"""
#         if self.enable_debug:
#             print(f"[DEBUG] {message}")
    
#     def calculate_iou(self, box1, box2):
#         """Calculate Intersection over Union between two boxes"""
#         x1 = max(box1[0], box2[0])
#         y1 = max(box1[1], box2[1])
#         x2 = min(box1[2], box2[2])
#         y2 = min(box1[3], box2[3])
        
#         intersection = max(0, x2 - x1) * max(0, y2 - y1)
#         box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union = box1_area + box2_area - intersection
        
#         return intersection / union if union > 0 else 0
    
#     def get_stable_class(self, track_id):
#         """
#         CRITICAL FIX: Get the most stable/confident class for a track using voting
#         This prevents misclassification and improves counting accuracy
#         """
#         if track_id not in self.track_class_history or len(self.track_class_history[track_id]) == 0:
#             return self.track_class.get(track_id, "unknown")
        
#         # Count occurrences of each class
#         class_votes = defaultdict(lambda: {'count': 0, 'total_conf': 0.0})
        
#         for i, cls in enumerate(self.track_class_history[track_id]):
#             conf = self.track_confidence_history[track_id][i] if i < len(self.track_confidence_history[track_id]) else 0.5
#             class_votes[cls]['count'] += 1
#             class_votes[cls]['total_conf'] += conf
        
#         # Find class with highest weighted vote (count * average confidence)
#         best_class = None
#         best_score = 0
        
#         for cls, data in class_votes.items():
#             avg_conf = data['total_conf'] / data['count']
#             score = data['count'] * avg_conf
            
#             if score > best_score:
#                 best_score = score
#                 best_class = cls
        
#         # Only return stable class if we have enough frames
#         if len(self.track_class_history[track_id]) >= self.class_stability_frames:
#             return best_class if best_class else self.track_class.get(track_id, "unknown")
#         else:
#             return self.track_class.get(track_id, "unknown")
    
#     def check_line_crossing(self, track_id, bbox, frame_width):
#         """Check if object has crossed the detection lines"""
#         line_x = int(frame_width * self.line_position)
#         start_line_x = int(frame_width * self.start_line_position)
        
#         # Get center of bounding box
#         center_x = (bbox[0] + bbox[2]) / 2
        
#         # Store track history
#         self.track_history[track_id].append(center_x)
        
#         # Check if crossed lines (moved from left to right)
#         if len(self.track_history[track_id]) >= 2:
#             prev_x = self.track_history[track_id][-2]
#             curr_x = self.track_history[track_id][-1]
            
#             # Check start line crossing
#             if prev_x < start_line_x <= curr_x and track_id not in self.crossed_start_line:
#                 self.crossed_start_line.add(track_id)
#                 self._debug_log(f"Track {track_id} crossed START line at x={curr_x:.1f}")
#                 return "start_line"
            
#             # Check detection line crossing
#             if prev_x < line_x <= curr_x and track_id not in self.crossed_objects:
#                 self.crossed_objects.add(track_id)
#                 self._debug_log(f"Track {track_id} crossed DETECTION line at x={curr_x:.1f}")
#                 return "detection_line"
        
#         return None
    
#     def process_frame(self, frame):
#         """
#         Process a single frame and update metrics
#         CRITICAL FIX: Now uses IOU-based class assignment like the working script
#         """
#         height, width = frame.shape[:2]
        
#         # Run YOLO detection
#         results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
#         # CRITICAL FIX: Prepare detections with class info (like working script)
#         detections = []
#         detection_info = []  # Store (bbox, class_name, confidence)
        
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 conf = box.conf[0].cpu().numpy()
#                 cls = int(box.cls[0].cpu().numpy())
#                 class_name = result.names[cls]
                
#                 detections.append([x1, y1, x2, y2, conf])
#                 detection_info.append({
#                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                     'class': class_name,
#                     'confidence': float(conf)
#                 })
        
#         if self.enable_debug and len(detections) > 0:
#             self._debug_log(f"Found {len(detections)} detections")
        
#         # Update tracker
#         if len(detections) > 0:
#             detections_array = np.array(detections)
#             tracks = self.tracker.update(detections_array)
            
#             # CRITICAL FIX: Assign classes to tracks using IOU-based matching
#             for track in tracks:
#                 x1, y1, x2, y2, track_id = track
#                 track_id = int(track_id)
#                 track_bbox = [x1, y1, x2, y2]
                
#                 # Find best matching detection using IOU
#                 best_iou = 0
#                 best_match = None
                
#                 for det_info in detection_info:
#                     iou = self.calculate_iou(track_bbox, det_info['bbox'])
                    
#                     if iou > best_iou:
#                         best_iou = iou
#                         best_match = det_info
                
#                 # Update class if good match found (IOU > 0.3)
#                 if best_match and best_iou > 0.3:
#                     self.track_class[track_id] = best_match['class']
#                     self.track_class_history[track_id].append(best_match['class'])
#                     self.track_confidence_history[track_id].append(best_match['confidence'])
#         else:
#             tracks = self.tracker.update()
        
#         # Process tracks
#         for track in tracks:
#             x1, y1, x2, y2, track_id = track
#             track_id = int(track_id)
            
#             # CRITICAL FIX: Get stable class using voting mechanism
#             class_name = self.get_stable_class(track_id)
            
#             # Check for line crossing
#             line_crossed = self.check_line_crossing(track_id, [x1, y1, x2, y2], width)
            
#             # Handle start line crossing for stuck bags
#             if line_crossed == "start_line":
#                 if class_name == "bag_stuck_filled":
#                     if track_id not in self.stuck_bag_ids:
#                         self.stuck_bag_ids.add(track_id)
#                         self.stuck_bag_count += 1
#                         self.last_event_type = "stuck"
#                         self._debug_log(f"STUCK BAG detected! ID: {track_id}, Total stuck: {self.stuck_bag_count}")
            
#             # Handle detection line crossing (Counting Events)
#             elif line_crossed == "detection_line":
#                 # CRITICAL FIX: Only count if NOT a stuck bag
#                 if track_id not in self.stuck_bag_ids:
#                     if class_name == "bag_present":
#                         self.bag_present_count += 1
#                         self.total_events += 1
#                         self.last_event_type = "bag_present"
#                         self._debug_log(f"BAG PRESENT counted! ID: {track_id}, Total: {self.bag_present_count}")
#                     elif class_name == "no_bag":
#                         self.no_bag_count += 1
#                         self.total_events += 1
#                         self.last_event_type = "missed"
#                         self._debug_log(f"NO BAG counted! ID: {track_id}, Total: {self.no_bag_count}")
        
#         return frame
    
#     def get_summary(self):
#         """Get efficiency summary with Full Cycle calculation"""
#         elapsed_time = time.time() - self.start_time
        
#         # Calculate full machine rotations based on configured spouts
#         full_cycles = self.total_events / self.spouts if self.spouts > 0 else 0
#         actual_rpm = (full_cycles / (elapsed_time / 60)) if elapsed_time > 0 else 0
        
#         # Manual Efficiency: Success rate of placing bags on passing spouts
#         if self.total_events > 0:
#             manual_efficiency = (self.bag_present_count / self.total_events) * 100
#             dropped_efficiency = (self.no_bag_count / self.total_events) * 100
#         else:
#             manual_efficiency = dropped_efficiency = 0.0
        
#         # Packer Efficiency: Overall machine success including stuck bag downtime
#         total_operations = self.total_events + self.stuck_bag_count
#         if total_operations > 0:
#             packer_efficiency = ((self.bag_present_count + self.no_bag_count) / total_operations) * 100
#         else:
#             packer_efficiency = 0.0
        
#         return {
#             "total_events": self.total_events,  # Raw sensor hits (same as total_cycles in working script)
#             "total_cycles": round(full_cycles, 2),  # Calculated machine rotations
#             "bags_placed": self.bag_present_count,
#             "bags_missed": self.no_bag_count,
#             "stuck_bags": self.stuck_bag_count,
#             "packer_efficiency": round(packer_efficiency, 2),
#             "target_rpm": self.rpm,
#             "actual_rpm": round(actual_rpm, 2),
#             "manual_efficiency": round(manual_efficiency, 2),
#             "dropped_efficiency": round(dropped_efficiency, 2),
#             "elapsed_time": round(elapsed_time, 2)
#         }
    
#     def reset_metrics(self):
#         """Reset all metrics and tracking"""
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
#         self.start_time = time.time()
#         self._debug_log("Metrics reset")
   
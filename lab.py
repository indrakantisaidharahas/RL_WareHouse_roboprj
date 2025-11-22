
# visual_pipeline.py
# Run: py -3.10 visual_pipeline.py  (requires camera or use video file)


import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.spatial.distance import cdist
import warnings


# Optional: suppress warnings for clean output
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -----------------------
# Color Detector (Task 3.1)
# -----------------------
class ColorDetector:
   def __init__(self, hsv_lower=(0, 120, 70), hsv_upper=(10, 255, 255), min_area=300):
       self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
       self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
       self.min_area = min_area


   def detect(self, frame):
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
       mask = cv2.medianBlur(mask, 5)
       kernel = np.ones((5, 5), np.uint8)
       mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
       contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       boxes = []
       for cnt in contours:
           if cv2.contourArea(cnt) < self.min_area:
               continue
           x, y, w, h = cv2.boundingRect(cnt)
           boxes.append((x, y, x + w, y + h))
       return boxes, mask




# -----------------------
# YOLO Detector Class (Task 3.2)
# -----------------------
class YOLODetector:
   def __init__(self, weights="yolov8n.pt", device=None, conf_thres=0.35):
       self.model = YOLO(weights)
       if device:
           self.model.to(device)
       self.conf_thres = conf_thres
       self.class_names = getattr(self.model.model, "names", None)


   def process_frame(self, frame, filter_classes=None):
       # returns list of {'class','conf','bbox'}
       res = self.model(frame, conf=self.conf_thres, verbose=False)[0]
       detections = []
       for b in res.boxes:
           cls_id = int(b.cls.item()) if hasattr(b, "cls") else int(b.cls)
           name = self.class_names[cls_id] if self.class_names else str(cls_id)
           conf = float(b.conf.item()) if hasattr(b, "conf") else float(b.conf)
           xyxy = b.xyxy.cpu().numpy().ravel() if hasattr(b, "xyxy") else np.array(b.xyxy).ravel()
           x1, y1, x2, y2 = map(int, xyxy)
           if (filter_classes is not None) and (name not in filter_classes):
               continue
           detections.append({"class": name, "conf": conf, "bbox": (x1, y1, x2, y2)})
       return detections




# -----------------------
# Simple Multi-Object Tracker + Re-ID + Prediction (Task 3.3)
# -----------------------
class Track:
   def __init__(self, tid, bbox, classname, frame, max_hist=40):
       self.id = tid
       self.bbox = bbox
       self.class_name = classname
       self.hits = 1
       self.missed = 0
       self.trace = deque(maxlen=max_hist)
       self.trace.append(self.centroid(bbox))
       self.appearance = self.compute_hist(frame, bbox)
       self.vx, self.vy = 0.0, 0.0


   def centroid(self, bbox):
       x1, y1, x2, y2 = bbox
       return ((x1 + x2) // 2, (y1 + y2) // 2)


   def compute_hist(self, frame, bbox):
       x1, y1, x2, y2 = bbox
       h, w = frame.shape[:2]
       x1c, y1c = max(0, x1), max(0, y1)
       x2c, y2c = min(w - 1, x2), min(h - 1, y2)
       patch = frame[y1c:y2c, x1c:x2c]
       if patch.size == 0:
           return None
       hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
       hist = cv2.calcHist([hsv], [0, 1], None, [30, 30], [0, 180, 0, 256])
       cv2.normalize(hist, hist)
       return hist.flatten()


   def predict_next(self):
       cx, cy = self.trace[-1]
       return int(cx + self.vx), int(cy + self.vy)


   def update(self, bbox, frame):
       oldc = self.trace[-1]
       newc = self.centroid(bbox)
       self.vx = newc[0] - oldc[0]
       self.vy = newc[1] - oldc[1]
       self.bbox = bbox
       self.hits += 1
       self.missed = 0
       self.trace.append(newc)
       new_hist = self.compute_hist(frame, bbox)
       if new_hist is not None:
           self.appearance = new_hist




class SimpleMultiTracker:
   def __init__(self, max_missed=8, max_dist=80):
       self.tracks = {}
       self.next_id = 0
       self.max_missed = max_missed
       self.max_dist = max_dist


   def update(self, detections, frame):
       if len(self.tracks) == 0:
           for d in detections:
               self._create(d, frame)
           return


       track_ids = list(self.tracks.keys())
       track_centroids = np.array([self.tracks[t].trace[-1] for t in track_ids])
       det_centroids = (
           np.array([self._centroid(d["bbox"]) for d in detections])
           if detections
           else np.empty((0, 2))
       )


       if det_centroids.size == 0:
           for tid in track_ids:
               self.tracks[tid].missed += 1
           self._prune()
           return


       dists = cdist(track_centroids, det_centroids)
       app_cost = np.zeros_like(dists)
       for i, tid in enumerate(track_ids):
           t = self.tracks[tid]
           for j, det in enumerate(detections):
               # ✅ Fixed: use compute_hist (no underscore)
               det_hist = Track(-1, det["bbox"], det["class"], frame).compute_hist(
                   frame, det["bbox"]
               )
               if (t.appearance is None) or (det_hist is None):
                   app_cost[i, j] = 1.0
               else:
                   app_cost[i, j] = cv2.compareHist(
                       t.appearance.astype("float32"),
                       det_hist.astype("float32"),
                       cv2.HISTCMP_BHATTACHARYYA,
                   )


       if dists.size == 0:
           for d in detections:
               self._create(d, frame)
           self._prune()
           return


       d_norm = dists / (dists.max() + 1e-6)
       combined = 0.65 * d_norm + 0.35 * app_cost
       assigned_tracks = set()
       assigned_dets = set()


       while True:
           idx = np.unravel_index(np.argmin(combined, axis=None), combined.shape)
           i, j = idx
           if combined[i, j] > 0.9 or dists[i, j] > self.max_dist:
               break
           if i in assigned_tracks or j in assigned_dets:
               combined[i, j] = 1e6
               if combined.min() > 1e5:
                   break
               continue
           assigned_tracks.add(i)
           assigned_dets.add(j)
           tid = track_ids[i]
           self.tracks[tid].update(detections[j]["bbox"], frame)
           combined[i, :] = 1e6
           combined[:, j] = 1e6
           if combined.min() > 1e5:
               break


       for k, det in enumerate(detections):
           if k not in assigned_dets:
               self._create(det, frame)


       matched_tids = {track_ids[i] for i in assigned_tracks}
       for tid in list(self.tracks.keys()):
           if tid not in matched_tids:
               self.tracks[tid].missed += 1
       self._prune()


   def _create(self, det, frame):
       self.tracks[self.next_id] = Track(
           self.next_id, det["bbox"], det["class"], frame
       )
       self.next_id += 1


   def _prune(self):
       to_del = [tid for tid, t in self.tracks.items() if t.missed > self.max_missed]
       for tid in to_del:
           del self.tracks[tid]


   def get_tracks(self):
       out = []
       for tid, t in self.tracks.items():
           out.append(
               {
                   "id": tid,
                   "bbox": t.bbox,
                   "class": t.class_name,
                   "trace": list(t.trace),
                   "prediction": t.predict_next(),
               }
           )
       return out


   @staticmethod
   def _centroid(bbox):
       x1, y1, x2, y2 = bbox
       return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])




# -----------------------
# Demo + Obstacle Avoidance Hook
# -----------------------
def demo(source=0, weights="yolov8n.pt", filter_classes=None):
   cap = cv2.VideoCapture(source)
   det = YOLODetector(weights=weights)
   tracker = SimpleMultiTracker()
   color_detector = ColorDetector()


   prev = time.time()
   while True:
       ok, frame = cap.read()
       if not ok:
           break


       color_boxes, mask = color_detector.detect(frame)
       for bb in color_boxes:
           cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 2)


       detections = det.process_frame(frame, filter_classes=filter_classes)
       tracker.update(detections, frame)
       tracks = tracker.get_tracks()


       for t in tracks:
           x1, y1, x2, y2 = t["bbox"]
           cid = t["id"]
           cls = t["class"]
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
           cv2.putText(
               frame,
               f"ID:{cid}-{cls}",
               (x1, y1 - 6),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5,
               (255, 255, 255),
               1,
           )
           for i in range(1, len(t["trace"])):
               cv2.line(frame, tuple(t["trace"][i - 1]), tuple(t["trace"][i]), (200, 200, 0), 2)
           px, py = t["prediction"]
           cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)


       # Obstacle avoidance hook (visual only)
       safety_zone = (
           frame.shape[1] // 2 - 80,
           frame.shape[0] - 200,
           frame.shape[1] // 2 + 80,
           frame.shape[0],
       )
       cv2.rectangle(
           frame,
           (safety_zone[0], safety_zone[1]),
           (safety_zone[2], safety_zone[3]),
           (0, 0, 255),
           2,
       )


       should_brake = False
       for t in tracks:
           px, py = t["prediction"]
           if (safety_zone[0] <= px <= safety_zone[2]) and (
               safety_zone[1] <= py <= safety_zone[3]
           ):
               should_brake = True
               cv2.putText(
                   frame,
                   "BRAKE!",
                   (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.0,
                   (0, 0, 255),
                   2,
               )
               break


       now = time.time()
       fps = 1.0 / (now - prev + 1e-6)
       prev = now
       cv2.putText(
           frame, f"FPS:{fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
       )
       cv2.imshow("Perception", frame)
       if cv2.waitKey(1) & 0xFF == ord("q"):
           break


   cap.release()
   cv2.destroyAllWindows()




if __name__ == "__main__":
   demo(0, weights="yolov8n.pt", filter_classes=None) for runong this code what modules shuls i iinstall

import torch
import numpy as np
import cv2
import time
from ultralytics import RTDETR

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection


class DETRClass:
    def __init__(self, capture_index):
        self.capture_index = capture_index

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using: {self.device}")

        self.model = RTDETR('rtdetr-l.pt')
        self.class_names = self.model.model.names
        
        #####################################################

        self.max_cosine_distance = 0.4
        self.encoder_file_name   = 'mars-small128.pb'
        
        self.metric  = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance)
        self.tracker = Tracker(self.metric, max_age = 70, n_init = 5)
        self.encoder = generate_detections.create_box_encoder(self.encoder_file_name, batch_size=1)
        
        #####################################################
    
    def plot_boxes(self, results, frame):
        boxes = results[0].boxes.cpu().numpy()

        detections = []
        for box in boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box
            detections.append(
                [int(x1), int(y1), int(x2), int(y2), conf, int(class_id)]
            )

        bbox = np.asarray([detection[:4] for detection in detections])
        bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]

        scores = [conf[4] for conf in detections]
        class_ids = [class_id[5] for class_id in detections]
        
        features = self.encoder(frame, bbox)
        
        detection_objects = []
        for bbox_id, bbox in enumerate(bbox):
            detection_objects.append(Detection(bbox, scores[bbox_id], features[bbox_id], self.class_names[class_ids[bbox_id]]))
        
        self.tracker.predict()
        self.tracker.update(detection_objects)
        
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() and track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_name = track.class_name
            tracks.append([track_id, bbox, class_name])
            
            
        for track in tracks:
            track_id       = track[0]
            bbox           = track[1]
            x1, y1, x2, y2 = bbox
            class_name     = track[2]
            
            cv2.putText(frame, f"ID: {track_id} | {class_name}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


        return frame
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            results = self.model.predict(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)

            cv2.putText(frame, f'FPS: {fps:.2f}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("DETR", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    
transformer_detector = DETRClass("YOUR_PATH_GOES_HERE")
transformer_detector()
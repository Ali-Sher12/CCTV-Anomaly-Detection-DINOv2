import cv2
from ultralytics import YOLO
import Globals as gb

class YOLO_MODEL:

    def __init__(self):
        self.model_path = gb.YOLO_MODEL_VERSION
        self.model = YOLO(self.model_path)
        self.listOfItems = []

    def draw_overlay(self,final_frame):
        if self.listOfItems is None:
            return final_frame

        dark_yellow = (0, 204, 204)  # BGR format — cv2 uses Blue-Green-Red, not RGB
        for feat_list in self.listOfItems:
            x1 = int(feat_list[2])
            y1 = int(feat_list[3])
            x2 = int(feat_list[4])
            y2 = int(feat_list[5])        
            cv2.rectangle(final_frame, (x1, y1), (x2, y2), dark_yellow, 2)
            label = f"{feat_list[0]} {feat_list[1]:.2f}"
            if feat_list[6] != -1:
                label = f"{feat_list[0]}  {feat_list[1]*100:.1f}%  id:{feat_list[6]}"
            cv2.putText(final_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dark_yellow, 2)
        return final_frame

    def run_model(self,final_frame):
        self.listOfItems.clear()
        self.listOfItems = []
        results = self.model.track(final_frame,persist = True, verbose=False)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = -1
                if box.id is not None:
                    track_id = int(box.id[0])
                self.listOfItems.append([class_name,confidence,x1,y1,x2,y2,track_id])

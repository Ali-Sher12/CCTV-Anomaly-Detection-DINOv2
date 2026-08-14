import cv2
import tkinter as tk
import Accessories as Acc
import Globals as gb
from gui import AnomalyDetectionGUI
from DINO import DINO_MODEL as dino
#from YOLO import YOLO_MODEL as yolo

def main():

    dino_model = dino()
    cap = None
    ############# Tkinter Setup #############
    root = tk.Tk()
    gb.gui = AnomalyDetectionGUI(root)
    #########################################

    def setup_camera():
        nonlocal dino_model,cap

        # Release previous capture if re-opening
        if cap is not None:
            cap.release()

        ############# Video Capture Setup #############
        if gb.doVideoStream:
            gb.url = "http://10.13.12.117:8080/video"
        else:
            gb.url = 0

        cap = cv2.VideoCapture(gb.url)
        ###############################################

        gb.frameShape[0] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gb.frameShape[1] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        dino_model.camera_init()

        gb.gui.log("Camera initialized. Starting calibration...")

    setup_camera()

    def update_loop():
        nonlocal dino_model,cap

        if not gb.gui.is_running():
            return

        # Check if GUI requested a restart (Apply & Restart was clicked)
        if gb.gui.restart_requested:
            gb.gui.clear_restart_flag()
            setup_camera()
            root.after(50, update_loop)
            return

        if cap is None or not cap.isOpened():
            gb.gui.log("Camera not available. Retrying...")
            root.after(1000, update_loop)
            return

        frameRead, frame = cap.read()
        if not frameRead:
            if gb.doVideoStream:
                root.after(10, update_loop)
                return
            else:
                gb.gui.log("End of video stream.")
                return

        ##########     frame processing     ##########
        frame = cv2.flip(frame, 1)

        dino_model.getFrame(frame)
        processed_frame,overall_tier = dino_model.DINO_computation_loop()
        gb.gui.update_status(overall_tier)
        
        ##############################################
        gb.gui.update_frame(processed_frame)

        # Schedule the next frame; delay controls responsiveness
        root.after(gb.delay, update_loop)

    # Start the update loop
    root.after(100, update_loop)

    # Bind ESC to close
    root.bind("<Escape>", lambda e: gb.gui.on_closing())

    root.mainloop()

    if cap is not None:
        cap.release()

if __name__ == "__main__":
    main()

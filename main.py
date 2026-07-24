import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel

delay = 2 #milliseconds

url = "Assets/sample.mp4"
doVideoStream = True
frameShape = [640,480]
foreground_color = (0, 255, 0)#pure green
dead_zone_color = (0, 0, 0)

if __name__  == "__main__":

    #############   Model Setup    #############
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()
    ############################################


    #############    Mask Setup    #############
    mask_image = cv2.imread("Assets/Mask.png")    
    is_active = np.all(mask_image == foreground_color, axis=2)
    mask = is_active.astype(np.float32)
    mask = cv2.resize(mask, (frameShape[0], frameShape[1]), interpolation=cv2.INTER_NEAREST)    
    ############################################


    ############# Video Capture Setup #############
    if doVideoStream:
        url = "http://192.168.18.98:8080/video"

    cap = cv2.VideoCapture(url)
    ###############################################

    #loop
    while True:
        frameRead,frame = cap.read()
        if not frameRead:
            if doVideoStream:
                continue
            else:
                break

        ##########     frame processing     ##########
        Ignored_zone_done_mask = frame.copy()
        Ignored_zone_done_mask[mask==0] = 0
        rgb_frame = cv2.cvtColor(Ignored_zone_done_mask, cv2.COLOR_BGR2RGB)


        processed_frame = frame.copy()
        ##############################################
        
        cv2.imshow("Processed Stream", processed_frame)
        if cv2.waitKey(delay) == 27:
            break

cap.release()
cv2.destroyAllWindows()
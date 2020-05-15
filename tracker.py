import cv2
import numpy as np

def read_clip_rgb(path):
    cap = cv2.VideoCapture(path)
    clip_buf = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break 
        clip_buf.append(frame)
    return clip_buf, (width, height)

def read_clip_gray(path):
    cap = cv2.VideoCapture(path)
    clip_buf = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        clip_buf.append(frame)
    return clip_buf, (width, height)

class ball_tracker(object):
    def __init__(self, clip, trackerType, video_size):
        self.clip = clip
        self.size = video_size
        self.trackerType = trackerType
        self.save_video_name = 'output_5.avi'

    def tracker_init(self):
        if self.trackerType == 'MedianFlow':
            tracker = cv2.TrackerMedianFlow_create()
        elif self.trackerType == 'MOOSE':
            tracker = cv2.TrackerMOSSE_create()        
        return tracker
    
    def draw_ROI(self, frame):
        bboxes = []
        colors = []
        while True:
            bbox = cv2.selectROI('ROI', frame)
            bboxes.append(bbox)
            colors.append((0, 0, 255)) # ROI_color：red
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                cv2.destroyWindow('ROI')
                break
        return bboxes, colors
    
    def show_process_video(self, frame, bbox, colors):
        tracker = self.tracker_init()
        tracker.init(frame, bbox[0])
        for frame in self.clip:
            success, boxes = tracker.update(frame)
            #print(boxes)
            p1 = (int(boxes[0]), int(boxes[1]))
            p2 = (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3]))
            cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
            cv2.imshow('Tracker', frame)
            cv2.waitKey(40)
        cv2.destroyAllWindows()
    
    def save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.save_video_name, fourcc, 30, (self.size[0], self.size[1]))
        for frame in self.clip:
            out.write(frame)
        out.release()


if __name__ == '__main__':
    #path = ('./material/LHB_240FPS/Lin_toss_1227 (2).avi')
    path = ('./cam_1_13.avi')
    clip_buf, size = read_clip_rgb(path)
    ROI_frame = clip_buf[22]
    # 210 Lin_toss_1227 (2).avi
    # 22  cam_1_13.avi
    # 124 cam_1_15.avi      
    trackerType = "MedianFlow" 
    ball_tracker = ball_tracker(clip_buf, trackerType, size)
    bboxes, colors = ball_tracker.draw_ROI(ROI_frame)
    ball_tracker.show_process_video(frame, bboxes, colors)
    ball_tracker.save_video()
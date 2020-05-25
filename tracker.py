import cv2
import numpy as np
import queue
import csv
from sklearn.metrics.pairwise import cosine_similarity

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
        self.save_video_name = 'output_1.avi'

    def tracker_init(self):
        if self.trackerType == 'MedianFlow':
            tracker = cv2.TrackerMedianFlow_create()
        elif self.trackerType == 'MOOSE':
            tracker = cv2.TrackerMOSSE_create()   
        elif self.trackerType == 'CSRT':
            tracker = cv2.TrackerCSRT_create()     
        return tracker
    
    def draw_ROI(self, frame):
        bboxes = []
        colors = []
        while True:
            bbox = cv2.selectROI('ROI', frame, fromCenter=False, showCrosshair=True)
            bboxes.append(bbox)
            colors.append((0, 0, 255)) # ROI_color：red
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                cv2.destroyWindow('ROI')
                break
        return bboxes, colors

    def calculate_diff(self, pre_box, p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        center = (p1 + p2) / 2
        pre_p1 = np.array(pre_box[0], dtype=float)
        pre_p2 = np.array(pre_box[1], dtype=float)
        pre_center = (pre_p1 + pre_p2) / 2
        diff = np.sum(np.square(pre_center - center))
        return diff
        
    def first_half_video(self, ROI_frame, bbox, colors, select_frame):
        tracker = self.tracker_init()
        tracker.init(ROI_frame, bbox[0])
        clip = self.clip[select_frame:]
        i = select_frame
        roi = np.zeros((1,i), dtype=np.int8)
        roi_box = []
        pre_box = None
        while i > 0:
            success, boxes = tracker.update(self.clip[i-1])
            # box return (left top x, left top y, w, h)
            p1 = [int(boxes[0]), int(boxes[1])]
            p2 = [int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3])]
            if pre_box != None and self.calculate_diff(pre_box, p1, p2) > 7:
                roi[:,i-1] = 1 
            pre_box = (p1, p2)
            roi_box.append([p1, p2])    
            #cv2.rectangle(self.clip[i], p1, p2, colors[0], 2, 1)
            #cv2.imshow('Tracker', self.clip[i])
            #cv2.waitKey(30)
            i -= 1
        roi_box.reverse()
        indices = np.argwhere(roi == 1)
        indices = indices[:,1]
        roi[:,indices[0]:] = 1
        return roi, roi_box

    def last_half_viideo(self, ROI_frame, bbox, select_frame):
        tracker = self.tracker_init()
        tracker.init(ROI_frame, bbox[0])
        clip = self.clip[select_frame:]
        roi = np.zeros((1,len(clip)), dtype=np.int8)
        roi_box = []
        pre_box = None
        for i in range(len(clip)):
            frame = clip[i]
            success, boxes = tracker.update(frame)
            # box return (left top x, left top y, w, h)
            p1 = (int(boxes[0]), int(boxes[1]))
            p2 = (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3]))
            if pre_box != None and self.calculate_diff(pre_box, p1, p2) > 5:
                roi[:,i-1] = 1
            pre_box = (p1, p2)
            roi_box.append((p1, p2))
            #cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
            #cv2.imshow('Tracker', frame)
            #cv2.waitKey(30)       
        return roi, roi_box

    def continuous_cofident(self, roi_box, index):
        i = index
        points = np.array(roi_box[i:i+3], dtype=int) # shape:(3, 2, 2)
        p1 = points[:, 0, :] # shape:(3, 2)
        p2 = points[:, 1, :]
        center = (p1 + p2) / 2
        move = center[i+1, :] - center[i, :]
        next_move = center[i+2, :] - center[i+1, :]
        confident = cosine_similarity([move], [next_move]) # its value is between -1 and 1 
        return confident

    def check_continuous(self, roi_box, roi, check_type):
        # Remove the roi with low continuous (not ball)
        if check_type == 'first':
            roi_indices = np.argwhere(roi == 1)
            roi_indices = roi_indices[:,1]
            for i in roi_indices:
                if i + 2 < len(roi):
                    confident = self.continuous_cofident(roi_box, i)
                    if confident < 0.75:
                        roi[:,i] = 0
        # Add the roi with high continuous (ball)                
        elif check_type == 'last':
            noroi_indices = np.argwhere(roi == 0)
            noroi_indices = noroi_indices[:,1]
            for i in noroi_indices:
                if i + 2 < len(roi):
                    confident = self.continuous_cofident(roi_box, i)
                    if confident > 0.75:
                        roi[:,i] = 1
        return roi
       
    def show_process_video(self, ROI_frame, bbox, colors, select_frame):
        tracker = self.tracker_init()
        tracker.init(ROI_frame, bbox[0])
        first_roi, first_box = self.first_half_video(ROI_frame, bbox, colors, select_frame)
        first_roi = self.check_continuous(first_box, first_roi, 'first')
        last_roi, last_box = self.last_half_viideo(ROI_frame, bbox, select_frame)
        last_roi = self.check_continuous(last_box, last_roi, 'last')     
        for i in range(len(self.clip)):
            frame = self.clip[i]
            if i < select_frame:
                (p1, p2) = first_box[i]
                # if the roi == 1, the roi will be show on the video
                if first_roi[:, i] == 1:
                    p1 = (p1[0], p1[1])
                    p2 = (p2[0], p2[1])
                    cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
            else:
                p1, p2 = last_box.pop(0)
                #cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
                # if the roi == 1, the roi will be show on the video
                if last_roi[:, i - select_frame] == 1:
                    cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
            cv2.imshow('Tracker', frame)
            cv2.waitKey(20)
        
        cv2.destroyAllWindows()
        
    def save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.save_video_name, fourcc, 30, (self.size[0], self.size[1]))
        for frame in self.clip:
            out.write(frame)
        out.release()


if __name__ == '__main__':
    #path = ('./material/LHB_240FPS/Lin_toss_1227 (2).avi')
    path = ('./color/cam_7_965.avi')
    clip_buf, size = read_clip_rgb(path)
    select_frame = 595
    ROI_frame = clip_buf[select_frame]
    # 210 Lin_toss_1227 (2).avi
    # 22  cam_1_13.avi
    # 124 cam_1_15.avi
    # 595 cam_7_987.avi、cam_7_965.avi
    # Tracker type: MedianFlow, MOOSE, CSRT      
    trackerType = "CSRT" 
    ball_tracker = ball_tracker(clip_buf, trackerType, size)
    bboxes, colors = ball_tracker.draw_ROI(ROI_frame)
    ball_tracker.show_process_video(ROI_frame, bboxes, colors, select_frame)
    ball_tracker.save_video()
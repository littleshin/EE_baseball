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
        self.save_video_name = 'output.avi'

    def tracker_init(self):
        if self.trackerType == 'MedianFlow':
            tracker = cv2.TrackerMedianFlow_create()
        elif self.trackerType == 'MOOSE':
            tracker = cv2.TrackerMOSSE_create()   
        elif self.trackerType == 'CSRT':
            tracker = cv2.TrackerCSRT_create()     
        return tracker
    
    def set_roi_frame(self, roi_frame, select_frame):
        self.ROI_frame = roi_frame
        self.select_frame = select_frame
        
    def draw_ROI(self):
        self.bboxes = []
        self.colors = []
        while True:
            bbox = cv2.selectROI('ROI', self.ROI_frame, fromCenter=False, showCrosshair=True)
            self.bboxes.append(bbox)
            self.colors.append((0, 0, 255)) # ROI_color：red
            print("Press q to quit selecting boxes and start tracking")
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                cv2.destroyWindow('ROI')
                break

    def calculate_diff(self, pre_box, p1, p2):
        p1 = np.array(p1, dtype=int)
        p2 = np.array(p2, dtype=int)
        center = (p1 + p2) / 2
        pre_p1 = np.array(pre_box[0], dtype=int)
        pre_p2 = np.array(pre_box[1], dtype=int)
        pre_center = (pre_p1 + pre_p2) / 2
        diff = np.sqrt(np.sum(np.square(pre_center - center)))
        return diff
        
    def get_first_half_video_roi_and_box(self):
        tracker = self.tracker_init()
        tracker.init(self.ROI_frame, self.bboxes[0])
        clip = self.clip[self.select_frame:]
        i = self.select_frame
        roi = np.zeros((1,i), dtype=np.int8)
        roi_box = []
        pre_box = None
        finish = False
        while i > 0:
            success, boxes = tracker.update(self.clip[i-1])
            # box return (left top x, left top y, w, h)
            p1 = [int(boxes[0]), int(boxes[1])]
            p2 = [int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3])]
            if pre_box != None and finish == False:
                if self.calculate_diff(pre_box, p1, p2) > 5:
                    if self.calculate_diff(pre_box, p1, p2) > 20:
                        #roi[:, :i-1] = 0
                        roi[:, i-1:] = 1
                        finish = True
                    else:
                        roi[:, i-1] = 1
            pre_box = (p1, p2)
            roi_box.append([p1, p2])    
            #cv2.rectangle(self.clip[i], p1, p2, colors[0], 2, 1)
            #cv2.imshow('Tracker', self.clip[i])
            #cv2.waitKey(30)
            i -= 1
        roi_box.reverse()
        return roi, roi_box

    def get_last_half_video_roi_and_box(self):
        tracker = self.tracker_init()
        tracker.init(self.ROI_frame, self.bboxes[0])
        clip = self.clip[self.select_frame:]
        roi = np.ones((1,len(clip)), dtype=np.int8)
        roi_box = []
        pre_box = None
        for i in range(len(clip)):
            frame = clip[i]
            success, boxes = tracker.update(frame)
            # box return (left top x, left top y, w, h)
            p1 = (int(boxes[0]), int(boxes[1]))
            p2 = (int(boxes[0] + boxes[2]), int(boxes[1] + boxes[3]))
            if p1[0] < 0 or p1[1] < 0:
                roi[:, i-1:] = 0
            pre_box = (p1, p2)
            roi_box.append((p1, p2, success))
            #cv2.rectangle(frame, p1, p2, colors[0], 2, 1)
            #cv2.imshow('Tracker', frame)
            #cv2.waitKey(30)       
        return roi, roi_box

    def continuous_confident(self, roi_box, index, cf_num):
        i = index
        cf = cf_num  # check frame
        points = np.array(roi_box[i:i+cf], dtype=int) # shape:(5, 2, 2)
        p1 = points[:, 0, :] # shape:(5, 2)
        p2 = points[:, 1, :]
        center = (p1 + p2) / 2
        score = []
        for n in range(cf - 2):
            move = center[n+1, :] - center[n, :]
            next_move = center[n+2, :] - center[n+1, :]
            confident = cosine_similarity([move], [next_move])
            score.append(confident) 
        #move = center[i+1, :] - center[i, :]
        #next_move = center[i+2, :] - center[i+1, :]
        #confident = cosine_similarity([move], [next_move]) # its value is between -1 and 1 
        return sum(score) / len(score)

    def check_continuous(self, roi_box, roi, check_type):
        confident_threshold = 0.6
        cf_num = 5
        # Remove the roi with low continuous (not ball)
        if check_type == 'first':
            roi_indices = np.argwhere(roi == 1)
            roi_indices = roi_indices[:,1]
            for i in roi_indices:
                if i + cf_num - 1 < roi.shape[1]:
                    confident = self.continuous_confident(roi_box, i, cf_num)
                    print('Frame : ' + str(i) + ' confident : ' + str(confident))
                    if confident < confident_threshold:
                        roi[:,i] = 0
                        
        # Add the roi with high continuous (ball)
        elif check_type == 'last':
            roi_indices = np.argwhere(roi == 1)
            roi_indices = roi_indices[:,1]
            for i in roi_indices:
                if i + cf_num - 1 < len(roi):
                    confident = self.continuous_confident(roi_box, i, cf_num)
                    if confident < confident_threshold:
                        roi[:,i] = 0
        '''                
        elif check_type == 'last':
            noroi_indices = np.argwhere(roi == 0)
            noroi_indices = noroi_indices[:,1]
            for i in noroi_indices:
                if i + 2 < len(roi):
                    confident = self.continuous_confident(roi_box, i)
                    if confident > confident_threshold:
                        roi[:,i] = 1
        '''
    def check_success(self, last_roi, last_box):
        pre_box = None
        for i in range(10, len(last_box)):
            p1, p2, success = last_box[i]
            if pre_box != None and i + 4 < len(last_box):
                # check whether the box is stop and not track the ball or not
                if self.calculate_diff(pre_box, p1, p2) <= 1:
                    score = self.calculate_diff(pre_box, p1, p2)
                    check_pre_box = (p1, p2)
                    for j in range(i+1, i+6):
                        p1, p2, success = last_box[j]
                        score += self.calculate_diff(check_pre_box, p1, p2)
                        check_pre_box = (p1, p2)
                    score /= 6
                    if score <= 1:
                        last_roi[:, i:] = 0
                        break
            pre_box = (p1, p2)
       
    def show_process_video(self):
        tracker = self.tracker_init()
        tracker.init(self.ROI_frame, self.bboxes[0])
        first_roi, first_box = self.get_first_half_video_roi_and_box()
        first_type = 'first'
        #self.check_continuous(first_box, first_roi, first_type)
        last_roi, last_box = self.get_last_half_video_roi_and_box()
        self.check_success(last_roi, last_box)
        count_fail = 0       
        for i in range(len(self.clip)):
            frame = self.clip[i]
            if i < self.select_frame:
                (p1, p2) = first_box[i]
                #print('Frame :' + str(i) + ' center :(' + str((p1[0]+p2[0])/2) + ', ' + str((p1[1]+p2[1])/2) + ')')
                # if the roi == 1, the roi will be show on the video
                if first_roi[:, i] == 1:
                    #print('Frame :' + str(i) + ' center :(' + str((p1[0]+p2[0])/2) + ', ' + str((p1[1]+p2[1])/2) + ')')
                    p1 = (p1[0], p1[1])
                    p2 = (p2[0], p2[1])
                    cv2.rectangle(frame, p1, p2, self.colors[0], 2, 1)
            else:
                p1, p2, success = last_box.pop(0)
                if success == False:
                    count_fail += 1
                print('Frame :' + str(i) + ' center :(' + str((p1[0]+p2[0])/2) + ', ' + str((p1[1]+p2[1])/2) + ')')
                # if the roi == 1, the roi will be show on the video
                if last_roi[:, i - self.select_frame] == 1:
                    #print('Frame :' + str(i) + ' center :(' + str((p1[0]+p2[0])/2) + ', ' + str((p1[1]+p2[1])/2) + ')')
                    if count_fail < 6:
                        cv2.rectangle(frame, p1, p2, self.colors[0], 2, 1)
            cv2.imshow('Tracker', frame)
            cv2.waitKey(5)
        
        cv2.destroyAllWindows()
        
    def save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.save_video_name, fourcc, 30, (self.size[0], self.size[1]))
        for frame in self.clip:
            out.write(frame)
        out.release()


if __name__ == '__main__':
    #path = ('./material/LHB_240FPS/Lin_toss_1227 (2).avi')
    path = ('./color/cam_7_981.avi')
    clip_buf, size = read_clip_rgb(path)
    select_frame = 645
    ROI_frame = clip_buf[select_frame]
    # 210 Lin_toss_1227 (2).avi
    # 22  cam_1_13.avi
    # 124 cam_1_15.avi
    # 595 cam_7_987.avi, cam_7_965.avi, cam_7_970.avi
    # 645 cam_7_981.avi
    # Tracker type: MedianFlow, MOOSE, CSRT 
    trackerType = "CSRT" 
    ball_tracker = ball_tracker(clip_buf, trackerType, size)
    ball_tracker.set_roi_frame(ROI_frame, select_frame)
    ball_tracker.draw_ROI()
    ball_tracker.show_process_video()
    ball_tracker.save_video()
    
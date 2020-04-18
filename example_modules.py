import numpy  as np
import math
import cv2
import time

def read_clip_mono(path):
    cap = cv2.VideoCapture(path)
    clip_buf=[]
    rate = 1.5 # enhance the pixel value
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break 
        mono = frame[:,:,1] * rate
        mono[mono > 255] = 255
        clip_buf.append(mono)
    return clip_buf

def read_clip_rgb(path):
    cap = cv2.VideoCapture(path)
    clip_buf=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break 
        clip_buf.append(frame)
    return clip_buf

class MovingBallDetector(object):
    def __init__(self, frame, hist=8, thres=16, kr=7):
        self.WINDOW_NAME = "Example image"
        self.roi = self.cut_roi(frame)
        self.H, self.W = self.roi.shape 
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=thres, detectShadows=False) 
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kr,kr))  
        blob_params = self.set_blob_params()
        self.blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    def cut_roi(self, img):
        return img[0:540,0:720]

    def gen_differential_img(self, frame, mog=False):
        fgmask = self.fgbg.apply(frame)
        #kernel = np.ones((1,1), np.uint8) 
        if mog:
            fgmask_mog2 = fgmask
            fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_CLOSE, self.kernel) 
            fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_OPEN, self.kernel)
            #fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_OPEN, kernel)
            #fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_CLOSE, kernel) 
            return fgmask_mog2
        return fgmask
    
    def set_blob_params(self):
        ball_r = 10 
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 100
        # Filter by Area.
        params.filterByArea = True  # radius = 20~30 pixels (plate width = 265 pixels)
        params.minArea = ball_r*ball_r*math.pi *0.5 #    
        params.maxArea = ball_r*ball_r*math.pi *1.8 # 
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.6
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.6
        return params

    def draw_blob_detected_ball_on_img(self, img):
        inv_img = cv2.bitwise_not(img)
        keypoints = self.blob_detector.detect(inv_img)
        img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        '''
        for kp in keypoints:
            print("kp.size="+str(kp.size))
            print("kp.xy = (%d, %d)"%(kp.pt[0], kp.pt[1] ))
        '''
        return img ,keypoints 

    def demo_video(self, clip):
        for i, frame in enumerate(clip):
            fgmask = self.gen_differential_img(frame, mog=True)
            blob = self.draw_blob_detected_ball_on_img(fgmask)
            blob = blob[:,120:-140] # size : [540,460,3]
            cv2.imshow(self.WINDOW_NAME, blob)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.waitKey(0)
    def get_ball_xy(self,keypoints):
        points_xy = []
        for kp in keypoints:
            points_xy.append( (kp.pt[0], kp.pt[1]) )
        return points_xy

    def choose_ball(self,ball_xy,ball_detect):
        # if there are two or more detection of ball, choose the nearest of the last frame
        pre_ball = np.array(ball_detect[-1])
        balls = np.array(ball_xy)
        h,w = balls.shape
        pre_ball = np.repeat(pre_ball,h,axis=0)
        distance = np.sum( np.abs(balls - pre_ball),axis=0)
        pos = np.argmin(distance)
        balls = balls.astype(np.int32)
        return [(balls[pos,0],balls[pos,1])]


    def get_process_video(self,clip):
        process_video = []
        hit_frame = []
        ball_detect = []
        hit_time = 200
        for i, frame in enumerate(clip):
            fgmask = self.gen_differential_img(frame, mog=True)
            fgmask = cv2.medianBlur(fgmask, 1) # reduce the noise
            fgmask = fgmask[:,50:-130] # size : [540,540,3]
            blob, keypoints = self.draw_blob_detected_ball_on_img(fgmask)
            if keypoints != [] and i > hit_time:
                ball_xy = self.get_ball_xy(keypoints)
                if len(ball_xy) > 1:
                    ball_xy = self.choose_ball(ball_xy,ball_detect)
                ball_detect.append(ball_xy)

                if len(ball_detect) > 2:
                    hit = self.check_hit(ball_detect)
                    if hit:
                        hit_frame.append(i)
            process_video.append(blob)
        return process_video, hit_frame

    def check_hit(self,ball_detect):
        ball_1 = ball_detect[-3][0]
        ball_2 = ball_detect[-2][0]
        ball_3 = ball_detect[-1][0]
        ball_is_going_left_up = (ball_3[0] < ball_2[0] and ball_3[1] < ball_2[1]) \
            and (ball_2[0] < ball_1[0] and ball_2[1] < ball_1[1])
        ball_is_going_left_down = (ball_3[0] > ball_2[0] and ball_3[1] < ball_2[1]) \
            and (ball_2[0] > ball_1[0] and ball_2[1] < ball_1[1])
        hit = ball_is_going_left_up or ball_is_going_left_down
        return hit

        

def run_param_for_bgs(pause_mode):
    path_1 = ('./videos/Tang_toss_0101.avi')
    path_2 = ('./videos/Lin_toss_1227 (2).avi')
    clip_buf_1 = read_clip_mono(path_1)
    clip_buf_2 = read_clip_mono(path_2)
    frame_total_1= len(clip_buf_1)
    frame_total_2= len(clip_buf_2)

    thres = 10
    hist = 64
    hit_theshold = 0
    t0=time.time()
    print("hist = "+str(hist), "thres = "+str(thres))
    ball_detector_1 = MovingBallDetector(clip_buf_1[0],hist=hist, thres=thres, kr=3)
    ball_detector_2 = MovingBallDetector(clip_buf_2[0],hist=hist, thres=thres, kr=3)
    video_1, hit_frame_1 = ball_detector_1.get_process_video(clip_buf_1)
    video_2, hit_frame_2 = ball_detector_2.get_process_video(clip_buf_2)

    # print the hit frame of the video_n
    print("The hit frames of video 1: ", hit_frame_1)
    print("The hit frames of video 2: ", hit_frame_2)
    
    # Calculate the difference of two video and convert the process video to normal video
    if hit_frame_1[0] > hit_frame_2[0]:
        different = hit_frame_1[0] - hit_frame_2[0]
        video_1 = read_clip_rgb(path_1)
        video_1 = video_1[different-1:]
        video_2 = read_clip_rgb(path_2)
    else:
        different = hit_frame_2[0] - hit_frame_1[0]
        video_2 = read_clip_rgb(path_2)
        video_2 = video_2[different-1:]
        video_1 = read_clip_rgb(path_1)

    less = min(len(video_1), len(video_2))
    # playing the video 
    for i in range(less):
        hmerge = np.hstack((video_1[i],video_2[i]))
        cv2.namedWindow("Two video merge",0)
        cv2.resizeWindow("Two video merge", hmerge.shape[1], hmerge.shape[0])
        cv2.imshow('Two video merge', hmerge)       
        if pause_mode:
            if i in hit_frame_1 or i in hit_frame_2:
                cv2.waitKey(0)
            else:
                cv2.waitKey(40)
        else:
            cv2.waitKey(40)
  
# pause_mode = True : the video will pause at the hitting frame
# pause_mode = False : the video will not be pause
run_param_for_bgs(pause_mode=False)
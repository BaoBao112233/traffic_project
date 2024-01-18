
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import json
import time
from threading import Thread

# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--Width_video',help='Width_frame video' , default= 1280) 
parser.add_argument('--Height_video',help='Width_frame video' , default= 720) 
parser.add_argument('--video', help='Name of the video file',default='4K_road_traffic.mp4')
parser.add_argument('--path_save_json', help='Path save polygon jon',default="polygon.json")

args = parser.parse_args()

class VideoStream:
    """Camera object that controls video streaming"""
    def __init__(self,resolution=(640,480),framerate=30,STREAM_URL=''):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(STREAM_URL)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Open video file
VIDEO_PATH = 'rtsp://admin2:Atlab123@@192.168.1.64:554/Streaming/Channels/101'


VIDEO_NAME = args.video
WIDTH_VIDEO = int(args.Width_video)
HEIGHT_VIDEO = int(args.Height_video)

# Initialize video stream
videostream = VideoStream(resolution=(WIDTH_VIDEO,HEIGHT_VIDEO),framerate=18,STREAM_URL= VIDEO_PATH).start()
time.sleep(1)

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)


# Open video file
# video = cv2.VideoCapture(VIDEO_PATH)

# click left mouse to add point for polygon of dict points left and right 
def handle_point_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points['left'].append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        points['POINT_LEFT'] = [x, y]

# draw polygon left and right
def draw_polygon (frame, points):
    for point in points['left']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (255,0,0), -1)
    
    for point in points['right']:
        frame = cv2.circle( frame, (point[0], point[1]), 3, (0,255,0), -1)

    frame = cv2.polylines(frame, [np.int32(points['left'])], False, (255,0, 0), thickness=1)
    frame = cv2.polylines(frame, [np.int32(points['right'])], False, (0,255, 0), thickness=1)
    return frame


POINTS = {}
POINTS["area"]= []
POINTS['left'] =  []
POINTS['right'] =  []
POINTS['POINT_RIGHT']= [0,0]
POINTS['POINT_LEFT']= [0,0]
point_medial = []

print("draw right road \n")
while(True):

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = videostream.read()

    # cv2.imshow("Frame",frame)

    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (WIDTH_VIDEO, HEIGHT_VIDEO))
    # input_data = np.expand_dims(frame_resized, axis=0)

    # Ve ploygon
    frame = draw_polygon(frame_resized, POINTS)
    frame = cv2.circle( frame, (POINTS['POINT_RIGHT'][0], POINTS['POINT_RIGHT'][1]), 5, (0,255,255), -1)
    frame = cv2.circle( frame, (POINTS['POINT_LEFT'][0], POINTS['POINT_LEFT'][1]), 5, (255,0,255), -1)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame_resized)

    cv2.setMouseCallback('Object detector', handle_point_click, POINTS)
    # cv2.setMouseCallback('Object detector', handle_right_click, POINTS['POINT'] )

    if cv2.waitKey(1) == ord('a'):
        POINTS['area'] = POINTS['left']
        POINTS['left'] = []
        
    if cv2.waitKey(1) == ord('p'):
        POINTS['POINT_RIGHT'] = POINTS['POINT_LEFT']

    # Press 'q' to quit
    if cv2.waitKey(5) == ord('q'):
        break
    
    if cv2.waitKey(2) == ord('t'):
        print("draw left road \n")
        POINTS['right'] = POINTS['left'] 
        POINTS['left'] = []

# Clean up
cv2.destroyAllWindows()
videostream.stop()

print('POINT LEFT : ',POINTS['left'])
print('\n')
print('POINT RIGHT : ',POINTS['right'])

POINTS['size_width'] = WIDTH_VIDEO
POINTS['size_height'] = HEIGHT_VIDEO
print('\n')
print("SIZE SCREEN : ",POINTS['size_width'] ,POINTS['size_height'] )

# # Specify the file path where you want to save the JSON data

JSON_PATH = args.path_save_json

# Write the dictionary to a JSON file
with open(JSON_PATH, 'w') as json_file:
    json.dump(POINTS, json_file)
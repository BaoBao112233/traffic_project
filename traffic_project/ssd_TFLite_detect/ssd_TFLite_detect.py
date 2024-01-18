
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from ssdDetect import polygon_calculate
import time
from threading import Thread


# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
parser.add_argument('--Json_path', help='Path file polygon json',
                    default='polygon.json')                   
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu
JSON_PATH = args.Json_path


# Create a tracker based on tracker name
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):

    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
    return tracker

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


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   
  
# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# path json polygon
JSON_PATH = os.path.join(CWD_PATH,JSON_PATH)
print("JSON path : ",JSON_PATH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5
limit_area = 7000

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
# Open video file
VIDEO_PATH = 'rtsp://admin2:Atlab123@@192.168.1.64:554/Streaming/Channels/101'

imW,imH = 1280,720
# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=25,STREAM_URL= VIDEO_PATH).start()
time.sleep(1)



# video = cv2.VideoCapture(VIDEO_PATH)
# imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
# imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Get Polygon_calculate
polygon_cal = polygon_calculate(JSON_PATH,imW,imH)

# detect frame return boxes
def detect_ssd(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    boxes_new = []
    classes_new = []
    scores_new = []
    centroid_new = []
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            
            # scale boxes - values (0,1) to size witgh hight
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            if(polygon_cal.area_box((xmin,ymin,xmax,ymax),limit_area)):
                centroid_new.append([int((xmin+xmax)//2),int((ymin+ymax)//2)])
                boxes_new.append((xmin,ymin,xmax,ymax))
                classes_new.append(classes[i])
                scores_new.append(scores[i])
    return boxes_new,classes_new,scores_new,centroid_new

# ret, frame = video.read()

boxes, classes,scores ,centroids_old = [],[],[],[]

trackerType = trackerTypes[4]  
# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
# for bbox in boxes:
#     multiTracker.add(createTrackerByName(trackerType), frame, bbox)

count = 0
num_frame_to_detect = 5

while(True):
    t1 = cv2.getTickCount()
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = videostream.read()
    # if not ret:
    #   print('Reached the end of the video!')
    #   break
    
    _, frame = polygon_cal.cut_frame_polygon(frame)

    # get updated location of objects in subsequent frames
    success, boxes_update = multiTracker.update(frame)

    # print("Update frame , len boxes_update {}".format(len(boxes_update)))
    for i, newbox in enumerate(boxes_update):
        # print("new box : {} ".format(newbox))
        p1 = (int(newbox[0]),int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame,p1 , p2, (10, 255, 0), 2)


    if count == num_frame_to_detect:
        controids = polygon_cal.centroid(boxes_update)
        polygon_cal.draw_point_check(frame, controids)
        frame = polygon_cal.write_points_title(controids,centroids_old,frame)
        count = 0
        # print("wait reset count")
            # Calculate framerate


    if count == 0:
        start_time=time.time()
        boxes, classes,scores,centroids_old = detect_ssd(frame)
        # print("boxes :",boxes)
        # Create MultiTracker object
        multiTracker = cv2.MultiTracker_create()
        # Initialize MultiTracker
        for bbox in boxes:
            box_track = (bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])
            multiTracker.add(createTrackerByName(trackerType), frame, box_track)
            cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (10, 255, 0), 2)

        # print("detect frame and wait !  {}".format(3000))    
        if len(scores)==0:
            count=-1

    frame = polygon_cal.draw_polygon(frame)
    frame = cv2.circle( frame, (polygon_cal.points['right_check'][0], polygon_cal.points['right_check'][1]), 5, (0,255,255), -1)
    frame = cv2.circle( frame, (polygon_cal.points['left_check'][0], polygon_cal.points['left_check'][1]), 5, (255,0,255), -1)
        # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    frame = cv2.resize(frame, (2048, 1080))

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector 1', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    if count == num_frame_to_detect:
        cv2.waitKey(1)

    count+=1
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
# video.release()
# cv2.destroyAllWindows()

# Clean up
cv2.destroyAllWindows()
videostream.stop()
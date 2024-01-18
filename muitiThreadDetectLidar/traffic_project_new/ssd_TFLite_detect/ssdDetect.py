import cv2
import numpy as np
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from random import randint
from shapely.geometry import box

class polygon_calculate():
    def __init__(self,path_json,width_screen,height_screen):
        self.points = {}
        self.points['area'] = []
        self.points['right'] = []
        self.points['left'] = []
        self.points['right_check'] = 0
        self.points['left_check'] = 0
        self.width_screen = width_screen
        self.height_screen = height_screen

        self.load_points_form_json(path_json,width_screen,height_screen)
        
        self.tracker_type = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    def load_points_form_json(self, path_json,width_new,height_new):
        try:
            with open(path_json) as json_file:
                data = json.load(json_file)
            width = data['size_width']
            height = data['size_height']
            widght_scale = width_new/width
            height_scale = height_new/height
            
            self.points['area'].clear()
            self.points['left'].clear()
            self.points['right'].clear()
            self.points['right_check'] = [int(data['POINT_RIGHT'][0] * widght_scale),int(data['POINT_RIGHT'][1]*height_scale)]
            self.points['left_check'] = [int(data['POINT_LEFT'][0] * widght_scale),int(data['POINT_LEFT'][1]*height_scale)]
            for i in data['area'] :
                self.points['area'].append([int(i[0]*widght_scale),int(i[1]*height_scale)])
            self.points['area'] = np.array(self.points.get("area", [])).reshape((-1,1,2))
            for i in data['left'] :
                self.points['left'].append([int(i[0]*widght_scale),int(i[1]*height_scale)])
            for i in data['right'] :
                self.points['right'].append([int(i[0]*widght_scale),int(i[1]*height_scale)])
            print("Load polygon success")
        except:
            print("Error: path json file is not exist")
        
    def area_box(self,bbox,limit_area):
        rectangle = box(bbox[0],bbox[1],bbox[2],bbox[3])
        return rectangle.area < limit_area

    # draw point check
    def draw_point_check(self,frame, point_check):
        for point in point_check:
            frame = cv2.circle( frame, (point[0], point[1]), 5, (65,33,1), -1)
    
    # draw polygon left and right
    def draw_polygon (self,frame):
        points = self.points
        for point in points['left']:
            frame = cv2.circle( frame, (point[0], point[1]), 3, (255,0,0), -1)
        
        for point in points['right']:
            frame = cv2.circle( frame, (point[0], point[1]), 3, (0,255,0), -1)

        frame = cv2.polylines(frame, [np.int32(points['left'])], False, (255,0, 0), thickness=2)
        frame = cv2.polylines(frame, [np.int32(points['right'])], False, (0,255, 0), thickness=2)
        return frame
    
    # draw tracker object
    def draw_tracker(self,frame,boxes,colors):
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        return frame
    # check point inside polygon left and right or outside polygon
    def isInside(self,points,centroid):
        polygon = Polygon(points)
        centroid = Point(centroid)
        # print(polygon.contains(centroid))
        return polygon.contains(centroid) 

    # calculate controid for box_update tracker
    def centroid(self,bboxes):
        controids = []
        for box in bboxes:
            x, y, w, h = map(int, box)
            controids.append([x + w//2, y + h//2])
        return controids

    # calculate centroid for box detect ,model

    def centroid_dt(self,bboxes):
        controids = []
        for box in bboxes:
            xmin, ymin, xmax, ymax = map(int, box)
            controids.append([int((xmin+xmax)//2),int((ymin+ymax)//2)])         
        return controids



    def write_points_title(self,points,points_old,frame):
        polygon = self.points

        for i in range(len(points)):
            point_Infor = {}
            point_old = points_old[i]
            point_new = points[i]
            if self.isInside(polygon['left'], point_new):
                point_Infor['location'] = 'left'
                if self.distance(point_new,point_old,self.points['left_check']):
                    point_Infor['direction'] = True
                    # frame = self.alert(frame,"Keep going to left",point_new)
                else:
                    point_Infor['direction'] = False
                    frame = self.alert(frame,point_Infor,point_new)
            elif self.isInside(polygon['right'], point_new):
                point_Infor['location'] = 'right'
                if self.distance(point_new,point_old,self.points['right_check']):
                    point_Infor['direction'] = True
                    # frame = self.alert(frame,"Keep going to right",point_new)
                else:
                    point_Infor['direction'] = False
                    frame = self.alert(frame,point_Infor,point_new)
            else:
                point_Infor['location'] = 'outside'
                point_Infor['direction'] = True
                frame = self.alert(frame,point_Infor,point_new)

        return frame
    

    def check_result(self,points,points_old):
        polygon = self.points
        point_Infor = {
            "Left":False,
            "Right":False,
            "Forbidden_left":False,
            "Forbidden_right":False,
            "freeze" :False
        }
        for i in range(len(points)):
            
            point_old = points_old[i]
            point_new = points[i]
            if self.isInside(polygon['left'], point_new):
                point_Infor['Left'] = True
                if self.distance(point_new,point_old,self.points['left_check']) > 0:
                    point_Infor['Forbidden_left'] = True
                elif self.distance(point_new,point_old,self.points['left_check']) ==0:
                    point_Infor['freeze'] = True
            elif self.isInside(polygon['right'], point_new):
                point_Infor['Right'] = True
                if self.distance(point_new,point_old,self.points['right_check'])>0:
                    point_Infor['Forbidden_right'] = True
                elif self.distance(point_new,point_old,self.points['right_check'])==0 :
                    point_Infor['freeze'] = True
            # print('KC : ',,"\n")
        return point_Infor


    def alert(self, frame,point_Infor,point):
        if point_Infor['location'] != 'outside':
            if not point_Infor['direction'] :
                frame = cv2.putText(frame, "Forbidden direction", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                frame = cv2.putText(frame, "OK", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else:
            frame = cv2.putText(frame, "Outside", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        return frame
    
    # function calculate distance between 2 point
    def distance(self,point_new, point_old,point_check):
        point_new =  Point(point_new) 
        point_old =  Point(point_old)
        point_check= Point(point_check)
        return (point_check.distance(point_old) - point_check.distance(point_new))
    
    def cut_frame_polygon(self, frame):
        # Tạo một mask đen với kích thước bằng với frame
    
        mask = np.zeros_like(frame)


        # print("area:",[self.points['area']])
        # Vẽ đa giác trắng lên mask
        cv2.fillPoly(mask, [self.points['area']],(255, 255, 255))
        # cv2.fillPoly(mask, [polygon_right], (255, 255, 255))

        # Áp dụng mask để cắt frame
        result = cv2.bitwise_and(frame, mask)

        return frame, result
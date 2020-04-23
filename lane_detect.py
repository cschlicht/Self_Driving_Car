import numpy as np
import cv2
import matplotlib.pyplot as plt 
import warnings 
import time
import math

import Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi


motor.forward()



    

       
#def main():s


##was main but changed 


#def detect_lane(frame):
    
    #edges = Detect_Edges(frame)
    #cropped_edges = Cut_top_half(edges)
    #line_segments =Detect_line_segment(cropped_edges)
    #lane_lines =  Avg_slope(line_segments,frame)

    #return lane_lines


        

    
def Cut_top_half(edges):
    height, width = np.shape(edges)
    mask = np.zeros_like(edges)


    #delete top hald of screen 
    poly = np.array ([[(0,height*1/2),(width,height*1/2),(width,height),(0,height)]],np.int32)

    #fill array poly with mask 
    cv2.fillPoly(mask, poly, 255)

    
    cropped_edges = cv2.bitwise_and(edges, mask)
    



    return cropped_edges

def Detect_line_segment(cropped_edges):
    rho = 1 #distance precision in pixel
    angle = np.pi / 180 
    min_threshold = 20 
    min_line_length = 25
    max_line_gap = 50
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), min_line_length, max_line_gap)
    
    print(line_segments)
         
    return line_segments



def Avg_slope(line_segments,frame):
    #if slope is > 0 then on left
    #if slope is < 0 on the right 

    height, width,_ = np.shape(frame)
    right_fit = []
    left_fit = []
    lane_lines = []

    if line_segments is None:
        return 'no line seg'

    bound = 1/3

    left_boundary = width*(1 -bound)
    right_boundary =  width*bound

    for line_segment in line_segments:
        x1,y1,x2,y2 = line_segment[0]
        fit = np.polyfit((x1,x2),(y1,y2),1)
        slope = fit[0]
        intercept = fit[1]

        #lines on left postive slope 
        #lines on right have positive slope

        if slope < 0:
            if x1 < left_boundary and x2 < left_boundary:
                left_fit.append((slope,intercept))
        else:
            if x1 > right_boundary and x2 > right_boundary:
                right_fit.append((slope,intercept))

    left_fit_avg = np.average(left_fit,axis =0)
    if len(left_fit) > 0:
        lane_lines.append(Make_points(frame,left_fit_avg))

    right_fit_avg = np.average(right_fit,axis=0)
    if len(right_fit) > 0:
        lane_lines.append(Make_points(frame,right_fit_avg))
    print(lane_lines)

    return lane_lines

def Make_points(frame,line_parameters):
    height, width, _ = np.shape(frame)
    slope, intercept = line_parameters
    y1 = height 
    y2 = int(y1*(3/5))

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return np.array ([x1,y1,x2,y2])






def Detect_Edges(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_blue = np.array([60,40,40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    #cv2.imshow('mask',mask)

    edges = cv2.Canny(mask, 200, 400)
    
    

    return edges


def display_lines(frame,lines,line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)  
            cv2.line(line_image, (x1,y1),(x2,y2),line_color,line_width)

    #addweight() blends two images 
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return line_image

def detect_two_lines(frame, lane_lines):
    height, width, _ = np.shape(frame)
    _,_,left_x2,_ = lane_lines[0]
    _,_,right_x2,_ = lane_lines[1]

    print(left_x2,right_x2)
    mid = int(width/2)
    xoffset = (left_x2 + right_x2) / 2 - mid 
    yoffset = int(height)/2

    angle_to_mid_radian = math.atan(xoffset / yoffset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    print(steering_angle)

    return steering_angle

def detect_one_line(frame,lane_lines):
    x1,_,x2,_ = lane_lines[0][0]
    x_offset = x2 - x1
    y_offset = int(height/2)
    

def display_middle_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
	heading_image = np.zeros_like(frame)
	height, width, _  = np.shape(frame)


	steering_angle_radian = steering_angle/180.0 * math.pi
	x1 = int(width / 2)
	y1 = height
	x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
	y2 = int(height / 2)

	cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
	heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

	return heading_image

def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lanes=5, max_angle_deviation_one_lane=1):

	if num_of_lane_lines == 2:
		max_angle_deviation = max_angle_deviation_two_lanes
	else:
		max_angle_deviation = max_angle_deviation_one_lane

	angle_deviation = new_steering_angle - curr_steering_angle
	if abs(angle_deviation) > max_angle_deviation:
		stabilize_steering_angle = int(curr_steering_angle + max_angle_deviation * angle_deviation / abs(angle_deviation))
	else:
		stabilized_steering_angle = new_steering_angle

	return stabilized_steering_angle


    
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    _,frame = cap.read()
    
    warnings.simplefilter('ignore', np.RankWarning)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    edges = Detect_Edges(frame)
    cropped_edges = Cut_top_half(edges)
    line_segments =Detect_line_segment(cropped_edges)
    lane_lines =  Avg_slope(line_segments,frame)
    lane_lines_image = display_lines(frame,lane_lines)
    
    steering_angle = detect_two_lines(frame, lane_lines)
    heading_image = display_middle_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5)



    cv2.imshow("heading_image",heading_image)
    
    cv2.imshow("lane lines", lane_lines_image)
    cv2.imshow("Canny",cropped_edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

#if __name__ == "__main__":
    #main()





    

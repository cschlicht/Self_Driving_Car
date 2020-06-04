import numpy as np
import cv2
import matplotlib.pyplot as plt 
import warnings 
import time
import math
import motor_main as motor
import car_dir as car
import RPi.GPIO as GPIO
import PCA9685 as p
import PCA9685 as servo
#import Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi



'''
Function to use to turn
#car.turn(angle)
'''
    

       
#def main():s


##was main but changed 


#def detect_lane(frame):
    
    #edges = Detect_Edges(frame)
    #cropped_edges = Cut_top_half(edges)
    #line_segments =Detect_line_segment(cropped_edges)
    #lane_lines =  Avg_slope(line_segments,frame)

    #return lane_lines


'''

Detect_Edges: This 


    Parameters: 




    Return:



'''
car.setup()
motor.setup()

def Detect_Edges(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_blue = np.array([60,40,40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    #cv2.imshow('mask',mask)

    edges = cv2.Canny(mask, 200, 400)
    
    

    return edges


        

'''
Cut_top_half: The purpose of this function is to be able to create an area of intrest 
    Parameters:
    edges         -> returned from canny function which detects all edges in the frame

    Return:
    cropped_edges -> returns the canny image but with the top half cut off to making the bottom half the area of intrest

'''    
def Cut_top_half(edges):
    height, width = np.shape(edges)
    mask = np.zeros_like(edges)


    #create point of intrest as a triangle 
    poly =  polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)


    #fill array poly with mask 
    cv2.fillPoly(mask, poly, 255)

    
    cropped_edges = cv2.bitwise_and(edges, mask)
    


    return cropped_edges

'''
Detect_line_segment: Purpose of this function is to detect all the line segments using HoughlinesP function 
    Parameters:
        cropped_edges   -> area of intrest + canny image 
    Return:
        line_segments   -> returns the values of all line segments found in the area of intrest
    

'''
def Detect_line_segment(cropped_edges):
    rho = 1 #distance precision in pixel
    angle = np.pi / 180
    min_threshold = 50
    min_line_length = 50
    max_line_gap = 150 
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), min_line_length, max_line_gap)
    
    
    return line_segments

'''
Avg_slope:
    Parameters:


    Return:

'''

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
        for x1,y1,x2,y2 in line_segment:
            if (x1==x2):
                #logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                return 'not found'
            #x1,y1,x2,y2 = line_segment[0]
            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]

            #lines on left postive slope 
            #lines on right have positive slope

            if (slope < 0):
                if (x1 < left_boundary and x2 < left_boundary):
                    left_fit.append((slope,intercept))
            else:
                if (x1 > right_boundary and x2 > right_boundary):
                    right_fit.append((slope,intercept))

    left_fit_avg = np.average(left_fit,axis =0)
    #print("left avg",left_fit_avg)
    if (len(left_fit) > 0):
        lane_lines.append(Make_points(frame,left_fit_avg))

    right_fit_avg = np.average(right_fit,axis=0)
    #print("right avg",right_fit_avg)
    if (len(right_fit) > 0):
        lane_lines.append(Make_points(frame,right_fit_avg))
        
    print(lane_lines)
    #time.sleep(0.5)
    return lane_lines

'''
Make_points:
    Parameters:
        frame            ->  is each individual frame caprtured from the video 
        line_parameters  ->  this is the line that will be inputed to find it coordinates 
    Return:
        np.array         ->  returns a X1,Y1,X2,Y2 value from the line that was given 


'''

def Make_points(frame,line_parameters):
    height, width, _ = np.shape(frame)
    slope, intercept = line_parameters
    y1 = height 
    y2 = int(y1*(3/5))

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))

    #time.sleep(0.5)
    return np.array ([x1,y1,x2,y2])




'''
display_lines:
    Parameters:

    frame      -> 
    lines       -> 
    line_color  ->
    line_width  ->


    Return:

    line_image  -> 



'''


def display_lines(frame,lines,line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)  
            cv2.line(line_image, (x1,y1),(x2,y2),line_color,line_width)

    #addweight() blends two images 
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return line_image


'''
detect_two_lines:
    Parameters:
 
    frame          -> 
    lane_lines     -> 


    Return:
    steering_angle -> 


'''
def detect_two_lines(frame, lane_lines):
    height, width, _ = np.shape(frame)
    _,_,left_x2,_ = lane_lines[0]
    _,_,right_x2,_ = lane_lines[1]

    
    mid = int(width/2)
    xoffset = (left_x2 + right_x2) / 2 - mid 
    yoffset = int(height)/2

    angle_to_mid_radian = math.atan(xoffset / yoffset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    print(steering_angle)

    return steering_angle

'''
detect_one_line:
    Parameters:

    Return:
'''

def detect_one_line(frame,lane_lines):
    x1,_,x2,_ = lane_lines[0][0]
    x_offset = x2 - x1
    y_offset = int(height/2)


'''
display_middle_line:
    Parameters:

    frame          -> 
    steering_angle -> 


    return: 

    heading_image -> 

'''
    

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
'''
stabilize_steering_angle:
    Parameters:

    frame          -> 
    steering_angle -> 


    Return:

    heading_image -> 
'''

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

def drive_car(steering_angle):

    if steering_angle is None:
        print('no angle found')
        return 0 
    if steering_angle == 90 :
        print ("motor moving forward")
        motor.forward()

    elif steering_angle > 0 and steering_angle <= 89:
        print("car turning left")
        car.turn_left()
    elif steering_angle > 90 and steering_angle <= 180:
        print ("car turning right")
        car.turn_right()
    #elif data == "Home":
      #  print ("recv home cmd")
        #motor.ctrl(0)
        #car_dir.home()


def main():  
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        _,frame = cap.read()
        
        warnings.simplefilter('ignore', np.RankWarning)
       #frame = cv2.GaussianBlur(frame, (5, 5), 0)
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
        #plt.imshow(cropped_edges)
        #plt.show()
        drive_car(steering_angle)
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            

if __name__ == "__main__":
    main()





    

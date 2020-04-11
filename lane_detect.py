import numpy as np
import cv2
import matplotlib.pyplot as plt 



    

       
#def main():


##was main but changed 

def detect_lane():
    


        
    edges = Detect_Edges(cap)
    cropped_edges = Cut_top_half(edges)
    line_segments =Detect_line_segment(cropped_edges)
    lane_lines =  Avg_slope(line_segments,cap)

    return lane_lines


        

    
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
    min_threshold = 10 
    min_line_length = 8
    max_line_gap = 5 
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), min_line_length, max_line_gap)
    
    print(line_segments)
         
    return line_segments



def Avg_slope(line_segments,cap):
    #if slope is > 0 then on left
    #if slope is < 0 on the right 

    ret, frame = cap.read()
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
        x1,y1,x2,y2 = line_segment.reshape(4)
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
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array ([x1,y1,x2,y2])






def Detect_Edges(cap):

    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_blue = np.array([60,40,40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    #cv2.imshow('mask',mask)

    edges = cv2.Canny(mask, 200, 400)
    
    

    return edges


def display_lines(cap,lines,line_color=(0, 255, 0), line_width=2):
    ret, frame = cap.read()
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)  
            cv2.line(line_image, (x1,y1),(x2,y2),line_color,line_width)

    #addweight() blends two images 
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return line_image

cap = cv2.VideoCapture(0)
while (True):

    lane_lines_image = display_lines(cap, detect_lane())
    cv2.imshow("lane lines", lane_lines_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

#if __name__ == "__main__":
    #main()





    

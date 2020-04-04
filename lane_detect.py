import numpy as np
import cv2




    

       
def main():
    cap = cv2.VideoCapture(0)
    while (True):
        
        edges = Detect_Edges(cap)
        cropped_edges = Cut_top_half(edges)
        Detect_line_segment(cropped_edges)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
	max_line_gap = 4 
	line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), min_line_length, max_line_gap)
	
	
	return line_segments



def Avg_slope(line_segments,cap):
    #if slope is > 0 then on left
    #if slope is < 0 on the right 

    ret, frame = cap.read()
    height, width = np.shape(frame)
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
            if (x1 = x2):
                print ('UND')
                continue 

            fit = np.polyfit((x1,x2),(y1,y2),1)
            slope = fit[0]
            intercept = fit[1]

            if slope < 0:
                if x1 < left_boundary and x2 < left_boundary:
                    left_fit.append((slope,intercept))
            else:
                if x1 > right_boundary and x2 > right_boundary:
                    right_fit.append((slope,intercept))
            






def Detect_Edges(cap):

    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_blue = np.array([60,40,40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    cv2.imshow('mask',mask)

    edges = cv2.Canny(mask, 200, 400)
    

    return edges


if __name__ == "__main__":
    main()





    

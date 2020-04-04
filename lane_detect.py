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
    mask = np.zereos_like(edges)

    #delete top hald of screen 
    poly = np.arrray ([[(0,height*1/2),(width,height*1/2),(width,height),(0,height)]],np.int32)

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
	
	print (line_segments)
	return line_segments






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





    

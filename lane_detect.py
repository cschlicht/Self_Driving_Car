import numpy as np
import cv2




    

       
def main():
    cap = cv2.VideoCapture(0)
    while (True):
        
        Detect_Edges(cap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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





    

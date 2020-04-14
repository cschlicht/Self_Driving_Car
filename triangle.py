import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

#this file is used to help detect edges 
def main():
	cap = cv2.VideoCapture(0)
	while (True):
		ret, frame = cap.read()
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		lower_blue = np.array([60,40,40])
    	upper_blue = np.array([150, 255, 255])
    	mask = cv2.inRange(hsv, lower_blue, upper_blue)


    	cv2.imshow('mask',mask)
    	plt.imshow(edges)
    	plt.show(0)





if __name__ == "__main__":
	main()
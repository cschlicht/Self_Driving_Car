#!/usr/bin/env python
import RPi.GPIO as GPIO 
import car_dir_main
#import motor_main
import socket
from time import ctime          # Import necessary modules   

# ctrl_cmd = ['forward', 'backward', 'left', 'right', 'stop', 'read cpu_temp', 'home', 'distance', 'x+', 'x-', 'y+', 'y-', 'xy_home']

# busnum = 1          # Edit busnum to 0, if you uses Raspberry Pi 1 or 0

UDP_IP = "192.168.1.106"
UDP_PORT = 6661



# tcpSerSock = socket(AF_INET, SOCK_STREAM)    # Create a socket.
# tcpSerSock.bind(ADDR)    # Bind the IP address and port number of the server. 
# tcpSerSock.listen(5)     # The parameter of listen() defines the number of connections permitted at one time. Once the 
#                          # connections are full, others will be rejected. 

#create udp socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("check")
#bind socket
s.bind((UDP_IP,UDP_PORT))
print("socketBind")

# video_dir.setup(busnum=busnum)
# car_dir.setup(busnum=busnum)
# motor.setup(busnum=busnum)     # Initialize the Raspberry Pi GPIO connected to the DC motor. 
# video_dir.home_x_y()
# car_dir.home()

while True:
   # print 'Waiting for connection...'
   
        data, addr = s.recvfrom(1024)
        data = data.decode('UTF-8')
        print("recieved message: ", data)
        
        # Analyze the command received and control the car accordingly.
        if not data:
            break
        if data == "Forward":
            print ("motor moving forward")
            #motor.forward()
        elif data == "Backword":
            print ("recv backward cmd")
            #motor.backward()
        elif data == "Left":
            print("recv left cmd")
            #car_dir.turn_left()
        elif data == "Right":
            print ("recv right cmd")
            #car_dir.turn_right()
        elif data == "Home":
            print ("recv home cmd")
            #motor.ctrl(0)
            #car_dir.home()
      
s.close()




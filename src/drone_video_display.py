#!/usr/bin/env python

# This display window listens to the drone's video feeds and updates the display at regular intervals
# It also tracks the drone's status and any connection problems, displaying them in the window's status bar
# By default it includes no control functionality. The class can be extended to implement key or mouse listeners if required

# Import the ROS libraries, and load the manifest file which through <depend package=... /> will give us access to the project dependencies
import roslib; roslib.load_manifest('ardrone_tutorials')
import rospy
import cv2 
import numpy as np
import KalmanFilter as kf 
import matplotlib.pyplot as plt
from image_converter import ToOpenCV, ToRos
from KalmanFilter import KalmanFilter

# Import the two types of messages we're interested in
from sensor_msgs.msg import Image    	 # for receiving the video feed
from ardrone_autonomy.msg import Navdata # for receiving navdata feedback

# We need to use resource locking to handle synchronization between GUI thread and ROS topic callbacks
from threading import Lock
from drone_controller import BasicDroneController

# An enumeration of Drone Statuses
from drone_status import DroneStatus

# The GUI libraries
from PySide import QtCore, QtGui

# Some Constants
CONNECTION_CHECK_PERIOD = 250 #ms
GUI_UPDATE_PERIOD = 20 #ms
DETECT_RADIUS = 4 # the radius of the circle drawn when a tag is detected


class DroneVideoDisplay(QtGui.QMainWindow):
	StatusMessages = {
		DroneStatus.Emergency : 'Emergency',
		DroneStatus.Inited    : 'Initialized',
		DroneStatus.Landed    : 'Landed',
		DroneStatus.Flying    : 'Flying',
		DroneStatus.Hovering  : 'Hovering',
		DroneStatus.Test      : 'Test (?)',
		DroneStatus.TakingOff : 'Taking Off',
		DroneStatus.GotoHover : 'Going to Hover Mode',
		DroneStatus.Landing   : 'Landing',
		DroneStatus.Looping   : 'Looping (?)'
		}
	DisconnectedMessage = 'Disconnected'
	UnknownMessage = 'Unknown Status'
	
	def __init__(self):
		# Construct the parent class
		super(DroneVideoDisplay, self).__init__()

		# Setup our very basic GUI - a label which fills the whole window and holds our image
		self.setWindowTitle('AR.Drone Video Feed')
		self.imageBox = QtGui.QLabel(self)
		self.setCentralWidget(self.imageBox)

		self.controller = BasicDroneController()

		# Subscribe to the /ardrone/navdata topic, of message type navdata, and call self.ReceiveNavdata when a message is received
		self.subNavdata = rospy.Subscriber('/ardrone/navdata',Navdata,self.ReceiveNavdata) 
		
		# Subscribe to the drone's video feed, calling self.ReceiveImage when a new frame is received
		self.subVideo   = rospy.Subscriber('/ardrone/image_raw',Image,self.ReceiveImage)

		'''BEGIN CHANGES'''
		#Define Kalman Filter constants
		time = GUI_UPDATE_PERIOD
		time2 = time*time

		#1D case for velocity in the x-direction
		dimension = 1
		A = np.identity(dimension)
		B = np.matrix(time)
		H = np.identity(dimension)
		P = np.identity(dimension)
		Q = np.identity(dimension)
		R = np.identity(dimension)

		#tweak covariance matrices
		Q = np.dot(1,Q)
		R = np.dot(0.1, R)

		#create the Kalman Filter instance
		self.kfilter = KalmanFilter(A, P, R, Q, H, B, dimension)

		#create empty array to house our estimates
		self.state_estimate = []
		self.state_real = []


		#### Computer vision code

		self.x_pix = 320
		self.y_pix = 240

		self.x_ang = np.radians(54.4)
		self.y_ang = np.radians(37.8)

		self.prev_frame = None
		self.prev_points = None
		self.prev_time = None
		self.vel = []

		plt.ion()

		'''END CHANGES'''
		
		# Holds the image frame received from the drone and later processed by the GUI
		self.image = None
		self.imageLock = Lock()

		self.tags = []
		self.tagLock = Lock()
		
		# Holds the status message to be displayed on the next GUI update
		self.statusMessage = ''

		# Tracks whether we have received data since the last connection check
		# This works because data comes in at 50Hz but we're checking for a connection at 4Hz
		self.communicationSinceTimer = False
		self.connected = False

		# A timer to check whether we're still connected
		self.connectionTimer = QtCore.QTimer(self)
		self.connectionTimer.timeout.connect(self.ConnectionCallback)
		self.connectionTimer.start(CONNECTION_CHECK_PERIOD)
		
		# A timer to redraw the GUI
		self.redrawTimer = QtCore.QTimer(self)
		self.redrawTimer.timeout.connect(self.RedrawCallback)
		self.redrawTimer.start(GUI_UPDATE_PERIOD)

	# Called every CONNECTION_CHECK_PERIOD ms, if we haven't received anything since the last callback, will assume we are having network troubles and display a message in the status bar
	def ConnectionCallback(self):
		self.connected = self.communicationSinceTimer
		self.communicationSinceTimer = False

	def RedrawCallback(self):
		if self.image is not None:
			# We have some issues with locking between the display thread and the ros messaging thread due to the size of the image, so we need to lock the resources
			self.imageLock.acquire()
			try:			

					''' 
					TODO:

					1. Create Kalman Filter instance in constructor.         DONE 
					2. Create optical flow instance in constructor. 	     DONE
					3. Retrieve controller navdata here.                     DONE 
					4. Retrieve image matrix here. Conver to cv matrix.      DONE
					5. Run optical flow alg. on image. 						 DONE
					6. Make prediction with controller data.                 DONE
					7. Update prediction with image data.                    DONE
					8. Plot estimate vs real continuously					 DONE
					9. GetHeight() function in drone_controller.py 			 INCOMPLETE
					10. GetTime() function in here						     INCOMPLETE
					'''

					'''BEGIN CHANGES'''
					#convert the ROS image to OpenCV and apply some processing. then convert back to ROS
					openimage = ToOpenCV(self.image)

					# make picture 2D
					frame = cv2.cvtColor(openimage, cv2.COLOR_BGR2GRAY)

					''' TODO: Implement GetHeight in drone_controller.py '''

					feature_params = dict( maxCorners = 100,
	                       qualityLevel = 0.3,
	                       minDistance = 7,
	                       blockSize = 7 )

					if self.prev_frame is None:
						self.prev_frame = frame 
						self.prev_points = cv2.goodFeaturesToTrack(self.prev_frame, mask=None, **feature_params)
						self.prev_time = GUI_UPDATE_PERIOD #self.GetTime() is set to a constant for now
					else:

						h = self.controller.GetHeight()
						xdist = (h * np.tan(self.x_ang / 2.0))
						ydist = (h * np.tan(self.y_ang / 2.0)) 

						pix2mx = self.x_pix / xdist
						pix2my = self.y_pix / ydist

						curr_frame = frame

						''' Implement GetTime() '''
						curr_time = GUI_UPDATE_PERIOD #self.GetTime() is set to constant for now
						new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, curr_frame, self.prev_points)

						good_new = new_points[status==1]
						good_old = self.prev_points[status==1]
						assert good_new.shape == good_old.shape


						### Calculate velocity components
						sum_x = 0.0
						sum_y = 0.0
						ptslen = len(good_new)
						xcomp = 0
						ycomp = 0
						for x in range(ptslen):
							xcomp = ((good_new[x][1] - good_old[x][1]) / self.x_pix) / (curr_time / 1000.0) #- self.prev_time
							ycomp = ((good_new[x][0] - good_old[x][0]) / self.y_pix) / (curr_time / 1000.0)
							sum_y += ycomp
							sum_x += xcomp

						avg_x = np.divide(xcomp, ptslen)
						avg_y = np.divide(ycomp, ptslen)

						self.vel.append(avg_x) #only x for now

						# iterate next frames
						self.prev_frame = curr_frame
						self.prev_points = new_points
						self.prev_time = curr_time


					#Convert to ROS
					ROSimage = ToRos(openimage)

					##############################################
					### Change velocity to get correct one here ## 
					##############################################
					estimated_velocity = self.vel[-1:] ######
					##############################################

					u_k = 0
					real_velocity = 0
					#update the measured accelerations and velocities
					if self.communicationSinceTimer == True:
						real_velocity = self.controller.GetVelocity()
						u_k = self.controller.GetAcceleration()
					z_k = estimated_velocity

					#Kalman Filter step
					self.kfilter.predictState(u_k)
					self.kfilter.getKalmanGain()
					self.kfilter.update(z_k)

					self.state_estimate.append(self.kfilter.x_k)
					self.state_real.append(real_velocity)

					#plot everything here
					#plt.plot(self.state_estimate, label = "estimated velocity")
					plt.plot(self.state_real)
					plt.draw()

					'''END CHANGES'''

					# Convert the ROS image into a QImage which we can display
					image = QtGui.QPixmap.fromImage(QtGui.QImage(ROSimage.data, ROSimage.width, ROSimage.height, QtGui.QImage.Format_RGB888))
					if len(self.tags) > 0:
						self.tagLock.acquire()
						try:
							painter = QtGui.QPainter()
							painter.begin(image)
							painter.setPen(QtGui.QColor(0,255,0))
							painter.setBrush(QtGui.QColor(0,255,0))
							for (x,y,d) in self.tags:
								r = QtCore.QRectF((x*image.width())/1000-DETECT_RADIUS,(y*image.height())/1000-DETECT_RADIUS,DETECT_RADIUS*2,DETECT_RADIUS*2)
								painter.drawEllipse(r)
								painter.drawText((x*image.width())/1000+DETECT_RADIUS,(y*image.height())/1000-DETECT_RADIUS,str(d/100)[0:4]+'m')
							painter.end()
						finally:
							self.tagLock.release()
			finally:
				self.imageLock.release()

			# display the window.
			self.resize(image.width(),image.height())
			self.imageBox.setPixmap(image)

		# Update the status bar to show the current drone status & battery level
		self.statusBar().showMessage(self.statusMessage if self.connected else self.DisconnectedMessage)

	def ReceiveImage(self,data):
		# Indicate that new data has been received (thus we are connected)
		self.communicationSinceTimer = True

		# We have some issues with locking between the GUI update thread and the ROS messaging thread due to the size of the image, so we need to lock the resources
		self.imageLock.acquire()
		try:
			self.image = data # Save the ros image for processing by the display thread
		finally:
			self.imageLock.release()

	def ReceiveNavdata(self,navdata):
		# Indicate that new data has been received (thus we are connected)
		self.communicationSinceTimer = True

		# Update the message to be displayed
		msg = self.StatusMessages[navdata.state] if navdata.state in self.StatusMessages else self.UnknownMessage
		self.statusMessage = '{} (Battery: {}%)'.format(msg,int(navdata.batteryPercent))

		self.tagLock.acquire()
		try:
			if navdata.tags_count > 0:
				self.tags = [(navdata.tags_xc[i],navdata.tags_yc[i],navdata.tags_distance[i]) for i in range(0,navdata.tags_count)]
			else:
				self.tags = []
		finally:
			self.tagLock.release()

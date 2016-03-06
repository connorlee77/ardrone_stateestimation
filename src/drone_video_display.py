#!/usr/bin/env python

# This display window listens to the drone's video feeds and updates the display at regular intervals
# It also tracks the drone's status and any connection problems, displaying them in the window's status bar
# By default it includes no control functionality. The class can be extended to implement key or mouse listeners if required

# Import the ROS libraries, and load the manifest file which through <depend package=... /> will give us access to the project dependencies
import roslib; roslib.load_manifest('ardrone_tutorials')
import rospy
import cv2 as cv
import numpy as np
import KalmanFilter as kf 


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

from image_converter import ToOpenCV, ToRos


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

		'''Begin Kalman Filter init'''
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
		kf = kf.KalmanFilter(A, P, R, Q, H, B, dimension)

		#create empty array to house our estimates
		state_estimate = []

		'''End Kalman Filter init'''
		
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
					2. Create optical flow instance in constructor.
					3. Retrieve controller navdata here.                     DONE 
					4. Retrieve image matrix here. Conver to cv matrix.      DONE
					5. Run optical flow alg. on image.

					6. Make prediction with controller data.                 DONE
					7. Update prediction with image data.                    DONE
					8. Plot estimate vs real continuously
					'''
					#convert the ROS image to OpenCV and apply some processing. then convert back to ROS
					openimage = ToOpenCV(self.image)
					#estimated_velocity = opencvfnction(openimage) 
					ROSimage = ToRos(openimage)

					#update the measured accelerations and velocities
					real_velocity = self.controller.GetVelocity()
					u_k = self.controller.GetAcceleration()
					z_k = estimated_velocity

					kf.predictState(u_k)
					kf.getKalmanGain()
					kf.update(z_k)

					state_estimate.append(kf.x_k)

					#plot everything here




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

'''if __name__=='__main__':
	import sys
	rospy.init_node('ardrone_video_display')
	app = QtGui.QApplication(sys.argv)
	display = DroneVideoDisplay()
	display.show()
	status = app.exec_()
	rospy.signal_shutdown('Great Flying!')
	sys.exit(status)'''
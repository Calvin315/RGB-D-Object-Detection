import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from skimage import feature
from skimage import img_as_ubyte
from foos import intersection_over_union, area_of_box

f = 1.93 # focal length of RealSense D435
confidence = 0.70

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Fill holes in depth
holeFilter = rs.hole_filling_fil ter()
holeFilter.set_option(rs.option.holes_fill, 1 )

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_distance_in_meters = 2 # 2 meters
clipping_distance = clipping_distance_in_meters / depth_scale
print(int(clipping_distance+ 1 ))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

def nothing (x):
	pass
	
cv.namedWindow( "Options" )
cv.createTrackbar( 'Lower Thresh' , 'Options' , 570 , 2000 , nothing)
cv.createTrackbar( 'Upper Thresh' , 'Options' , 1400 , 2000 , nothing)
cv.createTrackbar( 'Sigma' , 'Options' , 7 , 15 , nothing)

try :
	while True :
	
	#Make sliders
	lowerCannyThresh = cv.getTrackbarPos( 'Lower Thresh' , 'Options' )
	upperCannyThresh = cv.getTrackbarPos( 'Upper Thresh' , 'Options' )
	sigma = cv.getTrackbarPos( 'Sigma' , 'Options' )
	
	# Wait for a coherent pair of frames: depth and color
	frames = pipeline.wait_for_frames()
	
	# Align the depth frame to color frame
	aligned_frames = align.process(frames)
	
	# Get aligned frames
	depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480
	depth image
	color_frame = aligned_frames.get_color_frame()
	
	# Validate that both frames are valid
	if not depth_frame or not color_frame:
		continue
		
	# Let RealSense fill holes in depth map
	depth_frame = holeFilter.process(depth_frame)
	
	# Convert images to numpy arrays
	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())
	
	# Background elimination
	# Remove background of color image - Set pixels further than clipping_distance to grey
	grey_color = 153
	depth_image_3d = np.dstack((depth_image, depth_image, depth_image)) # depth image is 1 channel, color is 3 channels
	color_bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d<= 0 ), grey_color, color_image)
	
	# Eliminate bg of depth image over clipping_distance away
	depth_bg_removed = np.where((depth_image > int(clipping_distance)) | (depth_image <=0 ), int(clipping_distance), depth_image)
	
	################ DEPTH OBJ DETECTION ##########################
	# Histogram Equilization
	clahe = cv.createCLAHE(clipLimit= 50.00 , tileGridSize=( 3 , 3 ))
	depth_eq = clahe.apply(depth_bg_removed)
	
	# Open and Closing Depth Image
	strel = cv.getStructuringElement(cv.MORPH_RECT, ( 5 , 5 ))
	depth_clean = cv.morphologyEx(depth_eq, cv.MORPH_OPEN, strel, 3 )
	depth_clean = cv.morphologyEx(depth_clean, cv.MORPH_CLOSE, strel, 3 )
	
	# Edge detection
	depth_edge = feature.canny(depth_clean, sigma=sigma,
	low_threshold=lowerCannyThresh, high_threshold=upperCannyThresh)
	depth_edge = img_as_ubyte(depth_edge)
	
	################ COLOR OBJ DETECTION ##########################
	# convert bg removed color image to greyscale
	color_bg_gray = cv.cvtColor(color_bg_removed, cv.COLOR_BGR2GRAY)
	
	# blur
	color_bg_gray = cv.blur(color_bg_gray, ( 3 , 3 ))
	
	# threshold greyscale image
	___, th = cv.threshold(color_bg_gray, 127 , 255 , cv.THRESH_OTSU)
	
	# Close and open threshold image
	th = cv.morphologyEx(th, cv.MORPH_CLOSE, strel, 3 )
	th = cv.morphologyEx(th, cv.MORPH_OPEN, strel, 3 )
	
	# bitwise not because why not
	th = cv.bitwise_not(th)
	
	############# BOUNDING BOXES ################
	color_result = np.copy(color_image)
	
	# Get contours of depth image
	contours, hierarchy = cv.findContours(depth_edge, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
	
	# Approximate Contours to Polygons + Get Bounding Boxes of depth image
	edgeRect = [ None ] * len(contours)
	contours_poly = [ None ] * len(contours)
	iou = []
	for i, c in enumerate(contours):
		contours_poly[i] = cv.approxPolyDP(c, 3 , True )
		edgeRect[i] = cv.boundingRect(contours_poly[i])
		
	# Draw polygonal contour + bonding rects + circles
	for i in range(len(contours)):
		color = ( 0 , 255 , 0 )
		cv.rectangle(color_result, (int(edgeRect[i][ 0 ]), int(edgeRect[i][ 1 ])), (int(edgeRect[i][ 0 ] + edgeRect[i][ 2 ]), int(edgeRect[i][ 1 ] + edgeRect[i][ 3 ] + 20 )), color, 2 )
		
	# Get contours of threshold image
	contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
	
	# Approximate Contours to Polygons + Get Bounding Boxes of threshold image
	thRect = [ None ] * len(contours)
	contours_poly = [ None ] * len(contours)
	for i, c in enumerate(contours):
		contours_poly[i] = cv.approxPolyDP(c, 3 , True )
		thRect[i] = cv.boundingRect(contours_poly[i])
	drawing = np.zeros((depth_edge.shape[ 0 ], depth_edge.shape[ 1 ], 3 ), dtype=np.uint8)
	# Draw polygonal contour + bonding rects + circles
	for i in range(len(contours)):
		color = ( 0 , 0 , 255 )
		cv.rectangle(color_result, (int(thRect[i][ 0 ]), int(thRect[i][ 1 ])), (int(thRect[i][ 0 ] + thRect[i][ 2 ]), int(thRect[i][ 1 ] + thRect[i][ 3 ])), color, 2 )
	# Chose boxes from depth edge and thresh images that have an overlap percentage of 50%
	boxes = []
	for i in range(len(edgeRect)):
		for j in range(len(thRect)):
			iou = (intersection_over_union([edgeRect[i][ 0 ], edgeRect[i][ 1 ], (edgeRect[i][ 0 ] + edgeRect[i][ 2 ]), (edgeRect[i][ 1 ] + edgeRect[i][ 3 ])], [thRect[j][ 0 ], thRect[j][ 1 ], (thRect[j][ 0 ] + thRect[j][ 2 ]), (thRect[j][ 1 ] + thRect[j][ 3 ])]))
			if (iou > confidence):
				boxes.append([edgeRect[i], thRect[j]])
				
	# Take choice boxes and choose the largest one
	choice = []
	dist = 0.0
	choice_dist = 0.0
	for i in range(len(boxes)):
		a1 = area_of_box(boxes[i][ 0 ])
		a2 = area_of_box(boxes[i][ 1 ])
		if a1 > a2:
			if a1 > 500 :
				choice = boxes[i][ 0 ]
		else :
			if a2 > 500 :
				choice = boxes[i][ 1 ]
				
	# Select Ground Truth
	# r = cv.selectROI('GT', color_image, showCrosshair=True, fromCenter=False)
	# display choice bounding box and estimate depth
	
	if len(choice) != 0 :
		startX = int(choice[ 0 ])
		startY = int(choice[ 1 ])
		endX = int(choice[ 0 ] + choice[ 2 ])
		endY = int(choice[ 1 ] + choice[ 3 ])
		
		# Display choice bounding box
		color = ( 0 , 255 , 255 )
		cv.rectangle(color_result, (startX, startY), (endX, endY), color, 7 )
		
		# display coice bounding box and estimate depth
		# Crop depth data
		depth = depth_image[startY:endY, startX:endX].astype(float)
		
		# Get distance of object and display
		depth = depth * depth_scale
		
		# display choice bounding box and estimate depth
		dist, _, _, _ = cv.mean(depth)
		label = "%f2" % dist + " Meters"
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv.putText(color_result, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5 , color, 2 )
		
	cv.imshow( "Depth" , depth_clean)
	cv.imshow( 'Edge' , depth_edge)
	cv.imshow( 'Bin' , th)
	cv.imshow( "RGB" , color_result)
	
	cv.waitKey( 1 )
finally :
	# Stop streaming
	pipeline.stop()
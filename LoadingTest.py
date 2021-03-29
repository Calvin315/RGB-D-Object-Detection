import pyrealsense2 as rs
import random as rng
import numpy as np
import cv2

Streaming = True;

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

if(Streaming == True):
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    config.enable_device_from_file("Floor Video.bag");

# Start streaming
profile = pipeline.start(config)

# Hole Filling Filter, Fill using nearest pixel
holeFilter = rs.hole_filling_filter()
holeFilter.set_option(rs.option.holes_fill, 2)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

clipping_distance_in_meters = 1 # value in meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Alignment of Color and Depth
align_to = rs.stream.color
align = rs.align(align_to)


def nothing(x):
    pass

cv2.namedWindow("Options");
cv2.createTrackbar('Lower Thresh', 'Options', 90, 1000, nothing)
cv2.createTrackbar('Upper Thresh', 'Options', 170, 1000, nothing)
cv2.createTrackbar('Depth', 'Options', 3000, 60000, nothing)

try:
    while True:

        lowerCannyThresh = cv2.getTrackbarPos('Lower Thresh', 'Options')
        upperCannyThresh = cv2.getTrackbarPos('Upper Thresh', 'Options')
        depthDist = cv2.getTrackbarPos('Depth', 'Options')

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Aligns the depth frames
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Hole Filtering
        depth_frame = holeFilter.process(depth_frame);

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Smoothing
        depth_image = cv2.blur(depth_image, (3, 3))

        # Eliminate bg over depthDist away
        depth_image = np.where((depth_image > depthDist) | (depth_image <= 0), int(depthDist), depth_image)

        # Open and Closing Depth Image
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_OPEN, strel, 3)
        depth_image = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, strel, 3)

        # Histogram Equilization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        depth_image = clahe.apply(depth_image)

        # Apply color mapping
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)

        # Closing Operator
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 13))
        depth_colormap = cv2.morphologyEx(depth_colormap, cv2.MORPH_ERODE, strel, 3)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RGB / Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB / Depth', images)

        # Derivatives / Scharr or Laplacian
        Deriv = cv2.Laplacian(depth_image, cv2.CV_16U, 3)

        deriv_colormap = cv2.applyColorMap(cv2.convertScaleAbs(Deriv, alpha=0.03), cv2.COLORMAP_PINK)

        # Show images
        DerivCannyEdge = cv2.Canny(depth_colormap, lowerCannyThresh, upperCannyThresh)
        cv2.namedWindow('Deriv Canny', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Deriv Canny', DerivCannyEdge)
        cv2.imshow('Laplacian', deriv_colormap)

        im2, contours, hierarchy = cv2.findContours(DerivCannyEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Approximate contours to polygons + get bounding rects
        boundRect = [None] * len(contours)
        contours_poly = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        drawing = np.zeros((DerivCannyEdge.shape[0], DerivCannyEdge.shape[1], 3), dtype=np.uint8)
        # Draw polygonal contour + bonding rects + circles
        for i in range(len(contours)):
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.rectangle(color_image, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        cv2.imshow('Bounding Boxes', color_image)

        # convex hull
        # hull_list = []
        # for i in range(len(contours)):
        #     hull = cv2.convexHull(contours[i])
        #     hull_list.append(hull)
        #
        # drawing = np.zeros((DerivCannyEdge.shape[0], DerivCannyEdge.shape[1], 3), dtype=np.uint8)
        # for i in range(len(contours)):
        #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        #     cv2.drawContours(drawing, contours, i, color)
        #     cv2.drawContours(drawing, hull_list, i, color)
        # # Show in a window
        # cv2.imshow('Contours', drawing)

        ######################################################
        # depth_image_8U = np.uint8(np.absolute(depth_image))
        # Deriv_8U = np.uint8(np.absolute(Deriv))

        ## Trying Histogram equalization
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        # depth_image = clahe.apply(depth_image)

        ## Function for first derivative
        # Deriv = cv2.Scharr(depth_image, cv2.CV_16U , 0, 1)

        ## Trying Canny Edge Detection on 2nd Deriv Images, Doesnt work...
        # DepthCannyEdge = cv2.Canny(depth_image_8U, lowerCannyThresh, upperCannyThresh)

        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # DerivCannyEdge = cv2.morphologyEx(DerivCannyEdge, cv2.MORPH_OPEN, kernel)
        # DepthCannyEdge = cv2.morphologyEx(DepthCannyEdge, cv2.MORPH_OPEN, kernel)
        #
        # Canny = np.hstack((DepthCannyEdge , DerivCannyEdge))
        # cv2.namedWindow('Canny - Depth / Deriv', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Canny - Depth / Deriv', Canny)

        cv2.waitKey(1)



finally:

    # Stop streaming
    pipeline.stop()
import cv2
import numpy as np
import os
import sys
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms

# Define functions from the first code snippet
def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def highPassFilter(kSize):
    global img
    
    print("Applying high pass filter")
    
    if not kSize % 2:
        kSize += 1
        
    kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
    
    filtered = cv2.filter2D(img, -1, kernel)
    
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127 * np.ones(img.shape, np.uint8)
    
    filtered = filtered.astype('uint8')
    
    img = filtered

def blackPointSelect():
    global img
    
    print("Adjusting black point for final output ...")
    
    img = img.astype('int32')
    img = map(img, blackPoint, 255, 0, 255)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)

    img = img.astype('uint8')

def whitePointSelect():
    global img
    
    print("White point selection running ...")

    _, img = cv2.threshold(img, whitePoint, 255, cv2.THRESH_TRUNC)

    img = img.astype('int32')
    img = map(img, 0, whitePoint, 0, 255)
    img = img.astype('uint8')

def blackAndWhite():
    global img
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    (l, a, b) = cv2.split(lab)

    img = cv2.add(cv2.subtract(l, b), cv2.subtract(l, a))

def scan_document(image):
    image = cv2.resize(image, (600, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

    if max_contour is None:
        print("No document contour found.")
        return None

    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

    rect_points = np.float32([point[0] for point in approx])
    rect_points = rect_points[np.argsort(rect_points[:, 1])]

    if rect_points[0][0] > rect_points[1][0]:
        rect_points[[0, 1]] = rect_points[[1, 0]]
    if rect_points[2][0] < rect_points[3][0]:
        rect_points[[2, 3]] = rect_points[[3, 2]]

    if len(rect_points) != 4:
        print("Could not detect four corners of the document.")
        return None

    dst = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect_points, dst)
    warped = cv2.warpPerspective(image, M, (600, 600))

    return warped

# Define functions from the second code snippet
# (show function, overlay_mask, overlay_ann, get_instance_segmentation_model, main)

# Load the image
img = cv2.imread("test2.jpg")

# Run the first set of processes
scanned_document = scan_document(img)

if scanned_document is not None:
    img = scanned_document

    blackPoint = 66
    whitePoint = 160

    mode = "GCMODE"

    if mode == "GCMODE":
        highPassFilter(kSize=51)
        whitePoint = 127
        whitePointSelect()
        blackPointSelect()
    elif mode == "RMODE":
        blackPointSelect()
        whitePointSelect()
    elif mode == "SMODE":
        blackPointSelect()
        whitePointSelect()
        blackAndWhite()

    print("\nFirst set of processes completed.")

    # Save the processed image
    cv2.imwrite('output2.jpg', img)
    
    # Run the second set of
    os.system("python model_text_image4.py")  # assuming second_code.py contains the second code snippet
    
else:
    print("No document contour found.")

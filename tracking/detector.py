import numpy as np
import cv2



def detect(frame, debugMode):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(gray,  50, 190, 3)

    ret, img_thresh = cv2.threshold(img_edges, 254, 255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    if (debugMode):
        cv2.imshow('gray', gray)

    if (debugMode):
        cv2.imshow('img_edges', img_edges)

    if (debugMode):
        cv2.imshow('img_thresh', img_thresh)


    min_radius_thresh= 3
    max_radius_thresh= 30

    centers=[]
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)

        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
    cv2.imshow('contours', img_thresh)

    print("Detected Centers : ",centers[0].astype('float16').tolist())

    return centers
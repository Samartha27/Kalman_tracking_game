import numpy as np
import cv2
from detector import detect
from kalman_filter import KalmanFilter
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Input the video directory")
    args = parser.parse_args()

    if args.input_dir:
        video_dir = args.input_dir
    else:
        video_dir = '../data/jumping_ball.mp4'

    cap = cv2.VideoCapture(video_dir)

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1) #(dt,u_x,u_y,sigma_a,sigma_x,sigma_y)

    debugMode=1

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter('./result/output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         32, size)

    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            break

        centers = detect(frame,debugMode)


        if (len(centers) > 0):

            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

            (x, y) = KF.predict()
            #cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (255, 0, 0), 2)

            (x1, y1) = KF.update(centers[0])
            cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)

            #cv2.putText(frame, "Kalman a priori", (int(x + 15), int(y)), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Kalman a posteriori estimate ", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Detected Position", (int(centers[0][0] + 15), int(centers[0][1] - 15)), 0, 0.5, (0,0,0), 2)

        result.write(frame)
        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            result.release()
            cap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(1)


if __name__ == "__main__":
    main()

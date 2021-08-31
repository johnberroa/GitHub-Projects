# Just a little fun adding effects to the OpenCV webcam stream tutorial script
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
kernel = np.ones((5, 5), np.uint8)
type = 'c'
color = 1

def set_background(frame):
    return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def open_cam(k, type, color):
    frames_passed = 0
    color_background = None
    gray_background = None
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frames_passed == 10:  # to allow for the camera to adjust exposure
            color_background, gray_background = set_background(frame)
        frames_passed += 1

        font = cv2.FONT_HERSHEY_PLAIN
        x = 30
        level = 1
        cv2.putText(frame, "Press the following keys for effects:", (10, x), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        x += (15 // 2)
        cv2.putText(frame, "'g': Gray", (20, x + (15 * level)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "'d': Difference image", (20, x + (15 * level)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "'b': Blocky", (20, x + (15 * level)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "'c': Color", (20, x + (15 * level)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "'p': Phantom", (20, x + (15 * level)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "'a': Alien", (20, x + (15 * level)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "Type 't' to set background image", (10, x + (15 * level + (16 // 2))), font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "Type 's' to save image", (10, x + (15 * level + (16 // 2))), font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
        level += 1
        cv2.putText(frame, "Type 'q' to quit", (10, x + (15 * level + (16 // 2))), font, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Play around with the stream
        if type == 'g':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            color = 0
            cv2.imshow('Stream', frame)
        elif type == 'd':
            if color == 1:
                frame = simpleDifference(frame, color_background)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = simpleDifference(frame, gray_background)
            cv2.imshow('Stream', frame)
        elif type == 'b':
            if color == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.erode(frame, k, iterations=5)
            else:
                frame = cv2.erode(frame, k, iterations=5)
            cv2.imshow('Stream', frame)
        elif type == 'a':
            if color == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.dilate(frame, k, iterations=5)
            else:
                frame = cv2.dilate(frame, k, iterations=5)
            cv2.imshow('Stream', frame)
        elif type == 'c':
            color = 1
            cv2.imshow('Stream', frame)
        elif type == 'p':
            if color == 1:
                frame = phantom(frame, color_background)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = phantom(frame, gray_background)
            cv2.imshow('Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('g'):
            type = 'g'
        elif cv2.waitKey(1) & 0xFF == ord('d'):
            type = 'd'
        elif cv2.waitKey(1) & 0xFF == ord('b'):
            type = 'b'
        elif cv2.waitKey(1) & 0xFF == ord('a'):
            type = 'a'
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            type = 'c'
        elif cv2.waitKey(1) & 0xFF == ord('p'):
            type = 'p'
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            # Can potentially overwrite previous photos
            cv2.imwrite(f'WebcamFun-{frames_passed}.jpg', frame)
        elif cv2.waitKey(1) & 0xFF == ord('t'):
            color_background, gray_background = set_background(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def simpleDifference(img1, img2):
    epsilon = 80
    diffImg = cv2.absdiff(img1, img2)
    _, diffImg = cv2.threshold(diffImg, epsilon, 255, cv2.THRESH_BINARY)
    return diffImg


def phantom(img1, img2):
    diffImg = cv2.absdiff(img1, img2)
    return diffImg


open_cam(kernel, type, color)

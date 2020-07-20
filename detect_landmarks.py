# This code is based on https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# import the necessary packages
from pathlib import Path

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from joblib import dump, load
from tqdm import tqdm

from utils import normalize_landmarks, extract_landmarks, parse_image_path

model = load("model_ITML.joblib")
norm_ref_landmark = None

action_reference_landmarks = np.load("reference_landmark_folder/bacio.npy")


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Initialize the camera
    cv2.namedWindow("FaceLandmarks")
    cap = cv2.VideoCapture(0)
    # vc.set(3, 320)
    # vc.set(4, 240)

    if cap.isOpened():
        rval, frame = cap.read()
    else:
        rval = False


    grab_next_landmark_frame = False
    while rval:
        # Read frame and convert to grayscale
        rval, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 0)
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            key = cv2.waitKey(1)

            if key == ord('c'):
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            rect = cv2.minAreaRect(shape)
            ((x, y), _, angle) = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            if key == ord('q'):
                rval = False

            if key == ord("s"):
                grab_next_landmark_frame = True

            if grab_next_landmark_frame:
                reference_landmark = extract_landmarks(frame)
                if reference_landmark != []:
                    norm_ref_landmark = normalize_landmarks(reference_landmark[0]).flatten()
                    grab_next_landmark_frame = False
                    print("Successfully saved reference neuter image")

            if norm_ref_landmark is not None:
                normalized_landmarks = normalize_landmarks(shape).flatten() - norm_ref_landmark

                distance_func = model.get_metric()
                distance = distance_func(action_reference_landmarks, normalized_landmarks)
                yes_no = model.predict([[action_reference_landmarks, normalized_landmarks]])
                cv2.putText(frame, f"distance {distance:.2f} {yes_no}", (int(x) - 10, int(y) - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # for (x, y) in normalized_landmarks:
                #     cv2.circle(frame, (int(x * 250 + 150), int(y * 250 + 150)), 1, (0, 255, 255), -1)

        cv2.imshow("FaceLandmarks", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
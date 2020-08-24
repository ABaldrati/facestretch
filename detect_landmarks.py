# This code is based on https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# import the necessary packages
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from imutils import face_utils
from joblib import load
from tensorflow import keras

from utils import normalize_landmarks, extract_landmarks, detector, predictor, normalize_landmarks_eyes, \
    manifold

MODELS_PATH = Path("models")
REFERENCE_FOLDER_PATH = Path("reference_landmark_folder")


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


def load_model(model_path: Path):
    if callable(model_path):
        return model_path, model_path
    elif model_path.suffix == ".h5":
        return keras.models.load_model(model_path), model_path
    elif model_path.suffix == ".joblib":
        return load(model_path), model_path


def main():
    actions = sorted(list(set(map(lambda i: i.stem, REFERENCE_FOLDER_PATH.iterdir()))))
    action_reference_landmarks = {}
    for action in actions:
        action_reference_landmarks[action] = np.load(str(REFERENCE_FOLDER_PATH.joinpath(f"{action}.npy")))

    models = sorted(list(set(map(lambda i: i, MODELS_PATH.iterdir()))))
    models.append(manifold)

    face_row_mapping = {}
    for i, action in enumerate(actions):
        face_row_mapping[action] = i

    model, current_model_path = load_model(models[0])

    # Initialize the camera
    cv2.namedWindow("FaceLandmarks")
    cap = cv2.VideoCapture(0)
    # vc.set(3, 320)
    # vc.set(4, 240)

    if cap.isOpened():
        rval, frame = cap.read()
    else:
        rval = False

    current_action = None
    norm_ref_landmark = None
    grab_next_landmark_frame = False

    faces_init = []
    previous_shapes = None
    SMOOTHING_FRAMES_FACTOR = 5

    while rval:
        # Read frame and convert to grayscale
        rval, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if current_action is not None:
            cv2.putText(frame, f"target action: {current_action}", (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
            if callable(current_model_path):
                cv2.putText(frame, f"model: {str(current_model_path.__name__)}", (400, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
            else:
                cv2.putText(frame, f"model: {str(current_model_path.stem)}", (400, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
        else:
            cv2.putText(frame, f"Premi 's' per acquisire volto neutro", (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)

        # detect faces in the grayscale image
        rects = detector(gray, 0)
        # loop over the face detections

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            current_shape = predictor(gray, rect)
            current_shape = face_utils.shape_to_np(current_shape)

            if len(faces_init) == SMOOTHING_FRAMES_FACTOR and previous_shapes is None:
                previous_shapes = deque(faces_init, maxlen=SMOOTHING_FRAMES_FACTOR)

            if previous_shapes is None:
                faces_init.append(current_shape)
            else:
                previous_shapes.append(current_shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if previous_shapes is not None:
                raw_weights = np.arange(SMOOTHING_FRAMES_FACTOR) + 1
                weights = (len(raw_weights) ** raw_weights / len(raw_weights))
                weights = weights / sum(weights)

                shape = np.zeros((68, 2))
                for weight, previous_shape in zip(weights, previous_shapes):
                    shape += weight * previous_shape
                shape = np.int0(shape)
            else:
                shape = current_shape

            # show the face number
            # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            key = cv2.waitKey(1)

            if key == ord('n'):
                for (x, y) in normalize_landmarks(shape):
                    cv2.circle(frame, (int(x * 250 + 150), int(y * 250 + 150)), 1, (0, 255, 255), -1)

                for (x, y) in normalize_landmarks(current_shape):
                    cv2.circle(frame, (int(x * 250 + 150), int(y * 250 + 150)), 1, (255, 0, 255), -1)

            if key == ord("c"):
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            if key == ord('a') and current_action is not None:
                current_action = actions[(actions.index(current_action) - 1) % len(actions)]

            if key == ord('d') and current_action is not None:
                current_action = actions[(actions.index(current_action) + 1) % len(actions)]

            if key == ord("w") and current_action is not None:
                model, current_model_path = load_model(models[(models.index(current_model_path) + 1) % len(models)])

            if key == ord("x") and current_action is not None:
                model, current_model_path = load_model(models[(models.index(current_model_path) - 1) % len(models)])

            if key == ord('q'):
                rval = False

            if key == ord("s"):
                grab_next_landmark_frame = True

            if grab_next_landmark_frame:
                reference_landmark = extract_landmarks(frame)
                if reference_landmark != []:
                    norm_ref_landmark = normalize_landmarks_eyes(reference_landmark[0]).flatten()
                    grab_next_landmark_frame = False
                    if current_action is None:
                        current_action = actions[0]
                    print("Successfully saved reference neuter image")

            if norm_ref_landmark is not None and current_action is not None:
                normalized_landmarks = normalize_landmarks_eyes(shape).flatten() - norm_ref_landmark

                if ".joblib" in str(current_model_path):
                    distance_func = model.get_metric()
                    distance = distance_func(action_reference_landmarks[current_action], normalized_landmarks)
                    cv2.putText(frame, f"distance {distance:.2f}", (int(x) - 10, int(y) - 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                elif ".h5" in str(current_model_path):
                    distance = model.predict(normalized_landmarks.reshape(1, -1))
                    distance = distance[:, face_row_mapping[current_action]]
                    np.set_printoptions(precision=2)
                    cv2.putText(frame, f"distance {1 - distance}", (int(x) - 10, int(y) - 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                elif callable(model):
                    distance = model(action_reference_landmarks[current_action], normalized_landmarks)
                    cv2.putText(frame, f"distance {distance:.2f}", (int(x) - 10, int(y) - 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # for (x, y) in normalized_landmarks:
                #     cv2.circle(frame, (int(x * 250 + 150), int(y * 250 + 150)), 1, (0, 255, 255), -1)

        cv2.imshow("FaceLandmarks", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

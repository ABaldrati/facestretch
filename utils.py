import itertools
from pathlib import Path
from typing import List

import cv2
import dlib
import numpy as np
import skimage as sk
from imutils import face_utils
from matplotlib import pyplot as plt
from more_itertools import windowed
from scipy import linalg as la
from skimage import transform


def manifold(flattened_landmarks1, flattened_landmarks2):  # manifold distance
    landmarks1 = flattened_landmarks1.reshape((-1, 2))
    landmarks2 = flattened_landmarks2.reshape((-1, 2))
    g1 = np.outer(landmarks1, landmarks1.T)
    g2 = np.outer(landmarks2, landmarks2.T)
    # print('Printing matrix shapes'), print(pose_descriptor_1.shape), print(g1.shape)

    distance = np.trace(g1) + np.trace(g2) - 2.0 * np.trace(la.fractional_matrix_power(
        np.dot(la.fractional_matrix_power(g1, 0.5), np.dot(g2, la.fractional_matrix_power(g1, 0.5))), 0.5))
    return distance


def get_rotation_matrix(angle):
    cost = np.cos(np.deg2rad(angle))
    sint = np.sin(np.deg2rad(angle))
    rotation_matrix = np.array([[cost, -sint],
                                [sint, cost]])
    return rotation_matrix


def normalize_landmarks(landmarks):
    (x, y), (w, h), angle = cv2.minAreaRect(landmarks)

    if abs(angle + 90) < abs(angle):
        angle += 90

    landmarks_matrix = np.hstack([np.array(landmarks), np.ones((len(landmarks), 1))])
    center_point = (x + w / 2, y + h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, scale=1)
    rotated_landmarks = np.dot(landmarks_matrix, rotation_matrix.T)[:, :2]
    rotated_landmarks_int = np.int0(rotated_landmarks)

    x_bounding_rect, y_bounding_rect, w_bounding_rect, h_bounding_rect = cv2.boundingRect(rotated_landmarks_int)
    rotated_landmarks[:, 0] = (rotated_landmarks[:, 0] - x_bounding_rect) / w_bounding_rect
    rotated_landmarks[:, 1] = (rotated_landmarks[:, 1] - y_bounding_rect) / h_bounding_rect

    return rotated_landmarks


def normalize_landmarks_eyes(landmarks):
    """Normalize landmarks with respect to eyes and the nose instead of the whole landmarks' bounding box"""

    eyes_landmarks = landmarks[36:48]

    (x, y), (w, h), angle = cv2.minAreaRect(eyes_landmarks)

    if abs(angle + 90) < abs(angle):
        angle += 90

    landmarks_matrix = np.hstack([np.array(landmarks), np.ones((len(landmarks), 1))])
    center_point = tuple(landmarks[30])
    rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, scale=1)
    rotated_landmarks = np.dot(landmarks_matrix, rotation_matrix.T)[:, :2]
    rotated_landmarks_int = np.int0(rotated_landmarks)

    x_bounding_rect, y_bounding_rect, w_bounding_rect, h_bounding_rect = cv2.boundingRect(rotated_landmarks_int)
    rotated_landmarks[:, 0] = (rotated_landmarks[:, 0] - x_bounding_rect) / w_bounding_rect
    rotated_landmarks[:, 1] = (rotated_landmarks[:, 1] - y_bounding_rect) / h_bounding_rect

    return rotated_landmarks


def extract_landmarks(image) -> List[np.ndarray]:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(grayscale_image, 0)

    landmarks_faces = []

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        landmarks = predictor(grayscale_image, rect)
        landmarks = face_utils.shape_to_np(landmarks)

        landmarks_faces.append(landmarks)

    return landmarks_faces


def parse_image_path(path: Path):
    return path.stem.split("_")


def interpolate_landmarks(base_landmarks, target_landmarks, rate):
    return (1 - rate) * base_landmarks + rate * target_landmarks


def add_images_pairings(paths_indices_mapping, similar_images, training_pairs_indices, training_pairs_labels,
                        similarity, similar_pairs, sliding_window_size=15):
    for window in windowed(similar_images, sliding_window_size):
        for path1, path2 in itertools.combinations(window, 2):
            if (path1, path2) not in similar_pairs:
                index1 = paths_indices_mapping[path1]
                index2 = paths_indices_mapping[path2]

                training_pairs_indices = np.vstack([training_pairs_indices, np.array([index1, index2])])
                training_pairs_labels = np.append(training_pairs_labels, similarity)

                if similarity != -1:
                    similar_pairs.add((path1, path2))

    return training_pairs_indices, training_pairs_labels


def generate_images_from_video(src_path: Path, dst_folder_path: Path):
    subject, category = parse_image_path(src_path)
    video = cv2.VideoCapture(str(src_path))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_density = 100 / (num_frames - 1)
    current_frame = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        # cv2.imwrite(str(dst_folder_path.joinpath(f"{subject}_{category}_{int(current_frame)}.jpg")), frame)
        if category != "neutro":
            plt.imsave(str(dst_folder_path.joinpath(f"{subject}_{category}_{int(round(current_frame))}.png")), frame)
        else:
            plt.imsave(str(dst_folder_path.joinpath(f"{subject}_{category}{int(current_frame)}_{0}.png")), frame)
        # current_frame = min(current_frame + frame_density, 100.)
        current_frame += frame_density


def rotate_image(image_array: np.ndarray, degree: int):
    return sk.transform.rotate(image_array, degree, preserve_range=True).astype(np.uint8)


def horizontal_flip(image_array):
    return image_array[:, ::-1]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

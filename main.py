import itertools
import operator
from pathlib import Path
from typing import List

import cv2
import dlib
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt
from metric_learn import MMC, ITML, LMNN, MLKR, LFDA, NCA
import pickle
from joblib import dump, load

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


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


def generate_training_pairs(path: Path):
    # This code assumes that each image in the training path has only one face in it

    paths_indices_mapping = {}
    landmarks_matrix = np.empty((0, 68 * 2))

    training_pairs_indices = np.empty((0, 2), dtype=np.int16)
    training_pairs_labels = np.empty((0, 2), dtype=np.int16)

    for image_path in sorted(path.iterdir(), key=lambda p: (
            p.name.split("_")[:-1], 101 if "neutro" in p.name else int(p.stem.split("_")[-1]))):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        landmarks_list = extract_landmarks(image)

        if not landmarks_list:
            raise Exception(f"landmarks not found in image ({image_path})")

        for landmarks in landmarks_list:  # `landmarks_list` should contain only one element to properly populate `paths_indices_mapping`
            normalized_landmarks = normalize_landmarks(landmarks)

            landmarks_matrix = np.vstack([landmarks_matrix, normalized_landmarks.flatten()])
            paths_indices_mapping[image_path] = landmarks_matrix.shape[0] - 1

    neuter_images = list(filter(lambda f: "neutro" in f.name, path.iterdir()))
    occhiolinodx_images = list(filter(lambda f: "occhiolinodx" in f.name, path.iterdir()))
    occhiolinosx_images = list(filter(lambda f: "occhiolinosx" in f.name, path.iterdir()))
    cruccio_images = list(filter(lambda f: "cruccio" in f.name, path.iterdir()))
    sorriso_images = list(filter(lambda f: "sorriso" in f.name, path.iterdir()))
    sorrisino_images = list(filter(lambda f: "sorrisino" in f.name, path.iterdir()))
    gengive_images = list(filter(lambda f: "gengive" in f.name, path.iterdir()))
    bacio_images = list(filter(lambda f: "bacio" in f.name, path.iterdir()))

    all_images = [occhiolinodx_images,
                  occhiolinosx_images,
                  cruccio_images,
                  sorriso_images,
                  sorrisino_images,
                  gengive_images,
                  bacio_images]

    similar_pairs = set()

    for category_images in all_images:
        image_subjects = set(map(operator.itemgetter(0), map(parse_image_path, category_images)))

        for subject in image_subjects:
            subject_images = list(
                sorted(filter(lambda i: subject in i.name and "neutro" not in i.name, category_images),
                       key=lambda p: (
                           p.name.split("_")[:-1], 101 if "neutro" in p.name else int(p.stem.split("_")[-1]))))
            training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping, subject_images,
                                                                                training_pairs_indices,
                                                                                training_pairs_labels, 1,
                                                                                similar_pairs)

            images_25 = list(filter(lambda i: "25" in i.name, subject_images))
            for neuter_image in filter(lambda i: subject in i.name, neuter_images):
                for image_25 in images_25:
                    training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping,
                                                                                        [neuter_image, image_25],
                                                                                        training_pairs_indices,
                                                                                        training_pairs_labels, 1,
                                                                                        similar_pairs)

        levels = [25, 50, 75, 100]

        for level in levels:
            level_images = list(filter(lambda i: str(level) in i.name, category_images))
            training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping, level_images,
                                                                                training_pairs_indices,
                                                                                training_pairs_labels, 1, similar_pairs)

    for neuter_image_1, neuter_image_2 in itertools.combinations(neuter_images, 2):
        training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping,
                                                                            [neuter_image_1, neuter_image_2],
                                                                            training_pairs_indices,
                                                                            training_pairs_labels, 1, similar_pairs)

    all_images_flattened = list(itertools.chain.from_iterable(all_images))
    combinations = list(itertools.combinations(all_images_flattened, 2))
    for path1, path2 in combinations:
        if ((path1, path2) not in similar_pairs) and ((path2, path1) not in similar_pairs):
            training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping,
                                                                                [path1, path2],
                                                                                training_pairs_indices,
                                                                                training_pairs_labels, -1,
                                                                                similar_pairs)

    return landmarks_matrix, training_pairs_indices, training_pairs_labels


def add_images_pairings(paths_indices_mapping, similar_images, training_pairs_indices, training_pairs_labels,
                        similarity, similar_pairs):
    for path1, path2 in zip(similar_images, list(similar_images)[1:]):
        index1 = paths_indices_mapping[path1]
        index2 = paths_indices_mapping[path2]

        training_pairs_indices = np.vstack([training_pairs_indices, np.array([index1, index2])])
        training_pairs_labels = np.append(training_pairs_labels, similarity)

        if similarity != -1:
            similar_pairs.add((path1, path2))

    return training_pairs_indices, training_pairs_labels


def generate_training_supervised_dataset_categorical(path: Path):
    # This code assumes that each image in the training path has only one face in it

    landmarks_matrix = np.empty((0, 68 * 2))
    # training_pairs_labels = ["neutro", "occhiolinodx", "occhiolinosx", "cruccio", "sorriso", "sorrisino", "bacio",
    #                          "gengive"]
    training_paris_labels = []
    training_paris_labels = np.array(training_paris_labels, dtype="str")

    for image_path in sorted(path.iterdir(), key=lambda p: (
            p.name.split("_")[:-1], 101 if "neutro" in p.name else int(p.stem.split("_")[-1]))):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        landmarks_list = extract_landmarks(image)

        if not landmarks_list:
            raise Exception(f"landmarks not found in image ({image_path})")

        for landmarks in landmarks_list:  # `landmarks_list` should contain only one element to properly populate `paths_indices_mapping`
            normalized_landmarks = normalize_landmarks(landmarks)

            landmarks_matrix = np.vstack([landmarks_matrix, normalized_landmarks.flatten()])
            category = parse_image_path(image_path)[1]
            training_paris_labels = np.append(training_paris_labels, category)

    return landmarks_matrix, training_paris_labels


def generate_training_supervised_dataset_regression(path: Path):
    # This code assumes that each image in the training path has only one face in it

    landmarks_matrix = np.empty((0, 68 * 2))
    # training_pairs_labels = ["neutro", "occhiolinodx", "occhiolinosx", "cruccio", "sorriso", "sorrisino", "bacio",
    #                          "gengive"]
    training_paris_labels = []
    training_paris_labels = np.array(training_paris_labels, dtype=np.int8)

    for image_path in sorted(path.iterdir(), key=lambda p: (
            p.name.split("_")[:-1], 101 if "neutro" in p.name else int(p.stem.split("_")[-1]))):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        landmarks_list = extract_landmarks(image)

        if not landmarks_list:
            raise Exception(f"landmarks not found in image ({image_path})")

        for landmarks in landmarks_list:  # `landmarks_list` should contain only one element to properly populate `paths_indices_mapping`
            normalized_landmarks = normalize_landmarks(landmarks)

            landmarks_matrix = np.vstack([landmarks_matrix, normalized_landmarks.flatten()])
            deformation_index = parse_image_path(image_path)[2]
            if type(deformation_index) is str:
                deformation_index = 0
            training_paris_labels = np.append(training_paris_labels, deformation_index / 100)

    return landmarks_matrix, training_paris_labels


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


def generate_video_dataset(src_path: Path):
    dst_folder_path = src_path.parent.joinpath("dataset_video_images")
    for video in src_path.iterdir():
        generate_images_from_video(video, dst_folder_path)


def main():
    generate_video_dataset(Path("dataset_video").resolve())


    landmarks_matrix, training_pairs_labels = generate_training_supervised_dataset_regression(
        Path("dataset_video_images").resolve())
    model = MLKR()
    model.fit(landmarks_matrix, training_pairs_labels)
    # model = ITML(preprocessor=landmarks_matrix)
    # model.fit(training_pairs_indices, training_pairs_labels)

    dump(model, 'model_MLKR_video.joblib')


if __name__ == '__main__':
    main()

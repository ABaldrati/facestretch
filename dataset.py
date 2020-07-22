import itertools
import operator
import sys
from pathlib import Path
from random import sample
from typing import List

import cv2
import numpy as np
from more_itertools import windowed

from utils import parse_image_path, extract_landmarks, normalize_landmarks, interpolate_landmarks, add_images_pairings, \
    generate_images_from_video, normalize_landmarks_eyes


def generate_weakly_supervised_interpolated_dataset(src_path: Path, rates: List[int], normalize_eyes=False):
    if 0 in rates:
        rates.remove(0)
    if 1 not in rates:
        rates.append(1)

    action_images = filter(lambda i: "neutro" not in i.name, src_path.iterdir())

    landmarks_matrix = np.empty((0, 68 * 2))
    landmark_indices_mapping = {}
    training_pairs_indices = np.empty((0, 2), dtype=np.int16)
    training_pairs_labels = np.empty((0, 1), dtype=np.int16)

    neuter_landmarks = {}

    subjects = set(map(lambda i: parse_image_path(i)[0], src_path.iterdir()))
    actions = set(map(lambda i: parse_image_path(i)[1], src_path.iterdir()))
    actions.remove("neutro")

    for subject in subjects:
        print(f"Importing subject {subject}")
        subject_neuter_image_path = list(src_path.glob(f"{subject}_neutro.*"))[0]
        subject_neuter_image = cv2.imread(str(subject_neuter_image_path), cv2.IMREAD_UNCHANGED)
        landmarks_found = extract_landmarks(subject_neuter_image)

        if not landmarks_found:
            print(f"Neuter landmarks not found for subject {subject}. Skipping...", file=sys.stderr)
            action_images = list(filter(lambda i: subject not in i.name, list(action_images)))

            continue

        if normalize_eyes:
            neuter_subject_landmarks = normalize_landmarks_eyes(landmarks_found[0]).flatten()
        else:
            neuter_subject_landmarks = normalize_landmarks(landmarks_found[0]).flatten()

        print(f"Extracted landmarks from {subject_neuter_image_path}")

        neuter_landmarks[subject] = neuter_subject_landmarks
        landmarks_matrix = np.vstack([landmarks_matrix, np.zeros(len(neuter_subject_landmarks))])
        landmark_indices_mapping[f"{subject}_neutro"] = landmarks_matrix.shape[0] - 1

    for image_path in action_images:
        print(f"Processing {image_path}")
        subject, action = parse_image_path(image_path)
        subject_action_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        landmarks_found = extract_landmarks(subject_action_image)

        if not landmarks_found:
            print(f"Action landmarks ({action}) not found for subject {subject}. Skipping...", file=sys.stderr)
            continue

        action_subject_landmarks = normalize_landmarks(landmarks_found[0]).flatten()
        neuter_subject_landmarks = neuter_landmarks[subject]

        for rate in rates:
            interpolated_subject_landmarks = interpolate_landmarks(neuter_subject_landmarks, action_subject_landmarks, rate) - neuter_subject_landmarks
            landmarks_matrix = np.vstack([landmarks_matrix, interpolated_subject_landmarks])
            landmark_indices_mapping[f"{subject}_{action}_{rate}"] = landmarks_matrix.shape[0] - 1

    for action in actions:
        for rate in rates:
            for subj1, subj2 in itertools.combinations(subjects, 2):
                subj1_landmarks_index = landmark_indices_mapping.get(f"{subj1}_{action}_{rate}")
                subj2_landmarks_index = landmark_indices_mapping.get(f"{subj2}_{action}_{rate}")

                if not (subj1_landmarks_index and subj2_landmarks_index):
                    continue

                training_pairs_indices = np.vstack([training_pairs_indices,
                                                    np.array([subj1_landmarks_index, subj2_landmarks_index])])
                training_pairs_labels = np.append(training_pairs_labels, 1)

    for act1, act2 in itertools.combinations(actions, 2):
        for rate in rates:
            for subj1, subj2 in itertools.combinations(subjects, 2):
                subj1_landmarks_index = landmark_indices_mapping.get(f"{subj1}_{act1}_{rate}")
                subj2_landmarks_index = landmark_indices_mapping.get(f"{subj2}_{act2}_{rate}")

                if not (subj1_landmarks_index and subj2_landmarks_index):
                    continue

                training_pairs_indices = np.vstack([training_pairs_indices,
                                                    np.array([subj1_landmarks_index, subj2_landmarks_index])])
                training_pairs_labels = np.append(training_pairs_labels, -1)

    for action in actions:
        for rate in rates:
            for subj1, subj2 in itertools.combinations(subjects, 2):
                subj1_landmarks_index = landmark_indices_mapping.get(f"{subj1}_{action}_{rate}")
                subj2_landmarks_index = landmark_indices_mapping.get(f"{subj2}_neutro")

                if not (subj1_landmarks_index and subj2_landmarks_index):
                    continue

                training_pairs_indices = np.vstack([training_pairs_indices,
                                                    np.array([subj1_landmarks_index, subj2_landmarks_index])])
                training_pairs_labels = np.append(training_pairs_labels, -1)

    return landmarks_matrix, training_pairs_indices, training_pairs_labels


def generate_training_weakly_supervised(path: Path, normalize_eyes=False):
    # This code assumes that each image in the training path has only one face in it

    paths_indices_mapping = {}
    landmarks_matrix = np.empty((0, 68 * 2))

    training_pairs_indices = np.empty((0, 2), dtype=np.int16)
    training_pairs_labels = np.empty((0, 1), dtype=np.int16)

    for image_path in sorted(path.iterdir(), key=lambda p: (
            p.name.split("_")[:-1], 101 if "neutro" in p.name else int(p.stem.split("_")[-1]))):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        landmarks_list = extract_landmarks(image)

        if not landmarks_list:
            raise Exception(f"landmarks not found in image ({image_path})")

        for landmarks in landmarks_list:  # `landmarks_list` should contain only one element to properly populate `paths_indices_mapping`
            if normalize_eyes:
                normalized_landmarks = normalize_landmarks_eyes(landmarks)
            else:
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

            lows = range(5)
            images_low = list(filter(lambda i: any(str(level) in i.name for level in lows), subject_images))
            for neuter_image in filter(lambda i: subject in i.name, neuter_images):
                for image_low in images_low:
                    training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping,
                                                                                        [neuter_image, image_low],
                                                                                        training_pairs_indices,
                                                                                        training_pairs_labels, 1,
                                                                                        similar_pairs,
                                                                                        sliding_window_size=2)

        levels = range(101)

        for window_levels in windowed(levels, 5):
            level_images = list(filter(lambda i: any(str(level) in i.name for level in window_levels), category_images))
            training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping, level_images,
                                                                                training_pairs_indices,
                                                                                training_pairs_labels, 1, similar_pairs,
                                                                                sliding_window_size=len(level_images))

    for neuter_image_1, neuter_image_2 in itertools.combinations(neuter_images, 2):
        training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping,
                                                                            [neuter_image_1, neuter_image_2],
                                                                            training_pairs_indices,
                                                                            training_pairs_labels, 1, similar_pairs,
                                                                            sliding_window_size=2)

    all_images_flattened = list(itertools.chain.from_iterable(all_images))
    combinations = list(sample(list(itertools.combinations(all_images_flattened, 2)), len(similar_pairs)))
    for path1, path2 in combinations:
        if ((path1, path2) not in similar_pairs) and ((path2, path1) not in similar_pairs):
            training_pairs_indices, training_pairs_labels = add_images_pairings(paths_indices_mapping,
                                                                                [path1, path2],
                                                                                training_pairs_indices,
                                                                                training_pairs_labels, -1,
                                                                                similar_pairs,
                                                                                sliding_window_size=2)

    return landmarks_matrix, training_pairs_indices, training_pairs_labels


def generate_training_supervised_dataset_categorical(path: Path, normalize_eyes=False):
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
            if normalize_eyes:
                normalized_landmarks = normalize_landmarks_eyes(landmarks)
            else:
                normalized_landmarks = normalize_landmarks(landmarks)

            landmarks_matrix = np.vstack([landmarks_matrix, normalized_landmarks.flatten()])
            category = parse_image_path(image_path)[1]
            training_paris_labels = np.append(training_paris_labels, category)

    return landmarks_matrix, training_paris_labels


def generate_training_supervised_dataset_regression(path: Path, normalize_eyes=False):
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
            if normalize_eyes:
                normalized_landmarks = normalize_landmarks_eyes(landmarks)
            else:
                normalized_landmarks = normalize_landmarks(landmarks)

            landmarks_matrix = np.vstack([landmarks_matrix, normalized_landmarks.flatten()])
            deformation_index = parse_image_path(image_path)[2]
            if type(deformation_index) is str:
                deformation_index = 0
            training_paris_labels = np.append(training_paris_labels, deformation_index / 100)

    return landmarks_matrix, training_paris_labels


def generate_video_dataset(src_path: Path):
    dst_folder_path = src_path.parent.joinpath("dataset_video_images")
    for video in src_path.iterdir():
        generate_images_from_video(video, dst_folder_path)
import itertools
import operator
import sys
from pathlib import Path
from random import sample
from typing import List

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from joblib import dump
from metric_learn import ITML

from dataset import generate_weakly_supervised_interpolated_dataset

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def main():
    # generate_video_dataset(Path("dataset_video").resolve())

    # landmarks_matrix, training_pairs_labels = generate_training_supervised_dataset_regression(
    #    Path("dataset_video_images").resolve())

    saved_landmarks_matrix = Path("landmarks_matrix.npy")
    saved_training_pairs_indices = Path("training_pairs_indices.npy")
    saved_training_pairs_labels = Path("training_pairs_labels.npy")

    if not saved_landmarks_matrix.exists() or not saved_training_pairs_indices.exists() or not saved_training_pairs_labels.exists():
        print("Generating dataset...")

        landmarks_matrix, training_pairs_indices, training_pairs_labels = generate_weakly_supervised_interpolated_dataset(
            Path("dataset_new").resolve(), [0.5, 1])

        np.save(str(saved_landmarks_matrix), landmarks_matrix)
        np.save(str(saved_training_pairs_indices), training_pairs_indices)
        np.save(str(saved_training_pairs_labels), training_pairs_labels)
    else:
        print("Loading dataset from filesystem...")
        landmarks_matrix = np.load(saved_landmarks_matrix)
        training_pairs_indices = np.load(saved_training_pairs_indices)
        training_pairs_labels = np.load(saved_training_pairs_labels)

    print("Starting training...")
    model = SDML(verbose=True, preprocessor=landmarks_matrix, prior="identity", balance_param=0.4) #, convergence_threshold=1e-5)
    model.fit(training_pairs_indices, training_pairs_labels)
    # model = ITML(preprocessor=landmarks_matrix)
    # model.fit(training_pairs_indices, training_pairs_labels)

    dump(model, 'model_SDML.joblib')


if __name__ == '__main__':
    main()

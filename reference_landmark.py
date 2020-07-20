from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import normalize_landmarks, extract_landmarks, parse_image_path


def save_action_reference_landmarks(dataset_path: Path, reference_landmark_folder: Path, action):
    action_images = list(filter(lambda i: action in i.name, dataset_path.iterdir()))

    action_landmarks = []
    with tqdm(action_images) as t:
        for image in t:
            t.set_postfix_str(str(image), refresh=True)
            subject, _ = parse_image_path(image)
            neuter_landmarks_path = list(dataset_path.glob(f"{subject}_neutro*"))[0]
            neuter_image = cv2.imread(str(neuter_landmarks_path), cv2.IMREAD_UNCHANGED)
            neuter_image_landmarks = extract_landmarks(neuter_image)

            image = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            image_landmarks = extract_landmarks(image)

            if not (image_landmarks and neuter_image_landmarks):
                continue

            neuter_landmarks = normalize_landmarks(neuter_image_landmarks[0])
            action_landmarks.append(normalize_landmarks(image_landmarks[0]) - neuter_landmarks)

    reference_landmark = sum(action_landmarks) / len(action_landmarks)
    reference_landmark = reference_landmark.flatten()
    output_path = reference_landmark_folder.joinpath(f"{action}.npy")
    np.save(str(output_path), reference_landmark)


def generate_all_reference_landmark(dataset_path: Path, reference_landmark_folder: Path):
    actions = set(map(lambda i: parse_image_path(i)[1], dataset_path.iterdir()))
    for action in actions:
        save_action_reference_landmarks(dataset_path, reference_landmark_folder, action)


def main():
    reference_landmark_folder_path = Path("reference_landmark_folder")
    reference_landmark_folder_path.mkdir(parents=True, exist_ok=True)
    generate_all_reference_landmark(Path("dataset_new").resolve(), reference_landmark_folder_path)


if __name__ == '__main__':
    main()

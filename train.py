from datetime import datetime
from pathlib import Path

from joblib import dump
from metric_learn import ITML, LMNN, NCA, LFDA, SDML, MMC

from dataset import generate_weakly_supervised_interpolated_dataset, generate_training_supervised_dataset_categorical
from detect_landmarks import MODELS_PATH


def main():
    NAME_MODEL_TO_TRAIN = "NCA"
    # weakly-supervised possibilities "ITML" , "SDML" , "MMC"
    # supervised possibilities: "LMNN" , "NCA" , "LFDA"

    rates_list = [0.5, 1]

    training_start_time = datetime.now().isoformat()

    MODELS_PATH.mkdir(exist_ok=True)

    if NAME_MODEL_TO_TRAIN == "ITML":
        landmarks_matrix, training_pairs_indices, training_pairs_labels = generate_weakly_supervised_interpolated_dataset(
            Path("dataset_metric_learning").resolve(), rates_list)
        model = ITML(preprocessor=landmarks_matrix, max_iter=20_000, verbose=True)
        model.fit(training_pairs_indices, training_pairs_labels)
        dump(model, str(MODELS_PATH.joinpath(f'ITML_{training_start_time}.joblib')))

    elif NAME_MODEL_TO_TRAIN == "SDML":  # may not work due to graphical lasso solver
        landmarks_matrix, training_pairs_indices, training_pairs_labels = generate_weakly_supervised_interpolated_dataset(
            Path("dataset_metric_learning").resolve(), rates_list)
        model = SDML(preprocessor=landmarks_matrix, verbose=True, prior="identity", balance_param=0.2)
        model.fit(training_pairs_indices, training_pairs_labels)
        dump(model, str(MODELS_PATH.joinpath(f'SDML_{training_start_time}.joblib')))

    elif NAME_MODEL_TO_TRAIN == "MMC":
        landmarks_matrix, training_pairs_indices, training_pairs_labels = generate_weakly_supervised_interpolated_dataset(
            Path("dataset_metric_learning").resolve(), rates_list)
        model = MMC(preprocessor=landmarks_matrix, verbose=True, convergence_threshold=1e-5)
        model.fit(training_pairs_indices, training_pairs_labels)
        MODELS_PATH.mkdir(exist_ok=True)
        dump(model, str(MODELS_PATH.joinpath(f'MMC_{training_start_time}.joblib')))

    elif NAME_MODEL_TO_TRAIN == "LMNN":
        landmarks_matrix, training_labels = generate_training_supervised_dataset_categorical(
            Path("dataset_metric_learning").resolve())
        model = LMNN(max_iter=10000, convergence_tol=1e-5, verbose=True)
        model.fit(landmarks_matrix, training_labels)
        dump(model, str(MODELS_PATH.joinpath(f'LMNN_{training_start_time}.joblib')))

    elif NAME_MODEL_TO_TRAIN == "NCA":
        landmarks_matrix, training_labels = generate_training_supervised_dataset_categorical(
            Path("dataset_metric_learning").resolve())
        model = NCA(verbose=True)
        model.fit(landmarks_matrix, training_labels)
        dump(model, str(MODELS_PATH.joinpath(f'NCA_{training_start_time}.joblib')))

    elif NAME_MODEL_TO_TRAIN == "LFDA":
        landmarks_matrix, training_labels = generate_training_supervised_dataset_categorical(
            Path("dataset_metric_learning").resolve())
        model = LFDA()
        model.fit(landmarks_matrix, training_labels)
        dump(model, str(MODELS_PATH.joinpath(f'LFDA_{training_start_time}.joblib')))


if __name__ == '__main__':
    main()

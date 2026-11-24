# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "diabetic_data.csv"  # Adapter au vrai nom
MODEL_PATH = BASE_DIR / "model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.joblib"

PYTHON_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


# ==========================================
# TASKS : INSTALLATION ET QUALITÉ DU CODE
# ==========================================


@task(name="Install Dependencies")
def install_dependencies_task():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=BASE_DIR,
        check=True,
    )


@task(name="Code Formatting")
def format_code_task():
    subprocess.run(
        [sys.executable, "-m", "black"] + PYTHON_FILES, cwd=BASE_DIR, check=True
    )


@task(name="Code Quality")
def code_quality_task():
    subprocess.run(
        [sys.executable, "-m", "flake8"] + PYTHON_FILES, cwd=BASE_DIR, check=True
    )


@task(name="Code Security")
def code_security_task():
    subprocess.run(
        [sys.executable, "-m", "bandit", "-q", "--skip", "B404,B603,B607"]
        + PYTHON_FILES,
        cwd=BASE_DIR,
        check=True,
    )


@task(name="Unit Tests")
def unit_tests_task():
    subprocess.run([sys.executable, "-m", "pytest"], cwd=BASE_DIR, check=True)


# ==========================================
# TASKS : MACHINE LEARNING
# ==========================================


@task(name="Prepare Data")
def prepare_data_task():
    return prepare_data(file_path=str(DATA_PATH), test_size=0.2, random_state=42)


@task(name="Train Model")
def train_model_task(X_train, y_train):
    return train_model(X_train, y_train, random_state=42)


@task(name="Save Model")
def save_model_task(model, preprocessor):
    save_model(model, preprocessor, str(MODEL_PATH), str(PREPROCESSOR_PATH))


@task(name="Load Model")
def load_model_task():
    return load_model(str(MODEL_PATH), str(PREPROCESSOR_PATH))


@task(name="Evaluate Model")
def evaluate_model_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


# ==========================================
# FLOW 1 : ALL
# ==========================================


@flow(name="all")
def all_flow():

    install_dependencies_task()

    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()

    data = prepare_data_task()

    X_train, X_test, y_train, y_test, preprocessor, feature_names = data

    model = train_model_task(X_train, y_train)

    save_model_task(model, preprocessor)

    metrics = evaluate_model_task(model, X_test, y_test)

    print("Pipeline complet terminé avec succès !")

    return metrics


# ==========================================
# FLOW 2 : ENTRAINEMENT
# ==========================================


@flow(name="entrainement")
def train_flow():

    data = prepare_data_task()

    X_train, X_test, y_train, y_test, preprocessor, feature_names = data

    model = train_model_task(X_train, y_train)

    print("Entraînement terminé avec succès !")

    return model


# ==========================================
# FLOW 3 : EVALUATE
# ==========================================


@flow(name="evaluate")
def evaluate_flow():

    model, preprocessor = load_model_task()

    data = prepare_data_task()

    X_train, X_test, y_train, y_test, preprocessor_data, feature_names = data

    metrics = evaluate_model_task(model, X_test, y_test)

    print("Évaluation terminée avec succès !")

    return metrics


# ==========================================
# FLOW 4 : CODE
# ==========================================


@flow(name="code")
def code_flow():

    install_dependencies_task()

    format_code_task()
    code_quality_task()
    code_security_task()
    unit_tests_task()

    print("Vérification du code terminée avec succès !")


# ==========================================
# MAIN : CHOIX DU FLOW
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")

    parser.add_argument(
        "--flow",
        choices=["all", "entrainement", "evaluate", "code"],
        required=True,
        help="Flow à exécuter",
    )

    args = parser.parse_args()

    if args.flow == "all":
        all_flow()

    elif args.flow == "entrainement":
        train_flow()

    elif args.flow == "evaluate":
        evaluate_flow()

    elif args.flow == "code":
        code_flow()

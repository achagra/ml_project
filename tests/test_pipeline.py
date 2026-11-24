import os


def test_project_files_exist():
    assert os.path.exists("main.py")
    assert os.path.exists("model_pipeline.py")
    assert os.path.exists("pipeline_prefect.py")

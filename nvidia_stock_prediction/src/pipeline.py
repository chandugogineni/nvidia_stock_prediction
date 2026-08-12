import subprocess
import sys
import os


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def run_step(script):
    print("\n")
    print("=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)

    script_path = os.path.join(
        PROJECT_ROOT,
        "src",
        script
    )

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"\nFAILED: {script}"
        )

    print(f"\nCOMPLETED: {script}")


def main():

    print("=" * 70)
    print("NVIDIA STOCK PREDICTION - COMPLETE PIPELINE")
    print("=" * 70)

    steps = [

        # --------------------------------------
        # DATA
        # --------------------------------------

        "data_download.py",

        "feature_engineering.py",

        # --------------------------------------
        # NEWS
        # --------------------------------------

        "news_collection.py",

        "news_preprocessing.py",

        # --------------------------------------
        # OLLAMA LLM
        # --------------------------------------

        "generate_llm_features.py",

        "merge_llm_features.py",

        # --------------------------------------
        # DATA SPLITTING
        # --------------------------------------

        "data_splitting.py",

        # --------------------------------------
        # MODELS
        # --------------------------------------

        "baseline.py",

        "random_forest_training.py",

        "xgboost_training.py",

        # --------------------------------------
        # PREDICTION
        # --------------------------------------

        "prediction.py",
    ]

    for step in steps:

        run_step(step)

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
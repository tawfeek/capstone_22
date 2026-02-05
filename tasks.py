# tasks.py
import os
import shutil
import glob
from invoke import task
# Define default paths
RAW_DATA = "data/raw/listings LA.csv data/raw/listings NYC.csv"
CLEAN_DATA = "data/processed/listings_combined_clean.csv"
MODEL_PATH = "models/model.joblib"
PREDICTIONS = "results/predictions.csv"


@task
def clean(c):
    """
    Remove all generated files (processed data, models, results).
    """
    patterns = [
        "data/processed/*.csv",
        "data/processed/*.json",
        "models/*.joblib",
        "results/*.csv",
        "results/*.json",
        "wandb"
    ]

    print("🧹 Cleaning artifacts...")
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                print(f"   Removed: {filepath}")
            except Exception as e:
                print(f"   Error removing {filepath}: {e}")
    print("✅ Clean complete.")


@task
def preprocess(c, inputs=RAW_DATA, output=CLEAN_DATA, inference=False):
    """
    Run the preprocessing step.
    Usage: inv preprocess --inputs "file1.csv file2.csv"
    """
    print(f"🔨 Running Preprocessing on: {inputs}")

    # Handle both string and list inputs, and properly quote paths with spaces
    if isinstance(inputs, list):
        quoted_inputs = ' '.join(f'"{path}"' for path in inputs)
    elif isinstance(inputs, str):
        # Split by spaces but preserve quoted paths
        paths = inputs.split()
        quoted_inputs = ' '.join(f'"{path}"' for path in paths)
    else:
        quoted_inputs = str(inputs)

    cmd = f'python preprocess.py {quoted_inputs} --out-csv "{output}"'

    if inference:
        cmd += " --keep-missing-target"

    c.run(cmd, echo=True)


@task
def train(c, input_csv=CLEAN_DATA, model="random_forest",
          test_size=0.2, cv=5, n_iter=20, tune=True, wandb=False):
    """
    Train the model. Supports all train.py arguments.
    """
    print(f"🤖 Training {model} (CV={cv}, Iter={n_iter})...")

    # Build the command dynamically based on arguments
    cmd = (
        f"python train.py "
        f"--in-csv {input_csv} "
        f"--model-type {model} "
        f"--test-size {test_size} "
        f"--cv-folds {cv} "
        f"--n-iter {n_iter} "
        f"--out-model {MODEL_PATH}"
    )

    # Handle boolean flags
    if not tune:
        cmd += " --no-tune"
    if wandb:
        cmd += " --wandb"

    c.run(cmd, echo=True)

@task
def predict(c, input_csv=CLEAN_DATA, model_path=MODEL_PATH, output=PREDICTIONS):
    """
    Run inference using a trained model.
    """
    print(f"🔮 Predicting on {input_csv}...")
    cmd = (
        f"python predict.py "
        f"--in-csv {input_csv} "
        f"--model-path {model_path} "
        f"--out-csv {output}"
    )
    c.run(cmd, echo=True)

@task
def summary(c, pred_csv=PREDICTIONS):
    """
    Generate a summary of the predictions.
    """
    print("📊 Generating summary...")
    c.run(f"python results.py --pred-csv {pred_csv}", echo=True)


@task(pre=[clean])
def pipeline(c, model="random_forest", test_size=0.2, cv=5, n_iter=20, tune=True, wandb=True,is_trainig=True):
    """
    Run the full end-to-end pipeline with all training configuration options.
    """
    print("🚀 Starting full pipeline run...")

    # 1. Preprocess
    preprocess(c, inputs=RAW_DATA, output=CLEAN_DATA,is_training=is_trainig)

    # 2. Train (passing all arguments down)
    train(c, input_csv=CLEAN_DATA, model=model,
          test_size=test_size, cv=cv, n_iter=n_iter, tune=tune, wandb=wandb)

    # 3. Predict & Summary
    predict(c, input_csv=CLEAN_DATA, model_path=MODEL_PATH)
    summary(c)

    print("\n🎉 Pipeline finished!")

@task
def test_inference(c):
    """
    Simulate a production inference run on new raw data.
    """
    print("🧪 Testing Inference Pipeline...")

    # Simulate "new" data by just using one of the raw files
    # Note: We use a different output path to avoid overwriting training data
    test_input = "data/raw/listingsNYC.csv"
    temp_clean = "data/processed/inference_test.csv"
    temp_preds = "results/inference_test_preds.csv"

    if not os.path.exists(test_input):
        print(f"❌ Error: {test_input} not found. Cannot run test.")
        return

    # 1. Preprocess in INFERENCE mode (loads existing encoders if logic supported,
    # but based on your current preprocess.py, it cleans data without fitting new logic)
    # Note: Your current preprocess.py doesn't load a saved preprocessor (that's in the pipeline),
    # but it does cleaning. The 'inference' flag we added just sets is_training=False.
    preprocess(c, inputs=test_input, output=temp_clean, inference=True)

    # 2. Predict using the saved model pipeline
    predict(c, input_csv=temp_clean, output=temp_preds)

    print(f"✅ Inference test complete. Predictions saved to {temp_preds}")
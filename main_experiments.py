from invoke import Context
from tasks import pipeline, preprocess, train, predict, summary, clean

def main():
    # 1. Create a Context (Invoke needs this to execute shell commands)
    c = Context()

    # 2. Configure which steps to run
    # Set to True to run that step, False to skip
    steps = {
        "clean": True,          # Clean artifacts before running
        "preprocess": True,      # Run preprocessing only
        "train": True,          # Run training
        "predict": False,        # Run prediction
        "summary": False,        # Generate summary
        "full_pipeline": False,  # Run full pipeline (overrides individual steps)
    }

    # 3. Define your parameters here (instead of typing them in CLI)
    params = {
        # Preprocessing params
        "inputs": ["data/raw/listings LA.csv", "data/raw/listings NYC.csv"],
        "output": "data/processed/listings_combined_clean.csv",
        "inference": False,  # Set True to keep rows with missing target
        # Training params
        "is_training":False,
        "model": "hist_gradient_boosting",
        "test_size": 0.2,
        "cv": 5,
        "n_iter": 5,  # Number of Optuna trials
        "tune": True,
        "wandb": True
    }
    if steps["train"] or steps["full_pipeline"]:
        params["is_training"]=True
    print(f"🚀 Running experiments from main_experiments.py")
    print(f"Steps enabled: {[k for k, v in steps.items() if v]}")
    print(f"Parameters: {params}\n")

    # 4. Execute steps based on configuration
    if steps["full_pipeline"]:
        print("=" * 60)
        print("RUNNING FULL PIPELINE")
        print("=" * 60)
        pipeline(
            c,
            model=params["model"],
            test_size=params["test_size"],
            cv=params["cv"],
            n_iter=params["n_iter"],
            tune=params["tune"],
            wandb=params["wandb"],
            istraining=params["is_training"]
        )
    else:
        # Run individual steps
        if steps["clean"]:
            print("=" * 60)
            print("STEP 1: CLEANING")
            print("=" * 60)
            clean(c)
            print()

        if steps["preprocess"]:
            print("=" * 60)
            print("STEP 2: PREPROCESSING")
            print("=" * 60)
            preprocess(
                c,
                inputs=params["inputs"],
                output=params["output"],
                inference=params["inference"],
                is_training=params["is_training"]
            )
            print()

        if steps["train"]:
            print("=" * 60)
            print("STEP 3: TRAINING")
            print("=" * 60)
            train(
                c,
                input_csv=params["output"],
                model=params["model"],
                test_size=params["test_size"],
                cv=params["cv"],
                n_iter=params["n_iter"],
                tune=params["tune"],
                wandb=params["wandb"]
            )
            print()

        if steps["predict"]:
            print("=" * 60)
            print("STEP 4: PREDICTION")
            print("=" * 60)
            predict(c)
            print()

        if steps["summary"]:
            print("=" * 60)
            print("STEP 5: SUMMARY")
            print("=" * 60)
            summary(c)
            print()

    print("\n✅ Experiment run complete!")

if __name__ == "__main__":
    main()
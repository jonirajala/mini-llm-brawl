import json
import matplotlib.pyplot as plt
import argparse
import os


def get_latest_file(directory):
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    files = [f for f in files if f.endswith(".json")]
    if not files:
        return None
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    return latest_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a specific run.")
    parser.add_argument(
        "file_name", nargs="?", help="Name of the run to load", default=None
    )
    args = parser.parse_args()
    file_name = args.file_name

    if file_name is None:
        latest_file = get_latest_file("losses")
        if latest_file:
            file_name = "losses/" + latest_file
        else:
            print("No loss files found in the losses directory.")
            exit(1)
    else:
        file_name = "losses/" + file_name + ".json"

    # Load all losses from the single file
    try:
        with open(file_name, "r") as f:
            all_data = json.load(f)
            all_losses = all_data.get("train_losses", {})
            all_val_losses = all_data.get("val_losses", {})
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        exit(0)

    # Plot the training losses
    plt.figure(figsize=(12, 6))
    for model, model_losses in all_losses.items():
        plt.plot(model_losses, label=f"{model} train loss")

    for model, val_model_losses in all_val_losses.items():
        plt.plot(
            [i * 50 for i in range(len(val_model_losses))],
            val_model_losses,
            label=f"{model} val loss",
            linestyle="--",
        )

    plt.title("Model Training and Validation Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

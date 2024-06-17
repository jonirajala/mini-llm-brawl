import json
import matplotlib.pyplot as plt
import os

models = ["gpt", "llama", "mistral", "baseline_transformer", "gemma"]
losses = []

# Load the losses for each model
for model in models:
    try:
        with open(f'losses_{model}.json', 'r') as f:
            loaded_losses = json.load(f)
            losses.append((model, loaded_losses))
    except FileNotFoundError:
        print(f"File for model {model} not found. Skipping.")

# Plot the losses
plt.figure(figsize=(10, 6))
for model, model_losses in losses:
    plt.plot(model_losses, label=model)

plt.title('Model Training Losses')
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

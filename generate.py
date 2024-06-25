import torch
import os
import re
import tiktoken
from models import get_model
from train import get_config


def load_models(parameter_count, enc, directory='trained_models'):
    models = {}
    pattern = re.compile(r'(.+)-(\d+)M_(\d+)iters\.pt')
    lower_bound = parameter_count - 5
    upper_bound = parameter_count + 5
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            model_name, params, iters = match.groups()
            params = int(params)
            if lower_bound <= params <= upper_bound:
                model_path = os.path.join(directory, filename)

                model_name_split = model_name.split("-")
                model_name_raw, ind = model_name_split if len(model_name_split) > 1 else model_name_split[0], 0
                
                model = get_model(model_name_raw)
                configs = get_config(enc.n_vocab, model, parameter_count)
                config = configs[int(ind)]
                model = model(config)

                weights = torch.load(model_path)
                model.load_state_dict(weights)
                models[model_name] = model
    
    return models

def generate_text_with_models(models, enc,  device='mps'):
    for model_name, model in models.items():
        print(f"----Model: {model_name}----")
        model = model.to(device)
        model.eval()
        inp = torch.tensor(enc.encode("And that is  ")).to(device)
        with torch.no_grad():
            gen_text = model.generate(inp).detach().cpu().numpy()
        gen_text = enc.decode(gen_text)
        print(gen_text, "\n")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device =  "mps"
    else:
        device = "cpu"
        
    enc = tiktoken.get_encoding("gpt2")
    parameter_count = 100  # Replace with the desired parameter count
    print(f"Searching {parameter_count}M models")
    models = load_models(parameter_count, enc)
    print(f"loaded {len(models)} models")
    generate_text_with_models(models, enc, device)

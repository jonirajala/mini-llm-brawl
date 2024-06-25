import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_transformer import Transformer
from gemma import Gemma
from gpt import GPT
from llama import LLama
from mistral import Mistral
from mixtral import Mixtral
from custom import Custom


model_names = ["gpt", "llama", "mistral", "baseline_transformer", "gemma", "mixtral", "custom"]

def get_model(model_name):
    model_classes = {
        "gpt": GPT,
        "mistral": Mistral,
        "llama": LLama,
        "gemma": Gemma,
        "baseline_transformer": Transformer,
        "mixtral": Mixtral,
        "custom": Custom
    }
    return model_classes.get(model_name, None)
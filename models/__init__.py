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
from bitnet import BitNet

model_names = [
    "gpt",
    "llama",
    "mistral",
    "baseline_transformer",
    "gemma",
    "mixtral",
    "custom",
    "bitnet",
]


def get_model(model_name):
    model_classes = {
        "gpt": GPT,
        "mistral": Mistral,
        "llama": LLama,
        "gemma": Gemma,
        "baseline_transformer": Transformer,
        "mixtral": Mixtral,
        "custom": Custom,
        "bitnet": BitNet
    }
    return model_classes.get(model_name, None)

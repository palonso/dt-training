import torch
import torch.nn as nn

from models import *


class ModelFactory:
    def create(self, architecture):
        model = eval(f"{architecture.lower()}.{architecture}")()
        return model

import logging
from trainers import *

class TrainerFactory:
    def __init__(self):
        self.logger = logging.getLogger('TrainManager.TrainerFactory')

    def create(self, name, *args):
        try:
            return eval(f"{name.lower()}.{name}")(*args)
        except NameError:
            self.logger.error(f"the trainer `{name.lower()}.{name}` is not registered in the factory."
                               " may be missing from `trainers/__init__.py`?")
            # no much more to do
            # just exit for a cleaner log
            exit(1)
import logging
from models import *


class ModelFactory:
    def __init__(self):
        self.logger = logging.getLogger('TrainManager.ModelFactory')

    def create(self, name, *args, **kwargs):
        try:
            return eval(f"{name.lower()}.{name}")(*args, **kwargs)
        except NameError:
            self.logger.error(f"the model `{name.lower()}.{name}` is not registered in the factory."
                               " may be missing from `models/__init__.py`?")
            # no much more to do
            # just exit for a cleaner log
            exit(1)

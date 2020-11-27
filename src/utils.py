import logging
import sys
import argparse


CONSOLE_FORMATTER = logging.Formatter("%(levelname)s — %(message)s")
FILE_FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

# logging utils
# basedon: https://www.toptal.com/python/in-depth-python-logging
def get_console_handler(level):
   handler = logging.StreamHandler(sys.stdout)
   handler.setLevel(level)
   handler.setFormatter(CONSOLE_FORMATTER)
   return handler

def get_file_handler(path):
   handler = logging.FileHandler(path)
   handler.setFormatter(FILE_FORMATTER)
   return handler

def get_logger(logger_name, path):
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG) # better to have too much log than not enough
   logger.addHandler(get_file_handler(path))
   # with this pattern, it's rarely necessary to propagate the error up to parent
   logger.propagate = False
   return logger

# a fix for
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

from src.exception import CustomException
from src.logger import logging
import sys

# checking error handling and logging

try:
    a = 1/0
    logging.info('division done correctly')
except Exception as e:
    raise CustomException(e,sys) from e

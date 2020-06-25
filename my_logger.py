"""
@author: Li Xi
@file: my_logger.py
@time: 2020/2/7 20:56
@desc:
"""
import time
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(level=logging.DEBUG)

time_str = time.strftime('%Y-%m-%d', time.localtime(time.time()))
log_path = os.path.join('log', '{}.log'.format(time_str))
file_handler = logging.FileHandler(log_path, mode='a', encoding='utf=8')
file_handler.setFormatter(formatter)
file_handler.setLevel(level=logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

print(os.getcwd())
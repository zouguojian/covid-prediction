# -- coding: utf-8 --
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
parser.add_argument("-c", "--input_time_size", default=5, type=int, help="input_time_size")
parser.add_argument("-d", "--predict_time_size", default=3, type=int, help="predict_time_size")
parser.add_argument("-f", "--city_name", default='武汉', type=str, help="city_name")
args = parser.parse_args()
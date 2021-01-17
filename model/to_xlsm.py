from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--filepath', type=str, help='file path to convert csv to xlsm')

args = parser.parse_args()

onlyfiles = [f for f in listdir(args.filepath) if isfile(join(args.filepath, f))]


for file in onlyfiles:
    temp_file = file.split(".")
    if temp_file[-1] == "csv":
        read_filepath = args.filepath + "/" + file
        raw_data = pd.read_csv(read_filepath)
        raw_data.to_excel(args.filepath + "/" + temp_file[0] + ".xlsm", index=False)





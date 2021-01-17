import pandas as pd
import argparse
import datetime
parser = argparse.ArgumentParser()

parser.add_argument('--filename', type=str, help='file to fix dates on')
parser.add_argument('--days', type=int,help="Days shift needed")
parser.add_argument('--output', type=str, help = "File name for output")
args = parser.parse_args()


def rectify_dates(filename, shift_num, output):
    file = pd.read_csv(filename)
    file['Date'] = pd.to_datetime(file['Date'], format='%Y-%m-%d')
    time_difference = datetime.timedelta(days=shift_num)
    for i in range(len(file["Date"])):
        file["Date"][i] = file["Date"][i] + time_difference
    with open(output, "a+") as f:
        file.to_csv(f, header=True, index=False)

file = rectify_dates(filename=args.filename, shift_num=args.days, output=args.output)


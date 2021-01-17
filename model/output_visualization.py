import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("https://raw.githubusercontent.com/AC2005/covid-learning/main/output/sc_predicts.csv")

def calc_percent_error(data, startdate):
    individual_percent_error = list()
    dates = list()
    predict = list()
    actual = list()
    filtered_data = df.loc[df["Date"] >= startdate]
    filtered_data.reset_index(drop=True, inplace=True)
    for i in range(len(filtered_data)):
        actual_data = int(filtered_data['ActualCases'][i])
        predicts = int(filtered_data["Predicts"][i])
        if predicts == 0 or actual_data == 0:
            continue
        dates.append(filtered_data['Date'][i])
        predict.append(predicts)
        actual.append(actual_data)
        temp_perc_error = np.abs((actual_data - predicts)/(predicts))
        individual_percent_error.append(temp_perc_error)

    std = np.std(individual_percent_error)
    avg = np.average(individual_percent_error)

    max_list = [predict[i] * (1+2*std) for i in range(len(predict))]
    min_list = [predict[i] * (1-2*std) for i in range(len(predict))]
    adjusted_data_set = pd.DataFrame(
        {'Date': dates,
         'ActualCases': actual,
         'Preidcts': predict,
         '+2 STD': max_list,
         '-2 STD': min_list
         })
    return adjusted_data_set

new_data = calc_percent_error(data=df, startdate="2020-10-01")

# plt.plot(max)
# plt.plot(min)
# plt.show()


print(max)
print(min)

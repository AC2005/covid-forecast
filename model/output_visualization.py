import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv")

print(df["date"])

predict_file_window = "/Users/AndrewC 1/git/covid-forecast/output/us_predicts_v4_200.csv"
predict_file = "/Users/AndrewC 1/git/covid-forecast/output/us_predicts_v1_200.csv"

df = pd.read_csv(predict_file_window)
df2 = pd.read_csv(predict_file)

def calc_percent_error(data, startdate):
    individual_percent_error = list()
    dates = list()
    predict = list()
    actual = list()
    filtered_data = data.loc[data["Date"] >= startdate]
    filtered_data.reset_index(drop=True, inplace=True)
    for i in range(len(filtered_data)):
        actual_data = int(filtered_data['ActualCases'][i])
        predicts = int(filtered_data["Predicts"][i])
        if predicts == 0 or actual_data == 0:
            continue
        temp_perc_error = np.abs((actual_data - predicts)/(predicts))
        dates.append(filtered_data['Date'][i])
        predict.append(predicts)
        actual.append(actual_data)
        individual_percent_error.append(temp_perc_error)
    error_Q3 = np.median(np.sort(individual_percent_error)[int(len(individual_percent_error) / 2):])
    error_IQR = np.median(np.sort(individual_percent_error)[int(len(individual_percent_error) / 2):]) - np.median(
    np.sort(individual_percent_error[int(len(individual_percent_error) / 2):]))
    where_list = np.where(individual_percent_error > error_Q3 + 1.5 * error_IQR)
    remove_indices = where_list[0]
    for i in sorted(remove_indices, reverse=True):
        del individual_percent_error[i]
        del dates[i]
        del actual[i]
        del predict[i]
    std = np.std(individual_percent_error)
    avg = np.average(individual_percent_error)


    max_list = [predict[i] * (1+2*std) for i in range(len(predict))]
    min_list = [predict[i] * (1-2*std) for i in range(len(predict))]
    adjusted_data_set = pd.DataFrame(
        {'Date': dates,
         'ActualCases': actual,
         'Preidcts': predict,
         '+2 STD': max_list,
         '-2 STD': min_list,
         'Percent Error': individual_percent_error
         })
    return adjusted_data_set, avg

new_data, average_error = calc_percent_error(data=df, startdate="2020-10-01")
print(new_data)

plt.plot(df["Predicts"][0:32], color="blue")
plt.plot(df["ActualCases"][0:32], color="red")
plt.show()

plt.plot(df2["Predicts"], color="blue")
plt.plot(df2["ActualCases"], color="red")
plt.show()


print(max)
print(min)

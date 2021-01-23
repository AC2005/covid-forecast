import argparse
import logging
from model.sts_model import *
from model.nn_model import *

df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv")
dfUS = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv")
dfCounties = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")
logger = logging.getLogger(__name__)


def get_cases(state=None, county=None):
    if county is not None and state is not None:  # We have a county and a state
        dfCounties.drop(["fips", "deaths"], axis=1, inplace=True)
        cumulated_cases = dfCounties.loc[(dfCounties["county"] == county) & (dfCounties["state"] == state)]
        data_dates = cumulated_cases["date"]
        data_dates.reset_index(drop=True, inplace=True)
        cumulated_cases.drop(['date', 'state', 'county'], axis=1, inplace=True)
        cumulated_cases.reset_index(drop=True, inplace=True)
        temp_delta_cases = list()
        temp_delta_cases.append(cumulated_cases["cases"][0])
        for i in range(1, len(cumulated_cases)):
            temp = cumulated_cases["cases"][i] - cumulated_cases["cases"][i - 1]
            temp_delta_cases.append(temp)
        delta_cases = pd.DataFrame(temp_delta_cases, columns=["cases"])
        return cumulated_cases, delta_cases, data_dates
    if state is not None and county is None:  # State ONLY
        df.drop(["fips", "deaths"], axis=1, inplace=True)
        cumulated_cases = df.loc[df["state"] == state]
        data_dates = cumulated_cases["date"]
        data_dates.reset_index(drop=True, inplace=True)
        cumulated_cases.drop(["state", "date"], axis=1, inplace=True)
        cumulated_cases.reset_index(drop=True, inplace=True)
        temp_delta_cases = list()
        temp_delta_cases.append(cumulated_cases["cases"][0])
        for i in range(1, len(cumulated_cases)):
            temp = cumulated_cases["cases"][i] - cumulated_cases["cases"][i - 1]
            temp_delta_cases.append(temp)
        delta_cases = pd.DataFrame(temp_delta_cases, columns=["cases"])
        return cumulated_cases, delta_cases, data_dates
    if state is None and county is None:  # No county No state -> US dataset
        data_dates = dfUS["date"]
        dfUS.drop(["date", "deaths"], axis=1, inplace=True)
        cumulated_cases = dfUS
        temp_delta_cases = list()
        temp_delta_cases.append(dfUS["cases"][0])
        for i in range(1, len(cumulated_cases)):
            temp = cumulated_cases["cases"][i] - cumulated_cases["cases"][i - 1]
            temp_delta_cases.append(temp)
        delta_cases = pd.DataFrame(temp_delta_cases, columns=["cases"])
        return cumulated_cases, delta_cases, data_dates


def get_saved_cases(file_name=None, dates=None):
    if file_name is None:
        return 0, None
    df = pd.read_csv(file_name, sep=" ")
    last_date = df['Date'].iloc[-1]
    for index in range(len(dates)):
        if last_date == dates.iloc[index]:
            return index, last_date

    return 0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='AI Model')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--steps', type=int, help='Predict steps')
    parser.add_argument('--startdate', type=str, help='Start date')
    parser.add_argument('--days', type=int,help="Days for prediction")
    parser.add_argument('--state', type=str, help="State for prediction")
    parser.add_argument('--county', type=str, help='County for prediction, must have a state argument')
    parser.add_argument('--window_size', type=int, help="Window for prediction", default=0)
    args = parser.parse_args()

    cumulative_cases, daily_cases, dates = get_cases(state=args.state, county=args.county)
    start_index = dates.tolist().index(args.startdate)
    if start_index < 0:
        logger.error("Start date is not valid {}".format(args.startdate))
        return -1
    else:
        print("start_date {}, index: {}".format(dates.tolist()[start_index], start_index))
    last_index = start_index + args.days

    if args.model == 'sts':
        model_sts = modelSTS(200, daily_cases)
        model_sts.build_model_sts()
        trained_loss_curve = model_sts.training_surrogate()
        plt.plot(trained_loss_curve)
        plt.show()
        model_sts.forcast_surrogate(sample=100)
    elif args.model == "hmc":
        for day in range(start_index, last_index, args.steps):
            if args.window_size > 0:
                model_sts = modelSTS(200, daily_cases.loc[day-args.window_size:day])
            else:
                model_sts = modelSTS(200, daily_cases.loc[:day])
            model_sts.build_model_sts()
            model_sts.training_hmc()
            forecast_mean, forecast_scale, forecast_samples = model_sts.forcast_hmc(num_steps_forecast=args.steps)
            model_sts.save_result(args.output,
                                  model_sts.get_actual_dates(dates, day, day+args.steps),
                                  model_sts.get_actual_cases(daily_cases, day, day+args.steps),
                                  CASES_COLUMN,
                                  forecast_mean[:args.steps].numpy())
    elif args.model == 'nn':
        for day in range(start_index, last_index, args.steps):
            model_nn = modelNN()
            input, output = model_nn.data_prep(window_size=7)
            model_nn.build_model(input=input, output=output)
            forecast = model_nn.model_prediction(predict_set=input[-7:-6])
            model_nn.save_result(args.output,
                                 dates.iloc[day:day+args.steps],
                                 daily_cases.iloc[day:day+args.steps],
                                 forecast[:args.steps])


if __name__ == '__main__':
    main()












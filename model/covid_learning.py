import argparse
import logging
import json
import os
import datetime
from model.sts_model import *
# from model.nn_model import *

DATE_PATTERN="%Y-%m-%d"
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
    parser.add_argument('--param_output', type=str, help="File to write parameters")
    parser.add_argument('--steps', type=int, help='Predict steps')
    parser.add_argument('--startdate', type=str, help='Start date')
    parser.add_argument('--days', type=int,help="Days for prediction")
    parser.add_argument('--state', type=str, help="State for prediction")
    parser.add_argument('--county', type=str, help='County for prediction, must have a state argument')
    parser.add_argument('--window_size', type=int, help="Window for prediction", default=0)
    args = parser.parse_args()

    cumulative_cases, daily_cases, dates = get_cases(state=args.state, county=args.county)
    try:
        start_index = dates.tolist().index(args.startdate)
        print("start_date {}, index: {}".format(dates.tolist()[start_index], start_index))
    except:
        start_index=(datetime.datetime.strptime(args.startdate, DATE_PATTERN) \
                    - datetime.datetime.strptime(dates[0], DATE_PATTERN)).days
        logger.error("Start date is in future valid {}".format(args.startdate))

    last_index = start_index + args.days

    if args.model == 'sts':
        model_sts = modelSTS(200, daily_cases)
        model_sts.build_model_sts()
        trained_loss_curve = model_sts.training_surrogate()
        plt.plot(trained_loss_curve)
        plt.show()
        model_sts.forcast_surrogate(sample=100)
    elif args.model == "hmc":
        parameter = {}
        last_date = datetime.datetime.strptime(dates[len(dates)-1], DATE_PATTERN).timetuple()
        if os.path.exists((args.output)):
            current_result=pd.read_csv(args.output)
            last_output_date = datetime.datetime.strptime(current_result['Date'].values[len(current_result)-1], DATE_PATTERN).timetuple()
            if last_output_date.tm_year >= last_date.tm_year and last_output_date.tm_yday > last_date.tm_yday:
                next_output_date = (datetime.datetime.strptime(dates[len(dates)-1], DATE_PATTERN)
                                    + datetime.timedelta(days=1)).strftime(DATE_PATTERN)
                current_index = current_result.index[current_result['Date'] == next_output_date][0]
                updated_daily_cases = pd.DataFrame(np.append(daily_cases, current_result[PREDICTS].loc[current_index:]))
            else:
                updated_daily_cases = daily_cases
        else:
            updated_daily_cases=daily_cases

        forecast_cases = pd.DataFrame(np.zeros(last_index-start_index))
        for day in range(start_index, last_index, args.steps):
            if args.window_size > 0:
                if day > len(updated_daily_cases):
                    cases = pd.DataFrame(np.append(updated_daily_cases.loc[day-args.window_size:],forecast_cases.loc[:day-len(updated_daily_cases)-1]))
                else:
                    cases = updated_daily_cases.loc[day - args.window_size:day-1]
                model_sts = modelSTS(200, cases)
            else:
                model_sts = modelSTS(200, updated_daily_cases.loc[:day])
            model_sts.build_model_sts()
            model_sts.training_hmc()
            forecast_mean, forecast_scale, forecast_samples = model_sts.forcast_hmc(num_steps_forecast=args.steps)

            model_sts.save_result(args.output,
                                  model_sts.get_actual_dates(dates, day, day+args.steps),
                                  model_sts.get_actual_cases(daily_cases, day, day+args.steps),
                                  CASES_COLUMN,
                                  forecast_mean[:args.steps].numpy())
            if day >= len(updated_daily_cases):
                forecast_cases.iloc[day-len(updated_daily_cases)]=forecast_mean[:args.steps].numpy()[0]

            component_means, component_stddevs, forecast_component_means, forecast_component_stddevs = model_sts.get_trend_dist()
            if day < len(dates):
                param_key = dates[day]
            else:
                param_key = (datetime.datetime.strptime(dates[0], "%Y-%m-%d")+datetime.timedelta(days=day)).strftime("%Y-%m-%d")

            parameter[str(param_key)] = {
                "component_means": component_means,
                "component_stddevs": component_stddevs,
                "forecast_component_means": forecast_component_means,
                "forecast_component_stddevs": forecast_component_stddevs,

            }
        with open(args.param_output, "a") as f:
            #open json , if key exists, overwrite, then dump.
            json.dump(parameter, f, indent=2)

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












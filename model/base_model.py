import pandas as pd
import os.path
import numpy as np
import datetime

DATE = "Date"
ACTUALCASE = "ActualCases"
PREDICTS = "Predicts"
DATE_FORMAT = "%Y-%m-%d"
CASES_COLUMN = "cases"
DATE_COLUMN = "date"


class baseModel:
    def __init__(self):
        pass

    def save_result(self,
                    file_name: str,
                    dates: pd.DataFrame,
                    actual_data: pd.DataFrame,
                    col_name: str,
                    forecast_data: pd.DataFrame):
        """
        Save prediction along with actual result
        :param file_name: file to save result to
        :param dates: dataframe containing dates
        :param actual_data: actual covid data
        :param col_name: column containing actual cases
        :param forecast_data: prediction data
        :return:
        """
        df = pd.DataFrame({
            DATE: dates.tolist(),
            ACTUALCASE: actual_data[col_name].values.tolist(),
            PREDICTS:list(map(int, forecast_data.tolist()))},
            columns=[DATE, ACTUALCASE, PREDICTS])

        write_header = False if os.path.exists(file_name) else True
        with open(file_name, "a+") as f:
            df.to_csv(f, header=write_header, index=False)

    def get_actual_dates(self, dates: pd.DataFrame, start_day: int, end_day: int) -> np.array:
        """
        Retrieve dates column
        :param dates: dataframe containing dates
        :param start_day: start day for prediction
        :param end_day: end day of prediction
        :return: dates array
        """

        start_date = datetime.datetime.strptime(dates.iloc[0], DATE_FORMAT)

        dates_array = np.array([(start_date+datetime.timedelta(days=i)).strftime(DATE_FORMAT)
                                for i in range(start_day, end_day)])

        return dates_array

    def get_actual_cases(self, cases: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
        """
        :param cases: dataframe containing all actual cases
        :param start: start index of actual cases
        :param end: end index of actual cases
        :return: dataframe containing actual cases
        """
        if start >= len(cases):
            return pd.DataFrame(np.nan, index=[n for n in range(end-start)], columns=['cases'])

        if end < len(cases):
            return cases.iloc[start:end]

        return pd.DataFrame(cases[start:].astype(np.int).values.tolist()
                            + [np.NaN for i in range(end-len(cases))], columns=[CASES_COLUMN])

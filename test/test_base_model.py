from model.base_model import *
import datetime


def test_get_actual_dates_in_range():
    """
    Test get_actual_dates
    Test cases: when start and end are within the dates dataframe
    :return:
    """
    start_day = 10
    end_day = 20
    start_date = datetime.datetime.strptime("2020-10-10", DATE_FORMAT)
    base_dates = pd.DataFrame(np.array([(start_date+datetime.timedelta(days=i)).strftime(DATE_FORMAT) for i in range(start_day, end_day)]),
                              columns=[DATE_COLUMN])[DATE_COLUMN]

    base_model = baseModel()
    dates = base_model.get_actual_dates(base_dates, 1, 2)
    expected_dates = np.array([(datetime.datetime.strptime("2020-10-20", DATE_FORMAT)
                                + datetime.timedelta(days=1)).strftime(DATE_FORMAT)])

    assert np.array_equal(dates, expected_dates) is True


def test_get_actual_dates_out_range():
    """
    Test get_actual_dates
    Test cases: when start and end are out of the dates dataframe
    :return:
    """
    start_day = 0
    end_day = 10
    start_date = datetime.datetime.strptime("2020-10-10", DATE_FORMAT)
    base_dates = pd.DataFrame(np.array([(start_date + datetime.timedelta(days=i)).strftime(DATE_FORMAT)
                                        for i in range(start_day, end_day)]), columns=[DATE_COLUMN])[DATE_COLUMN]

    base_model = baseModel()
    dates = base_model.get_actual_dates(base_dates, 25, 27)
    expected_dates = np.array([(datetime.datetime.strptime("2020-10-10", DATE_FORMAT)
                                + datetime.timedelta(days=25)).strftime(DATE_FORMAT),
                               (datetime.datetime.strptime("2020-10-10", DATE_FORMAT)
                                + datetime.timedelta(days=26)).strftime(DATE_FORMAT)])

    assert np.array_equal(dates, expected_dates) is True


def test_get_actual_cases_in_range():
    """
    Test get_actual_cases
    When start and end are in the range of cases
    """
    test_cases_no = 10
    base_cases = pd.DataFrame(np.array([i for i in range(test_cases_no)]), columns=[CASES_COLUMN])
    base_model = baseModel()
    cases = base_model.get_actual_cases(base_cases, 5, 8).reset_index(drop=True)

    expected_cases = pd.DataFrame(np.array([5,6,7]), columns=[CASES_COLUMN])
    assert cases.equals(expected_cases) is True


def test_get_actual_cases_out_range():
    """
    Test get_actual_cases
    When start and end are out of the range of cases
    """
    test_cases_no = 10
    base_cases = pd.DataFrame(np.array([i for i in range(test_cases_no)]), columns=[CASES_COLUMN])
    base_model = baseModel()
    cases = base_model.get_actual_cases(base_cases, 12, 15).reset_index(drop=True)

    expected_cases = pd.DataFrame(np.array([np.NaN, np.NaN, np.NaN]), columns=[CASES_COLUMN])
    assert cases.equals(expected_cases) is True


def test_get_actual_cases_mixed_range():
    """
    Test get_actual_cases
    When start is in the range of cases but end is not
    """
    test_cases_no = 10
    base_cases = pd.DataFrame(np.array([i for i in range(test_cases_no)]), columns=[CASES_COLUMN])
    base_model = baseModel()
    cases = base_model.get_actual_cases(base_cases[CASES_COLUMN], 9, 12).reset_index(drop=True)

    expected_cases = pd.DataFrame(np.array([9, np.NaN, np.NaN]), columns=[CASES_COLUMN])
    assert cases.equals(expected_cases) is True

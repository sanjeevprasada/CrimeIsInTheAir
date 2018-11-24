import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import multiprocessing
import os
from tqdm import tqdm

''' References: 
    (1) https://stackoverflow.com/questions/24588437/convert-date-to-float-for-linear-regression-on-pandas-data-frame/24590666,
    (2) https://matplotlib.org/gallery/text_labels_and_annotations/date.html
'''


class Pollution:
    def __init__(self, county, feature, num_days_predict):
        self.county = county
        self.feature = feature
        self.num_days_predict = num_days_predict
        self.city_data = defaultdict()
        self.best_gamma = 0.001
        self.best_alpha = 10
        self.cities = []
        self.counties = []
        self.county_cities = defaultdict(lambda: defaultdict(list))   # [state][county] = list of all cities in county
        self.county_data = defaultdict()  # [id] = dataframe for given county
        self.county_data_all = defaultdict()
        self.county_info = defaultdict(list)  # [id] = [county, state]
        self.city_directory = 'city_data/'
        self.county_directory = 'county_data/'
        self.feature_names = ['NO2_AQI', 'O3_AQI', 'SO2_AQI', 'CO_AQI', 'id', 'State Code', 'State']

    def process_data(self, id_info=False, monthy_data=False):
        dataset = pd.read_csv('pollution_clean_8.csv', index_col='Date Local', parse_dates=True)

        features = list(dataset.columns.values)

        self.cities = dataset['City'].unique()
        city_info = defaultdict(lambda: defaultdict(str))

        # Create a dataframe and information dictionary for each city
        for c in tqdm(self.cities):
            self.city_data[c] = dataset[dataset['City'] == c]
            self.city_data[c].sort_index()

            city_row = self.city_data[c].iloc[0]
            for f in features:
                city_info[c][f] = city_row[f]

            # Depending on the application, process the identification information
            if id_info:
                self.city_data[c] = self.city_data[c][['NO2_AQI', 'O3_AQI', 'SO2_AQI', 'CO_AQI', 'id', 'State Code', 'State']]
            else:
                self.city_data[c] = self.city_data[c][['NO2_AQI', 'O3_AQI', 'SO2_AQI', 'CO_AQI']]

            self.city_data[c] = self.city_data[c].groupby(self.city_data[c].index).max()
            self.city_data[c].fillna(method='ffill', inplace=True)
            self.city_data[c].fillna(method='bfill', inplace=True)

            # merge daily data into monthly
            if monthy_data:
                self.city_data[c] = self.city_data[c].resample('MS').mean()

        # Find the cities belonging to each county
        for c in self.cities:
            state = city_info[c]['State']
            county = city_info[c]['County']
            self.county_cities[state][county].append(c)

        # Merge cities into single county dataframe
        for state, counties in self.county_cities.items():
            for county, cities in counties.items():
                county_id = city_info[cities[0]]['id']
                self.counties.append(county_id)

                self.county_data[county_id] = self.city_data[cities[0]]
                self.county_info[county_id] = [county, state]

    def find_best_parameters(self):
        data = self.county_data[self.county].values
        feature = data[:, self.feature]

        dates_pd = self.county_data[self.county].index

        days = ((dates_pd - dates_pd.min()) / np.timedelta64(1, 'D')).values
        days = np.reshape(days.astype(int), newshape=(days.shape[0], 1))
        days = days % 365

        train_index = int(0.8 * days.shape[0])

        x_train = days[0:train_index]
        y_train = feature[0:train_index]

        # Find the best hyperparameter using grid search and cross validation
        parameters = [{'kernel': ['rbf'], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1], 'alpha': [0.1, 1, 10, 50, 100]}]
        model = KernelRidge()
        clf = GridSearchCV(model, parameters, cv=3, scoring='neg_mean_squared_error', n_jobs=multiprocessing.cpu_count() - 1)
        clf.fit(x_train, y_train)

        best_alpha = clf.best_estimator_.alpha
        best_gamma = clf.best_estimator_.gamma

        print('Best MSE: {}'.format(-1 * clf.best_score_))
        print("Best gamma: {}, alpha: {}".format(best_gamma, best_alpha))

    def predict(self):
        data = self.county_data[self.county].values
        feature = data[:, self.feature]

        dates_pd = self.county_data[self.county].index
        days = ((dates_pd - dates_pd.min()) / np.timedelta64(1, 'D')).values
        days = np.reshape(days.astype(int), newshape=(days.shape[0], 1))
        days = days % 365

        train_index = int(0.8 * days.shape[0])

        x_train = days[0:train_index]
        y_train = feature[0:train_index]

        x_test = days[train_index:]
        y_test = feature[train_index:]

        model = KernelRidge(alpha=self.best_alpha, kernel='rbf', gamma=self.best_gamma)
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        #print("MSE on train set: {}".format(mse_train))

        y_test_pred = model.predict(x_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        #print("MSE on test set: {}".format(mse_test))

        # Train on all data, use that to predict 'x' days into the future
        model = KernelRidge(alpha=self.best_alpha, kernel='rbf', gamma=self.best_gamma)
        model.fit(days, feature)

        last_day = days[-1] + 1
        x_unseen = np.linspace(last_day, last_day + self.num_days_predict, self.num_days_predict)
        x_unseen = np.reshape(x_unseen.astype(int), newshape=(x_unseen.shape[0], 1))
        x_unseen = x_unseen % 365
        y_unseen_pred = model.predict(x_unseen)

        return y_train_pred, y_test_pred, y_unseen_pred, mse_train, mse_test

    def plot_data(self, y_train_pred, y_test_pred, y_unseen_pred):
        data = self.county_data[self.county].values
        feature = data[:, self.feature]

        dates_pd = self.county_data[self.county].index
        dates = dates_pd.values

        days = ((dates_pd - dates_pd.min()) / np.timedelta64(1, 'D')).values
        days = np.reshape(days.astype(int), newshape=(days.shape[0], 1))

        last_date = dates_pd.max()
        last_date = last_date + pd.DateOffset(days=1)
        extended_date = last_date + pd.DateOffset(days=self.num_days_predict - 1)
        x_extended = pd.date_range(start=last_date, end=extended_date)

        train_index = int(0.8 * days.shape[0])

        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)

        # Plot ground truth
        ax.plot(dates, feature, linewidth=1)

        # Plot prediction
        ax.plot(dates[0:train_index], y_train_pred, linewidth=1.5, color='yellow')
        ax.plot(dates[train_index:], y_test_pred, linewidth=1.5, color='orange')
        ax.plot(x_extended, y_unseen_pred, linewidth=1.5, color='red')

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        yearsFmt = mdates.DateFormatter('%Y')

        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)

        # Round to nearest year
        datemin = np.datetime64(dates[0], 'Y')
        datemax = np.datetime64(dates[-1], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)

        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

        fig.autofmt_xdate()
        plt.savefig('test.png', dpi=500)

    def save_cities(self):
        if not os.path.exists(self.city_directory):
            os.mkdir(self.city_directory)

        for c in self.cities:
            file_path = '{}{}.csv'.format(self.city_directory, c.replace(' ', '_'))
            if not os.path.exists(file_path):
                self.city_data[c].to_csv(file_path)

    def save_counties(self):
        if not os.path.exists(self.county_directory):
            os.mkdir(self.county_directory)

            for c in tqdm(self.counties):
                file_path = '{}{}.csv'.format(self.county_directory, c)
                if not os.path.exists(file_path) and c in self.county_data.keys() and self.county_data[c] is not None:
                    self.county_data[c].to_csv(file_path)

    def load_city(self, city):
        file_path = '{}{}.csv'.format(self.city_directory, city.replace(' ', '_'))
        self.city_data[city] = pd.read_csv(file_path, index_col='Date Local', parse_dates=True)

    def load_county(self, county):
        file_path = '{}{}.csv'.format(self.county_directory, county)
        self.county_data[county] = pd.read_csv(file_path, index_col='Date Local', parse_dates=True)

    def save_forecast(self, y_train_pred, y_test_pred, y_unseen_pred, monthly_data=False):
        dates_pd = self.county_data[self.county].index

        last_date = dates_pd.max()
        last_date = last_date + pd.DateOffset(days=1)
        extended_date = last_date + pd.DateOffset(days=self.num_days_predict - 1)
        x_extended = pd.date_range(start=last_date, end=extended_date)

        y_pred = np.concatenate((y_train_pred, y_test_pred, y_unseen_pred), axis=0)
        all_dates = dates_pd.append(x_extended).values

        directory = 'forecast_data_daily/'

        if not os.path.exists(directory):
            os.mkdir(directory)

        file_path = '{}{}_{}.csv'.format(directory, self.county, self.feature_names[self.feature].replace(' ', '_'))
        with open(file_path, 'w') as file:
            for index, row in enumerate(all_dates):
                if index == all_dates.shape[0] - 1:
                    file.write("{}, {}".format(row, y_pred[index]))
                else:
                    file.write("{}, {}\n".format(row, y_pred[index]))

    def save_forecast_accuracy(self, accuracy_info):
        with open('accuracy_{}.csv'.format(self.feature_names[self.feature]), 'w') as file:
            file.write("Location, MSE Train, MSE Test\n")
            for c in self.counties:
                [county, state] = self.county_info[c]
                [mse_train, mse_test] = accuracy_info[c]
                file.write("{}-{}, {}, {}\n".format(county, state, mse_train, mse_test))

    def create_all_forecasts(self, monthly_data=False):
        features = [0, 1, 2, 3]
        accuracy_info = defaultdict(list)
        for c in tqdm(self.counties):
            for f in features:
                self.feature = f
                self.county = c

                y_train_pred, y_test_pred, y_unseen_pred, mse_train, mse_test = poll.predict()
                accuracy_info[c] = [mse_train, mse_test]
                self.save_forecast(y_train_pred, y_test_pred, y_unseen_pred, monthly_data)

    def create_accuracy_files(self, num_counties=-1):
        self.process_data()
        self.counties = self.counties[0:num_counties]

        features = [0, 1, 2, 3]
        accuracy_info = defaultdict(list)
        for f in tqdm(features):
            for c in self.counties:
                self.feature = f
                self.county = c

                y_train_pred, y_test_pred, y_unseen_pred, mse_train, mse_test = poll.predict()
                accuracy_info[c] = [mse_train, mse_test]
                self.save_forecast(y_train_pred, y_test_pred, y_unseen_pred)

            self.save_forecast_accuracy(accuracy_info)

    def get_county_info(self):
        county_info = pd.read_csv('lat_long.csv')

        # get lat/long position
        county_info.Latitude = county_info.Latitude.str[0:-1]
        county_info.Longitude = county_info.Longitude.str[0:-1]
        county_info.to_csv('lat_long_filtered.csv', columns=['FIPS', 'Latitude', 'Longitude'], index=False)

        state_counties = defaultdict(list)
        county_state = defaultdict()
        all_counties = set()
        state_info = defaultdict(list)

        # get all counties in each state
        for index, row in county_info.iterrows():
            state = row['State']
            county_code = row['FIPS']
            state_counties[state].append(county_code)
            county_state[county_code] = state
            all_counties.add(county_code)
            state_info[state].append(county_code)

        # create data for counties that don't have any by averaging county data for that state
        all_counties = list(all_counties)
        counties_with_data = self.county_data.keys()

        print("Averaging data for unknown counties")
        for county_code in tqdm(all_counties):
            state = county_state[county_code]
            if county_code not in counties_with_data:
                # get all counties in this state
                counties_in_state = state_info[state]

                # get the counties that we have data for
                counties_with_data_for_state = list(set(counties_in_state) & set(counties_with_data))

                # average the county data
                combined = None
                if len(counties_with_data_for_state) == 1:
                    combined = self.county_data[counties_with_data_for_state[0]]
                if len(counties_with_data_for_state) == 2:
                    combined = pd.concat([self.county_data[counties_with_data_for_state[0]], self.county_data[counties_with_data_for_state[1]]], axis=1)
                    combined = combined.groupby(combined.columns.values, axis=1).mean()
                if len(counties_with_data_for_state) == 3:
                    combined = pd.concat([self.county_data[counties_with_data_for_state[0]], self.county_data[counties_with_data_for_state[1]]], axis=1)
                    combined = pd.concat([combined, self.county_data[counties_with_data_for_state[2]]], axis=1)
                    combined = combined.groupby(combined.columns.values, axis=1).mean()
                if len(counties_with_data_for_state) >= 4:
                    for current_index, county_with_data_for_state in enumerate(counties_with_data_for_state):
                        if current_index > 0:
                            combined = pd.concat([combined, self.county_data[counties_with_data_for_state[current_index]]], axis=1)
                        else:
                            combined = self.county_data[counties_with_data_for_state[0]]

                    combined = combined.groupby(combined.columns.values, axis=1).mean()

                self.county_data_all[county_code] = combined

            else:
                self.county_data_all[county_code] = self.county_data[county_code]

        # create county_daily_predictions data
        print("Predicting daily data")
        self.create_all_forecasts()

        # create county_daily data
        print("Saving all county daily data")
        self.county_data = self.county_data_all
        self.counties = all_counties
        self.county_directory = 'county_daily_data/'
        self.save_counties()

        # create county_monthly data
        print("Saving all county monthly data")
        for c in all_counties:
            if c in self.county_data.keys() and self.county_data[c] is not None:
                self.county_data[c] = self.county_data[c].resample('MS').mean()
        self.county_directory = 'county_monthly_data/'
        self.save_counties()


if __name__ == '__main__':
    id_info = False
    monthly_data = True
    county = 4013
    feature = 1
    num_days_predict = 365

    poll = Pollution(county, feature, num_days_predict)
    poll.process_data()
    poll.get_county_info()

    #poll.process_data(monthy_data=monthly_data)
    #poll.save_counties()

    #poll.create_accuracy_files(num_counties=5)
    #poll.create_all_forecasts()

    # poll.load_county(county)
    # y_train_pred, y_test_pred, y_unseen_pred = poll.predict()
    #
    # poll.plot_data(y_train_pred, y_test_pred, y_unseen_pred)
    #poll.save_forecast(y_train_pred, y_test_pred, y_unseen_pred)

    #
    #poll.save_cities()
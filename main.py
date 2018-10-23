import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import multiprocessing
import os

''' References: (1) https://stackoverflow.com/questions/24588437/convert-date-to-float-for-linear-regression-on-pandas-data-frame/24590666,
    (2) https://matplotlib.org/gallery/text_labels_and_annotations/date.html
'''


class Pollution:
    def __init__(self, city, feature, num_days_predict):
        self.city = city
        self.feature = feature
        self.num_days_predict = num_days_predict
        self.data = defaultdict()
        self.best_gamma = 0.001
        self.best_alpha = 10
        self.cities = []
        self.city_directory = 'city_data/'

    def process_data(self):
        dataset = pd.read_csv('pollution_clean.csv', index_col='Date Local', parse_dates=True)

        features = list(dataset.columns.values)

        self.cities = dataset['City'].unique()
        #self.cities = list(self.cities)[0:5]
        city_info = defaultdict(lambda: defaultdict(str))

        # Create a dataframe and information dictionary for each city
        for c in self.cities:
            self.data[c] = dataset[dataset['City'] == c]
            self.data[c].sort_index()

            city_row = self.data[c].iloc[0]

            for f in features:
                city_info[c][f] = city_row[f]

                self.data[c] = self.data[c][['NO2 AQI', 'O3 AQI', 'SO2 AQI', 'CO AQI']]
                self.data[c] = self.data[c].groupby(self.data[c].index).max()
                self.data[c].fillna(method='ffill', inplace=True)
                self.data[c].fillna(method='bfill', inplace=True)

    def find_best_parameters(self):
        city_data = self.data[self.city].values
        city_feature = city_data[:, self.feature]

        dates_pd = self.data[self.city].index
        dates = dates_pd.values

        days = ((dates_pd - dates_pd.min()) / np.timedelta64(1, 'D')).values
        days = np.reshape(days.astype(int), newshape=(days.shape[0], 1))
        days = days % 365

        train_index = int(0.8 * days.shape[0])

        x_train = days[0:train_index]
        y_train = city_feature[0:train_index]

        x_test = days[train_index:]
        y_test = city_feature[train_index:]

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
        city_data = self.data[self.city].values
        city_feature = city_data[:, self.feature]

        dates_pd = self.data[self.city].index
        days = ((dates_pd - dates_pd.min()) / np.timedelta64(1, 'D')).values
        days = np.reshape(days.astype(int), newshape=(days.shape[0], 1))
        days = days % 365

        train_index = int(0.8 * days.shape[0])

        x_train = days[0:train_index]
        y_train = city_feature[0:train_index]

        x_test = days[train_index:]
        y_test = city_feature[train_index:]

        model = KernelRidge(alpha=self.best_alpha, kernel='rbf', gamma=self.best_gamma)
        model.fit(x_train, y_train)

        y_train_pred = model.predict(x_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        print("MSE on train set: {}".format(mse_train))

        y_test_pred = model.predict(x_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        print("MSE on test set: {}".format(mse_test))

        # Train on all data, use that to predict 'x' days into the future
        model = KernelRidge(alpha=self.best_alpha, kernel='rbf', gamma=self.best_gamma)
        model.fit(days, city_feature)

        last_day = days[-1]
        x_unseen = np.linspace(last_day, last_day + self.num_days_predict, self.num_days_predict + 1)
        x_unseen = np.reshape(x_unseen.astype(int), newshape=(x_unseen.shape[0], 1))
        y_unseen_pred = model.predict(x_unseen)

        return y_train_pred, y_test_pred, y_unseen_pred

    def plot_data(self, train_prediction, test_prediction, future_prediction):
        city_data = self.data[self.city].values
        city_feature = city_data[:, self.feature]

        dates_pd = self.data[self.city].index
        dates = dates_pd.values

        days = ((dates_pd - dates_pd.min()) / np.timedelta64(1, 'D')).values
        days = np.reshape(days.astype(int), newshape=(days.shape[0], 1))

        train_index = int(0.8 * days.shape[0])

        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(111)

        # Plot ground truth
        ax.plot(dates, city_feature, linewidth=1)

        # Plot prediction
        ax.plot(dates[0:train_index], train_prediction, linewidth=1.5, color='r')
        ax.plot(dates[train_index:], test_prediction, linewidth=1.5, color='orange')
        #ax.plot(dates[train_index:], future_prediction, linewidth=1.5, color='y')

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
                self.data[c].to_csv(file_path)

    def load_city(self, city):
        file_path = '{}{}.csv'.format(self.city_directory, city.replace(' ', '_'))
        self.data[city] = pd.read_csv(file_path, index_col='Date Local', parse_dates=True)


if __name__ == '__main__':
    city = 'Phoenix'
    feature = 1
    num_days_predict = 365

    poll = Pollution(city, feature, num_days_predict)
    #poll.process_data()
    poll.load_city(city)

    poll.save_cities()

    y_train_pred, y_test_pred, y_unseen_pred = poll.predict()

    #poll.plot_data(y_train_pred, y_test_pred, y_unseen_pred)
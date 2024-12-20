import pandas as pd

from build_dataset import create_dataset, create_dataset_actors
from eda import visualization, visualization_actors
from logistic_regression import logistic_regression, logistic_regression_2
from random_forest import random_forest, random_forest_bagging, random_forest_2, random_forest_bagging_2
from svm import svm, svm_2
from xgboost_algorithm import xgboost, xgboost_2
from ann import ann, ann_2

if __name__ == '__main__':

    data_frame1 = pd.read_csv('datasets/oscardata_bestpicture.csv')
    data_frame2 = pd.read_csv('datasets/oscardata_2021_bestpicture.csv')
    data_frame3 = pd.read_csv('datasets/movie_data.csv')

    data_frame4 = pd.read_csv('datasets/oscardata_acting.csv')
    data_frame5 = pd.read_csv('datasets/oscardata_2021_acting.csv')

    merged_df = create_dataset(data_frame1, data_frame2, data_frame3)
    #visualization(merged_df)

    #merged_df = create_dataset_actors(data_frame4, data_frame5)
    #visualization_actors(merged_df)

    logistic_regression(merged_df)
    #svm(merged_df)
    #random_forest(merged_df)
    #random_forest_bagging(merged_df)
    #xgboost(merged_df)
    #ann(merged_df)

    #logistic_regression_2(merged_df)
    #svm_2(merged_df)
    #random_forest_2(merged_df)
    #random_forest_bagging_2(merged_df)
    #xgboost_2(merged_df)
    #ann_2(merged_df)

import numpy as np
import pandas as pd
from IPython.display import display


def file_download(path=None, sort=None):
    return pd.read_csv(path).sort_values(sort)


def to_datetime(df=None, col=None):
    df3 = pd.to_datetime(df[col], format="%a %b %d %H:%M:%S %z %Y", errors='coerce')
    ts2 = pd.Series(df3)
    return ts2


def dataframe_creation(df=None, with_reading_time_NaN=None, with_reading_time=None):
    if with_reading_time_NaN:
        ts = to_datetime(df=df, col='started_at')
        df = df.assign(started_at=ts)
        ts2 = to_datetime(df=df, col='read_at')
        df = df.assign(read_at=ts2)
        df['reading_time'] = (df['read_at'] - df['started_at']).dt.days
        if with_reading_time:
            index_with_nan = df.index[df.isnull().any(axis=1)]
            df.drop(index_with_nan, 0, inplace=True)

    df[['0', '1', '2', '3', 'localisation', '5']] = df.date_updated.str.split(" ", expand=True, )
    df = df.drop(
        ['0', '1', '2', '3', '5', 'rating', 'date_added', "book_id", 'read_at', 'started_at', 'date_updated', 'user_id',
         'date_added', 'date_updated', 'n_votes', 'n_comments'], axis=1)

    return df

def rating(df = None):
    return df[["review_id", "rating"]]

if __name__ == '__main__':
    df = file_download(path='data/initial/goodreads_train.csv', sort='book_id')

    df2 = dataframe_creation(df=df, with_reading_time_NaN=1, with_reading_time=0)
    pd.set_option('display.max_columns', None)
    print(df2.head())
    df2.to_csv("data/initial/goodreads_train_with_reading_time_NaN.csv",index=False)

    df3 = dataframe_creation(df=df, with_reading_time_NaN=1, with_reading_time=1)
    pd.set_option('display.max_columns', None)
    print(df3.head())
    df3.to_csv("data/initial/goodreads_train_with_reading_time.csv",index=False)

    df4 = dataframe_creation(df=df, with_reading_time_NaN=0, with_reading_time=0)
    pd.set_option('display.max_columns', None)
    print(df4.head())
    df4.to_csv("data/initial/goodreads_train_not_reading_time.csv",index=False)

    df5 = rating(df=df)
    df5.to_csv("data/initial/rating.csv",index=False)

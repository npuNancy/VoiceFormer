import os
import random
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt


class MergeProcessing:
    def __init__(self, df_list):
        self.df_list = df_list
        self.merged_df = self.merge(self.df_list)
        self.downsampled_df = self.adapt_sampling(self.merged_df, down_sample_rate=4000)
        self.interpolated_df = self.interpolation(self.downsampled_df)
        self.filted_df = self.high_pass_filter(self.interpolated_df, cutoff_fre=50, fs=8000)

    def get_merged_data(self):
        return self.filted_df

    def merge(self, df_list):
        df = pd.concat(df_list, axis=0, ignore_index=True)
        df = df.sort_values(by="time", ascending=True)
        df = df.reset_index(drop=True)

        df_list = []
        tmp_list = []

        for index, row in df.iterrows():
            if index == 0:

                tmp_list.append(self.get_row_dict(row))
            else:
                if row["time"] == df.iloc[index-1]["time"]:

                    tmp_list.append(self.get_row_dict(row))
                else:

                    df_list.append({
                        "time": tmp_list[0]["time"],
                        "data": np.mean([row["data"] for row in tmp_list]),
                        "sensor": "merged" if len(tmp_list) > 1 else tmp_list[0]["sensor"]
                    })
                    tmp_list = [self.get_row_dict(row)]

        df_list.append({
            "time": tmp_list[0]["time"],
            "data": np.mean([row["data"] for row in tmp_list]),
            "sensor": "merged" if len(tmp_list) > 1 else tmp_list[0]["sensor"]
        })

        df = pd.DataFrame(df_list)
        return df

    def adapt_sampling(self, df, down_sample_rate=4000):
        sample_rate_now = self.cal_sensor_sample_rate(df)
        duration = self.get_sensor_duration(df)
        sample_num_now = len(df)
        sample_num_down = int(duration * down_sample_rate)
        sample_num_diff = sample_num_now - sample_num_down
        row_index_list = self.select_n_numbers(df, sample_num_diff)
        if sample_rate_now < down_sample_rate:
            df_adapted_sampling = df
            row_dict = {"time": np.nan, "data": np.nan, "sensor": "adapted"}
            for row_index in row_index_list:
                df_adapted_sampling = self.insert_row(df_adapted_sampling, row_dict, row_index)
            return df_adapted_sampling

        else:
            """降采样"""
            df_down_sampling = df.drop(row_index_list, axis=0)
            df_down_sampling = df_down_sampling.reset_index(drop=True)
            return df_down_sampling

    def interpolation(self, df):
        row = {"time": np.nan, "data": np.nan, "sensor": "interpolation"}
        df_tmp = pd.DataFrame([row for i in range(len(df))])

        interpolated_df = pd.concat([df, df_tmp], axis=0, ignore_index=False)
        interpolated_df = interpolated_df.sort_index(axis=0, ascending=True, na_position='last')
        interpolated_df = interpolated_df.reset_index(drop=True)
        interpolated_df = interpolated_df.interpolate(method='linear', axis=0)

        return interpolated_df

    def high_pass_filter(self, df, cutoff_fre=50, fs=8000):
        wn = 2 * cutoff_fre / fs
        b, a = butter(4, wn, 'highpass')
        df["data"] = filtfilt(b, a, df["data"])
        return df

    def get_row_dict(self, row):
        row_dict = {
            "time": row["time"],
            "data": row["z"] if row["sensor"] == "acc" else (row["y"] if row["sensor"] == "gyr" else row["z"]),
            "sensor": row["sensor"]
        }
        return row_dict

    def cal_sensor_sample_rate(self, df):
        start, end = df["time"][0], df["time"][len(df)-1]
        duration = (end - start) / 1000
        sample_rate = len(df) / duration
        return sample_rate

    def get_sensor_duration(self, df):
        start, end = int(df["time"][0]), int(df["time"][len(df)-1])
        duration = (end - start) / 1000
        return duration

    def select_n_numbers(self, df, n):
        n = abs(n)
        step = len(df) // n
        if step <= 1:
            return []
        start = random.randint(0, step-1)
        index_list = [i for i in range(len(df))]
        result = [index_list[start+i*step] for i in range(n)]
        return result

    def insert_row(self, df, row_dict, row_index):
        row_index = min(max(row_index, 0), len(df)-1)
        df_part_1 = df.iloc[:row_index, :]
        df_part_2 = df.iloc[row_index:, :]
        row_df = pd.DataFrame([row_dict])
        df_res = pd.concat([df_part_1, row_df, df_part_2], axis=0, ignore_index=True)
        df_res = df_res.reset_index(drop=True)
        df_res = df_res.interpolate(method='linear', axis=0)
        return df_res

    def save_data(self, df, filepath):
        df.to_csv(filepath, index=False, header=True)


if __name__ == "__main__":
    pass

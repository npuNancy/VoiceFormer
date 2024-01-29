import os
import sys
import random
import pathlib
import numpy as np
import pandas as pd

from pydub import AudioSegment
from scipy.signal import butter, filtfilt

from meta import SensorType


class Sensorprocessing:
    def __init__(
            self, sensor_data_path, wave_data_folder,
            sensor_type, time_offset=9999, max_min_dict=None, compare_times=10):
        self.time_offset = time_offset / 1000

        self.sensor_type = sensor_type
        self.sensor_data_path = sensor_data_path
        self.wave_data_folder = wave_data_folder
        self.max_min_dict = max_min_dict
        self.compare_times = compare_times

        self.wave_data_path = self.get_wave_data_path()

        self.sensor_data = self.read_data()
        self.fs_sensor = self.cal_sensor_sample_rate()

        self.sensor_data = self.high_pass_filter(self.sensor_data, cutoff_fre=50, fs=self.fs_sensor)

        self.sensor_data = self.standardization()
        self.original_signal = self.__get_original_signal()
        self.sensor_data = self.time_alignment()
        self.time_alignment_signal = self.__get_time_alignment_signal()
        self.noise_reduction_signal = self.__get_noise_reduction_signal()

        self.sensor_data = self.cal_relative_timestamp_offset()
        self.sensor_data = self.add_sensor_type()
        self.sensor_data = self.change_column_name()

    def get_processed_data(self):
        return self.sensor_data

    def get_wave_data_path(self):
        _, sensor_file_name = os.path.split(self.sensor_data_path)
        wave_file_name = sensor_file_name.split("_")[0] + ".wav"
        if wave_file_name not in os.listdir(self.wave_data_folder):
            exit()
        wave_data_path = os.path.join(self.wave_data_folder, wave_file_name)
        return wave_data_path

    def read_data(self):
        return pd.read_csv(self.sensor_data_path, header=None)

    def time_alignment(self):
        wave = AudioSegment.from_wav(self.wave_data_path)
        wave_duration = wave.duration_seconds
        sensor_data = self.sensor_data
        sensor_duration = self.get_sensor_duration(sensor_data)

        if sensor_duration < wave_duration + self.time_offset:
            exit()

        sensor_data = sensor_data.iloc[round(self.time_offset * self.fs_sensor):, :]
        sensor_data = sensor_data.reset_index(drop=True)

        sensor_data["duration"] = sensor_data[0].apply(lambda x: float(x.split("#")[1]) - float(sensor_data[0][0].split("#")[1]))

        for i in range(0, len(sensor_data)-1):
            if sensor_data["duration"][i] >= wave_duration * 1000:
                sensor_data = sensor_data.iloc[:i+1, :]
                break

        sensor_data = sensor_data.drop(["duration"], axis=1)
        sensor_data = sensor_data.reset_index(drop=True)
        return sensor_data

    def standardization(self):
        df = self.sensor_data
        df[1] = (df[1] - df[1].mean()) / df[1].std()
        df[2] = (df[2] - df[2].mean()) / df[2].std()
        df[3] = (df[3] - df[3].mean()) / df[3].std()
        return df

    def normalization(self):
        df = self.sensor_data
        df[1] = (df[1] - self.max_min_dict["x_min"]) / (self.max_min_dict["x_max"] - self.max_min_dict["x_min"]) * 2 - 1
        df[2] = (df[2] - self.max_min_dict["y_min"]) / (self.max_min_dict["y_max"] - self.max_min_dict["y_min"]) * 2 - 1
        df[3] = (df[3] - self.max_min_dict["z_min"]) / (self.max_min_dict["z_max"] - self.max_min_dict["z_min"]) * 2 - 1
        return df

    def cal_relative_timestamp_offset(self):
        df = self.sensor_data
        if self.sensor_type == SensorType.acc:
            df[0] = df[0].apply(lambda x: float(x.split("#")[1]) - float(df[0][0].split("#")[1])
                                + round(random.uniform(0, 1), 2))
        elif self.sensor_type == SensorType.gyr:
            df[0] = df[0].apply(lambda x: float(x.split("#")[1]) - float(df[0][0].split("#")[1])
                                + round(random.uniform(0, 1), 2))
        return df

    def add_sensor_type(self):
        df = self.sensor_data
        df['sensor'] = self.sensor_type.name
        return df

    def cal_sensor_sample_rate(self):
        df = self.sensor_data
        duration = self.get_sensor_duration(df)
        sample_rate = len(df) / duration
        return sample_rate

    def get_sensor_duration(self, df):
        start, end = float(df[0][0].split("#")[1]), float(df[0][len(df)-1].split("#")[1])
        duration = (end - start) / 1000
        return duration

    def change_column_name(self):
        df = self.sensor_data
        df.columns = ["time", "x", "y", "z", "sensor"]
        return df

    def high_pass_filter(self, df, cutoff_fre=50, fs=500):
        b, a = butter(4, 2 * cutoff_fre / fs, 'highpass')
        df[1] = filtfilt(b, a, df[1])
        df[2] = filtfilt(b, a, df[2])
        df[3] = filtfilt(b, a, df[3])
        return df

    def __get_original_signal(self):
        df = self.sensor_data
        sensor_data = df[3].values if self.sensor_type == SensorType.acc else df[2].values
        return sensor_data

    def __get_time_alignment_signal(self):
        df = self.sensor_data
        sensor_data = df[3].values if self.sensor_type == SensorType.acc else df[2].values
        return sensor_data

    def __get_noise_reduction_signal(self):
        df = self.sensor_data
        sensor_data = df[3].values if self.sensor_type == SensorType.acc else df[2].values
        noise_threshold = 1
        sensor_data = [np.random.rand(1)[0]*0.1-0.05 if abs(x) < noise_threshold else x for x in sensor_data]
        return sensor_data


if __name__ == "__main__":
    pass

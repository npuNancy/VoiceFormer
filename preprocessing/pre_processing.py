import os
import sys
import pathlib
import itertools
import numpy as np
import pandas as pd
from merge_processing import MergeProcessing
from sensor_processing import Sensorprocessing


from meta import SensorType


class PreProcessing:
    def __init__(
            self, wave_data_folder, sensor_base_path, merged_data_save_folder,
            n, m, sensor_type_list, time_offset_acc, time_offset_gyr):

        self.wave_data_folder = wave_data_folder
        self.sensor_base_path = sensor_base_path
        self.merged_data_save_folder = merged_data_save_folder

        self.n, self.m = n, m
        self.sensor_type_list = sensor_type_list
        self.time_offset_acc = time_offset_acc
        self.time_offset_gyr = time_offset_gyr
        self.base_path_acc = os.path.join(sensor_base_path, "accelerometer")
        self.base_path_gyr = os.path.join(sensor_base_path, "gyroscope")

        self.mkdir(self.merged_data_save_folder)
        self.processed_sensor_dict = {}
        self.filename_list = os.listdir(self.base_path_acc)
        self.filename_dict = self.get_filename_dict(self.filename_list)
        max_min_dict_acc = {'x_max': 0.397648, 'x_min': 0.091238, 'y_max': 0.255844, 'y_min': 0.017363, 'z_max': 10.242473, 'z_min': 5.767105}
        max_min_dict_gyr = {'x_max': 0.012217, 'x_min': -0.042953, 'y_max': 0.008814, 'y_min': -0.026442, 'z_max': 0.01316, 'z_min': -0.003264}

        self.combined_filepath_dict = self.combine(self.filename_dict, self.sensor_type_list, self.n, self.m)
        for file_label, file_list in self.combined_filepath_dict.items():
            sensor_df_list = []
            for filepath, sensor_type, time_offset in file_list:
                if filepath in self.processed_sensor_dict:
                    sensor_df = self.processed_sensor_dict[filepath]
                else:
                    max_min_dict = max_min_dict_acc if sensor_type == SensorType.acc else max_min_dict_gyr
                    sensor_processing = Sensorprocessing(filepath, self.wave_data_folder, sensor_type, time_offset, max_min_dict)
                    sensor_df = sensor_processing.get_processed_data()
                    self.processed_sensor_dict[filepath] = sensor_df

                sensor_df_list.append(sensor_df)

            merged_data_save_path = os.path.join(self.merged_data_save_folder, f"{file_label}.csv")
            preprocess_merge = MergeProcessing(sensor_df_list)
            merged_data = preprocess_merge.get_merged_data()
            merged_data.to_csv(merged_data_save_path, index=False, header=True)

    def get_filename_dict(self, file_list):
        file_dict = {}
        for file in file_list:
            file_label = file.split("_")[0]
            if file_label not in file_dict:
                file_dict[file_label] = []
            file_dict[file_label].append(file)
        return file_dict

    def combine(self, file_dict, sensor_type_list, n, m):
        result_file_dict = {}
        for file_label, file_list in file_dict.items():
            if len(file_list) < n:
                continue

            combinations_list = itertools.combinations(file_list[:n], m)
            for idx, combinations_item in enumerate(combinations_list):

                result_file_label = f"{file_label}_{idx}"
                result_file_dict[result_file_label] = []
                for filename in combinations_item:
                    if SensorType.acc in sensor_type_list:
                        tmp = (os.path.join(self.base_path_acc, filename), SensorType.acc, self.time_offset_acc)
                        result_file_dict[result_file_label].append(tmp)
                    if SensorType.gyr in sensor_type_list:
                        tmp = (os.path.join(self.base_path_gyr, filename), SensorType.gyr, self.time_offset_gyr)
                        result_file_dict[result_file_label].append(tmp)

        result_file_dict = dict(sorted(result_file_dict.items(), key=lambda x: x[0]))
        return result_file_dict

    def get_sensor_max_min(self, base_path):
        max_min_dict = {"x_max": -999, "x_min": 999, "y_max": -999, "y_min": 999, "z_max": -999, "z_min": 999}
        filename_list = os.listdir(base_path)
        filepath_list = [os.path.join(base_path, filename) for filename in filename_list]
        for file in filepath_list:
            sensor_data = pd.read_csv(file, header=None)

            x_max, x_min = sensor_data[1].max(), sensor_data[1].min()
            y_max, y_min = sensor_data[2].max(), sensor_data[2].min()
            z_max, z_min = sensor_data[3].max(), sensor_data[3].min()

            max_min_dict["x_max"] = max(max_min_dict["x_max"], x_max)
            max_min_dict["x_min"] = min(max_min_dict["x_min"], x_min)
            max_min_dict["y_max"] = max(max_min_dict["y_max"], y_max)
            max_min_dict["y_min"] = min(max_min_dict["y_min"], y_min)
            max_min_dict["z_max"] = max(max_min_dict["z_max"], z_max)
            max_min_dict["z_min"] = min(max_min_dict["z_min"], z_min)
        return max_min_dict

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass


def work():
    wave_data_folder = "path/to/wave-data-folder"
    sensor_base_path = "path/to/sensor-base-path"
    merged_data_save_folder = "path/to/merged-data-save-folder"
    n, m = 8, 4
    sensor_type_list = [SensorType.acc, SensorType.gyr]

    time_offset_acc = 100
    time_offset_gyr = 100

    process = PreProcessing(wave_data_folder, sensor_base_path, merged_data_save_folder, n, m, sensor_type_list, time_offset_acc, time_offset_gyr)


if __name__ == "__main__":
    work()

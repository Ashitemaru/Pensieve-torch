import os

from config import config

def load_trace():
    cooked_trace_dir = config["cooked_trace_dir"]
    cooked_file_name_list = os.listdir(cooked_trace_dir)

    cooked_time_list = []
    cooked_bandwidth_list = []

    for cooked_file_name in cooked_file_name_list:
        cooked_time = []
        cooked_bandwidth = []
        with open(cooked_trace_dir + "/" + cooked_file_name, "rb") as handle:
            for line in handle:
                data_tuple = line.split()
                cooked_time.append(float(data_tuple[0]))
                cooked_bandwidth.append(float(data_tuple[1]))

        cooked_time_list.append(cooked_time)
        cooked_bandwidth_list.append(cooked_bandwidth)

    return cooked_time_list, cooked_bandwidth_list, cooked_file_name_list

if __name__ == "__main__":
    pass
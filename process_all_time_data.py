import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import scipy.signal as signal
import math
from pathlib import Path

# --- Paths and Constants ---
path = Path(r"D:\DLITE\Data\LWA_antenna_independantly\data_2_FEE")
path_save_fig = Path(r"D:\DLITE\Sun_Observering_Interferometry\Result_Image")

# Create directories if they don't exist
(path_save_fig / "avg_power_all").mkdir(parents=True, exist_ok=True)
(path_save_fig / "histogram_all").mkdir(parents=True, exist_ok=True)
(path_save_fig / "master_heatmap_all").mkdir(parents=True, exist_ok=True)

# --- Utility Functions ---
def get_all_folder_file_list(data_path, folder_name):
    _file_list = sorted(list((data_path / folder_name).glob(f"*.csv")))
    if not _file_list:
        return None, None, None, None, 0
    
    _frequency = np.loadtxt(_file_list[0], delimiter=',', usecols=0)
    _timeStamp_list = []
    _timeString_list = []
    
    for file in _file_list:
        file_name = file.name
        parts = file_name.split('_')
        if len(parts) >= 6:
            try:
                date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
                time_str = f"{parts[3]}:{parts[4]}:{parts[5]}"
                dateTime = datetime.datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
                _timeStamp_list.append(dateTime.timestamp())
                _timeString_list.append(time_str)
            except (ValueError, IndexError):
                print(f"Error parsing datetime for file: {file_name}")
                continue
    return _file_list, _frequency, np.array(_timeStamp_list), _timeString_list, len(_file_list)

def calculate_power(data_path, folder_name, file_list, number_file, frequency):
    if number_file == 0:
        return None, None
    data_matrix = np.zeros((len(frequency), number_file))
    power_list = np.zeros(number_file)
    for i, file in enumerate(file_list):
        try:
            data = np.loadtxt(file, delimiter=',')
            data_matrix[:, i] = data[:, 1]
            power_list[i] = np.mean(data[:, 1])
        except (IOError, IndexError, ValueError) as e:
            print(f"Error loading data from {file.name}: {e}")
            data_matrix[:, i] = np.nan
            power_list[i] = np.nan
    if np.all(np.isnan(power_list)):
        print(f"Folder {folder_name} has no valid power data.")
        return None, None
    return power_list, data_matrix

def remove_spike(x, threshold=3, window_size=11):
    if len(x) < window_size:
        return x
    return np.convolve(x, np.ones(window_size)/window_size, mode='same')

# --- Plotting Functions ---

def plot_multi_panel_graphs(valid_data, save_path):
    n = len(valid_data)
    if n == 0:
        return
        
    cols = 3
    rows = math.ceil(n / cols)
    
    fig, ax_graph = plt.subplots(rows, cols, figsize=[30, 4 * rows], layout='constrained', squeeze=False)
    ax_graph = ax_graph.flatten()
    
    for idx, (folder, time_list, power_list) in enumerate(valid_data):
        ax = ax_graph[idx]
        ax.set_ylabel('Signal (dB)')
        ax.grid(True)
        ax.plot(time_list, power_list, color='green', linestyle='-')
        ax.set_title(f'{folder}', fontsize=13)
        
        x = np.arange(len(time_list))
        step = max(1, len(time_list) // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(time_list[::step], rotation=45, ha='right')
        ax.set_xlabel('Time (hh:mm:ss)')

    for j in range(n, len(ax_graph)):
        fig.delaxes(ax_graph[j])
        
    fig.suptitle('DLITE radio - Average Power over Time (All Data)', fontsize=20)
    plt.show()
    fig.savefig(save_path / "avg_power_combined_all_data.png", dpi=600, bbox_inches='tight')

def plot_multi_panel_histograms(valid_data, save_path):
    n = len(valid_data)
    if n == 0:
        return
    
    cols = 3
    rows = math.ceil(n / cols)
    
    fig, ax_graph = plt.subplots(rows, cols, figsize=[30, 4 * rows], layout='constrained', squeeze=False)
    ax_graph = ax_graph.flatten()

    for idx, (folder, time_list, freq_list, matrix) in enumerate(valid_data):
        ax = ax_graph[idx]
        im = ax.imshow(matrix, cmap='inferno', interpolation='nearest', aspect='auto', origin='lower')
        ax.set_title(f'{folder}', fontsize=13)
        ax.set_xlabel('Time (hh:mm:ss)')
        ax.set_ylabel('Frequency (MHz)')

        step_x = max(1, len(time_list) // 20)
        x = np.arange(len(time_list))
        ax.set_xticks(x[::step_x])
        ax.set_xticklabels(time_list[::step_x], rotation=45, ha='right')
        
        step_y = max(1, len(freq_list) // 8)
        y = np.arange(len(freq_list))
        ax.set_yticks(y[::step_y])
        ax.set_yticklabels([f"{val:.2f}" for val in freq_list[::step_y]])
        
        fig.colorbar(im, ax=ax, label="Signal (dB)")

    for j in range(n, len(ax_graph)):
        fig.delaxes(ax_graph[j])
        
    fig.suptitle('DLITE radio - Spectrograms of Signal Power (All Data)', fontsize=20)
    plt.show()
    fig.savefig(save_path / "histograms_combined_all_data.png", dpi=600, bbox_inches='tight')

def plot_master_heatmap(valid_folders, mean_power_matrix, timeStamp_template, save_path):
    fig, ax = plt.subplots(1, 1, figsize=[24, 10], layout='constrained')
    ax.set_title('Interferometry 610MHz at Hoa Lac - Ha Noi (All Data)', fontsize=20)
    ax.set_xlabel('Date (YYYY_MM_DD)', fontsize=16)
    ax.set_ylabel('Time (hh:mm:ss) UTC+07', fontsize=16)
    
    im = ax.imshow(mean_power_matrix, cmap='inferno', interpolation='nearest', aspect='auto', origin='lower')
    
    x = np.arange(len(valid_folders))
    step_x = max(1, len(valid_folders) // 25)
    ax.set_xticks(x[::step_x])
    ax.set_xticklabels(valid_folders[::step_x], rotation=45, ha='right')
    
    if timeStamp_template is not None and len(timeStamp_template) > 1:
        step_seconds = 600
        template_times = [datetime.datetime.fromtimestamp(ts) for ts in timeStamp_template]
        start_time = template_times[0]
        end_time = template_times[-1]
        y_ticks = []
        y_tick_labels = []
        current_time = start_time
        while current_time <= end_time:
            time_diffs = np.abs(np.array(timeStamp_template) - current_time.timestamp())
            closest_idx = np.argmin(time_diffs)
            y_ticks.append(closest_idx)
            y_tick_labels.append(current_time.strftime('%H:%M:%S'))
            current_time += datetime.timedelta(seconds=step_seconds)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
    
    plt.colorbar(im, ax=ax).set_label("Power (dB)", fontsize=14)
    plt.show()
    fig.savefig(save_path / "master_heatmap_all_data.png", dpi=600, bbox_inches='tight')

# --- Main Execution Block ---

def main():
    data_folder_list = sorted([d.name for d in path.iterdir() if d.is_dir()])
    print(f"Number of folders: {len(data_folder_list)}")

    valid_data_graphs = []
    valid_data_histograms = []
    valid_folders_for_heatmap = []
    valid_power_lists_for_heatmap = []
    timeStamp_template = None

    for folder in data_folder_list:
        file_list, frequency, timeStamp_list, timeString_list, number_file = get_all_folder_file_list(path, folder)
        if number_file == 0 or len(timeStamp_list) == 0:
            print(f"Skipping folder {folder}: no valid files found.")
            continue
            
        power_list, data_matrix = calculate_power(path, folder, file_list, number_file, frequency)
        if power_list is None:
            continue
        valid_data_graphs.append((folder, timeString_list, power_list))
        smoothed_matrix = np.zeros_like(data_matrix)
        for i in range(data_matrix.shape[0]):
            smoothed_matrix[i, :] = remove_spike(data_matrix[i, :])
        valid_data_histograms.append((folder, timeString_list, frequency, smoothed_matrix))
        smoothed_power_list = remove_spike(power_list)
        valid_folders_for_heatmap.append(folder)
        valid_power_lists_for_heatmap.append(smoothed_power_list)
        if timeStamp_template is None:
            timeStamp_template = timeStamp_list

    print("Generating multi-panel power vs. time graphs for ALL data...")
    plot_multi_panel_graphs(valid_data_graphs, path_save_fig / "avg_power_all")
    
    print("Generating multi-panel spectrograms for ALL data...")
    plot_multi_panel_histograms(valid_data_histograms, path_save_fig / "histogram_all")
    
    print("Generating master heatmap for ALL data...")
    if valid_power_lists_for_heatmap:
        max_len = max(len(p) for p in valid_power_lists_for_heatmap)
        mean_power_matrix = np.zeros((max_len, len(valid_power_lists_for_heatmap)))
        for i, power_list in enumerate(valid_power_lists_for_heatmap):
            if len(power_list) != max_len:
                x_old = np.linspace(0, 1, len(power_list))
                x_new = np.linspace(0, 1, max_len)
                power_list_resized = np.interp(x_new, x_old, power_list)
                mean_power_matrix[:, i] = power_list_resized
            else:
                mean_power_matrix[:, i] = power_list
        plot_master_heatmap(valid_folders_for_heatmap, mean_power_matrix, timeStamp_template, path_save_fig / "master_heatmap_all")

if __name__ == "__main__":
    main()
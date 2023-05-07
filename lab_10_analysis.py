"""AER E 322 Lab 10 Analysis Script

This script generates graphs and performs necessary analysis on the two
datasets collected in Lab 10.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = ["Lab 10 Data/test1.txt", "Lab 10 Data/test2.txt"]
OUTPUT_PATH = "Lab 10 Data/graphs/"

def import_data(path):
    """Imports data from the text files.
    
    Args:
        path (str): Path to the text file.
    
    Returns:
        df_list (list): List of pandas dataframes containing the data.
            each dataframe contains the data for a single run. Each
            dataframe has the following columns:
                t (float): Time in seconds.
                F_1 (float): Force in Newtons.
                F_2 (float): Force in Newtons.
                F_3 (float): Force in Newtons.
                F_4 (float): Force in Newtons.
    """
    df_list = []
    for i in range(5):
        df_list.append(pd.read_csv(path, sep="\t", skiprows=1, header=0,
                                   usecols=range(i * 5, i * 5 + 5),
                                   names=["t", "F_1", "F_2", "F_3", "F_4"]))
    return df_list

def graph_data(data_frame, test_num, run_num, output_path, title=None):
    """Graphs the data from a single run.
    
    Args:
        data_frame (pandas.DataFrame): Dataframe containing time and
            force data.
        test_num (int): Test number.
        run_num (int): Run number.
        output_path (str): Path to the output directory.
        title (str): Middle title of the graph.
    """
    # Configure the title string
    if title is None:
        title = f"Force vs. Time - Test {test_num} Run {run_num}"
    else:
        title = f"Force vs. Time - {title} - Test {test_num} Run {run_num}"

    # Configure the output path string
    mod_title = title.lower()\
        .replace('.', '')\
        .replace(' - ', '-')\
        .replace(' ', '-')
    output_path = f"{output_path}{mod_title}.svg"

    # Select the relevant columns
    data_frame = data_frame[["t", "F_1", "F_2", "F_3", "F_4"]]

    # Plot the data
    axes = data_frame.plot(x="t", y=["F_1", "F_2", "F_3", "F_4"],
                           figsize=(8,6))

    # Add labels and legend
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Force (N)")
    axes.set_title(title)
    axes.legend(["Force 1", "Force 2", "Force 3", "Force 4"],
                loc="lower right")
    axes.grid(True)

    # Save the plot to file
    plt.savefig(output_path, format="svg")

    # Close the plot
    plt.close()

def remove_baseline(data_frame):
    """Removes the baseline from a single run.
    
    Args:
        data_frame (pandas.DataFrame): Dataframe containing time and
            force data.
    """
    # Calculate the baseline
    data_frame_no_baseline = data_frame.copy()

    # Remove the baseline
    # We ignore the first ten data points because they were usually erroneous
    data_frame_no_baseline.loc[:, "F_1"] -= data_frame.loc[10:, "F_1"].mean()
    data_frame_no_baseline.loc[:, "F_2"] -= data_frame.loc[10:, "F_2"].mean()
    data_frame_no_baseline.loc[:, "F_3"] -= data_frame.loc[10:, "F_3"].mean()
    data_frame_no_baseline.loc[:, "F_4"] -= data_frame.loc[10:, "F_4"].mean()

    return data_frame_no_baseline

def smooth_data(data_frame, window_size):
    """Applies a simple moving average filter to the F_1, F_2, F_3, and
        F_4 columns of the given DataFrame.

    Args:
        data_frame (pandas.DataFrame): Dataframe containing time and
            force data.
        window_size (int): Size of the window to use for the SMA filter.

    Returns:
        data_frame_smoothed (pandas.DataFrame): Dataframe containing the
            smoothed time and force data.
    """
    data_frame_smoothed = data_frame.copy()

    for col in ["F_1", "F_2", "F_3", "F_4"]:
        data_frame_smoothed[col] = data_frame[col].rolling(window_size).mean()

    return data_frame_smoothed

def calculate_frequency(data_frame):
    """Calculates the frequency of oscillation in each of the F_1, F_2,
        F_3, and F_4 datasets for a given pandas dataframe.

    Args:
        data_frame (pandas.DataFrame): Dataframe containing time and
            force data.

    Returns:
        fft_df (pandas.DataFrame): Dataframe containing the frequency
            and amplitude of oscillation in each of the F_1, F_2, F_3,
            and F_4 datasets.
    """
    # Select the relevant columns
    data_frame = data_frame[["t", "F_1", "F_2", "F_3", "F_4"]]

    # Compute the sampling frequency
    delta_t = data_frame["t"][1] - data_frame["t"][0]
    sampling_freq = 1/delta_t

    # Apply FFT to each column
    fft_df = pd.DataFrame()
    for col in data_frame.columns[1:]:
        force = data_frame[col]
        num_points = len(force)
        fft_result = np.fft.fft(force)
        two_sided_psd = np.abs(fft_result / num_points)
        one_sided_psd = two_sided_psd[:int(num_points / 2) + 1]
        one_sided_psd[1:-1] = 2 * one_sided_psd[1:-1]
        fft_freq = \
            sampling_freq * np.arange(0, int(num_points / 2) + 1) / num_points
        main_freq = fft_freq[np.argmax(one_sided_psd)]
        amplitude = np.max(one_sided_psd) * 2
        fft_df[col] = [main_freq, amplitude]

    fft_df.index = ['frequency', 'amplitude']
    return fft_df

def main():
    """Main function for Lab 10 analysis script."""
    for test_num, data_path in enumerate(DATA_PATH):
        # Import the data
        df_list = import_data(data_path)

        # Loop over each of the runs in this test
        for run_num, data_frame in enumerate(df_list):
            # Remove the baseline
            data_frame_no_baseline = remove_baseline(data_frame)

            # Graph the raw data
            graph_data(data_frame, test_num + 1, run_num + 1, OUTPUT_PATH)

            # Graph the data with the baseline removed
            graph_data(data_frame_no_baseline, test_num + 1, run_num + 1,
                       OUTPUT_PATH, "Baseline Removed")

            # Graph the data with the baseline removed from 5 s to 10 s
            graph_data(data_frame_no_baseline.loc[100:201],
                       test_num + 1, run_num + 1, OUTPUT_PATH,
                       "Baseline Removed - 5 s to 10 s")

            # Graph the data with the baseline removed from 5 s to 6 s
            graph_data(data_frame_no_baseline.loc[100:121],
                       test_num + 1, run_num + 1, OUTPUT_PATH,
                       "Baseline Removed - 5 s to 6 s")

            # Graph the smoothed data with the baseline removed
            graph_data(smooth_data(data_frame_no_baseline, 3), test_num + 1,
                       run_num + 1, OUTPUT_PATH, "Smoothed - Baseline Removed")

            # Graph the smoothed data with the baseline removed from 5 s
            # to 10 s
            graph_data(smooth_data(data_frame_no_baseline.loc[100:201], 3),
                       test_num + 1, run_num + 1, OUTPUT_PATH,
                       "Smoothed - Baseline Removed - 5 s to 10 s")

            # Graph the smoothed data with the baseline removed from 5 s
            # to 6 s
            graph_data(smooth_data(data_frame_no_baseline.loc[100:121], 3),
                       test_num + 1, run_num + 1, OUTPUT_PATH,
                       "Smoothed - Baseline Removed - 5 s to 6 s")

            # Calculate the frequency of oscillation
            fft_df = calculate_frequency(data_frame_no_baseline)
            print(f"Test {test_num + 1}, Run {run_num + 1}:\n"
                  f"F_1 Frequency: {fft_df['F_1'][0]:.2f} Hz | "
                  f"F_1 Amplitude: {fft_df['F_1'][1]:.2f} N\n"
                  f"F_2 Frequency: {fft_df['F_2'][0]:.2f} Hz | "
                  f"F_2 Amplitude: {fft_df['F_2'][1]:.2f} N\n"
                  f"F_3 Frequency: {fft_df['F_3'][0]:.2f} Hz | "
                  f"F_3 Amplitude: {fft_df['F_3'][1]:.2f} N\n"
                  f"F_4 Frequency: {fft_df['F_4'][0]:.2f} Hz | "
                  f"F_4 Amplitude: {fft_df['F_4'][1]:.2f} N\n\n")

if __name__ == "__main__":
    main()

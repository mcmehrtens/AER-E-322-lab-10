"""AER E 322 Lab 10 Analysis Script

This script generates graphs and performs necessary analysis on the two
datasets collected in Lab 10.
"""
import pandas as pd
import matplotlib.pyplot as plt

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

def graph_data(data_frame, test_num, run_num, output_path):
    """Graphs the data from a single run.
    
    Args:
        data_frame (pandas.DataFrame): Dataframe containing time and
            force data.
        test_num (int): Test number.
        run_num (int): Run number.
        output_path (str): Path to the output directory.
    """
    # Select the relevant columns
    data_frame = data_frame[['t', 'F_1', 'F_2', 'F_3', 'F_4']]
    
    # Plot the data
    axes = data_frame.plot(x='t', y=['F_1', 'F_2', 'F_3', 'F_4'],
                           figsize=(8,6))
    
    # Add labels and legend
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Force (N)')
    axes.set_title(f'Force vs. Time - Test {test_num} Run {run_num}')
    axes.legend(['Force 1', 'Force 2', 'Force 3', 'Force 4'])
    axes.grid(True)
    
    # Save the plot to file
    plt.savefig(output_path +
                f"force-vs-time-test-{test_num}-run-{run_num}.svg",
                format='svg')

def main():
    """Main function for Lab 10 analysis script."""
    for test_num, data_path in enumerate(DATA_PATH):
        df_list = import_data(data_path)
        for run_num, data_frame in enumerate(df_list):
            graph_data(data_frame, test_num + 1, run_num + 1, OUTPUT_PATH)

if __name__ == "__main__":
    main()

# Operating system and file management
import os
import shutil
import subprocess
import contextlib
import traceback
import gc
import glob, copy

# Jupyter notebook widgets and display
import ipywidgets as widgets
from IPython.display import display

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Plotting and visualization
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler

# Machine learning and preprocessing
from sklearn.model_selection import train_test_split
import pickle
from ta import trend, momentum, volatility, volume

# Mathematical and scientific computing
import math
from scipy.ndimage import gaussian_filter1d

# Type hinting
from typing import Callable, Tuple

# Deep learning with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set working directory
Working_directory = os.path.normpath("path/to/your/project")
os.chdir(Working_directory)

# CUDA device auto-selection
def auto_select_cuda_device(verbose=True):
    """Automatically select the least-used CUDA device, or fallback to CPU."""
    if not torch.cuda.is_available():
        if verbose:
            print("⚠️ No CUDA device available. Using CPU.")
        return torch.device("cpu")

    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_used = [int(x) for x in smi_output.strip().split('\n')]
        best_gpu = int(np.argmin(memory_used))
        if verbose:
            print(f"🎯 Auto-selected GPU: {best_gpu} ({memory_used[best_gpu]} MiB used)")
        return torch.device(f"cuda:{best_gpu}")
    except Exception as e:
        if verbose:
            print(f"⚠️ GPU detection failed. Falling back to cuda:0 ({e})")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Device assignment
device = auto_select_cuda_device()

# Utility functions
def ensure_folder(folder_path: str) -> None:
    """Ensure the given folder exists, create it if not."""
    os.makedirs(folder_path, exist_ok=True)

def plot_with_matplotlib(data: pd.DataFrame, 
                         title: str, 
                         interactive: bool = False, 
                         save_path: str = None, 
                         show_plot: bool = True, 
                         save_matplotlib_object: str = None) -> None:
    """
    Plot time-series data using Matplotlib with optional trend-based coloring.

    Args:
        - data (pd.DataFrame): Data containing a 'close' column (required).
        - title (str): Plot title.
        - interactive (bool): Enable zoom & pan if True.
        - save_path (str, optional): Path to save the figure.
        - show_plot (bool): Whether to display the plot.
        - save_matplotlib_object (str, optional): Path to save the Matplotlib object.

    Returns:
        - None: Displays or saves the plot as specified.
    """
    # Check if 'close' column exists
    if 'close' not in data.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    # Set default color from Matplotlib cycle
    default_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    
    # Define colors for different trends
    trend_colors = {
        0: 'black',
        1: 'yellow',
        2: 'red',
        3: 'green',
        4: default_blue
    }

    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot with trend-based coloring if 'trend' column exists
    if 'trend' in data.columns:
        legend_added = set()
        prev_idx = data.index[0]
        for idx, row in data.iterrows():
            if idx != prev_idx:
                trend_key = int(row['trend'])
                label = f'Trend {trend_key}' if trend_key not in legend_added else None
                ax.plot([prev_idx, idx], 
                        [data.loc[prev_idx, 'close'], row['close']],
                        color=trend_colors[trend_key], 
                        linestyle='-', 
                        linewidth=1,
                        label=label)
                legend_added.add(trend_key)
            prev_idx = idx
        ax.set_title(f"{title} (Connected, Colored by Trend)")
    else:
        # Plot default line if no 'trend' column
        ax.plot(data.index, data['close'], label='Closing Price', linestyle='-', marker='o', 
                markersize=2, linewidth=1, color=default_blue, markerfacecolor='green', markeredgecolor='black')
        ax.set_title(title)
    
    # Set axis labels and add legend/grid
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price (USD)')
    ax.legend()
    ax.grid()
    
    # Enable interactive features if requested
    if interactive:
        zoom_factory(ax)
        panhandler(fig)

    # Save the plot if a path is provided
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    # Save the Matplotlib object if requested
    if save_matplotlib_object:
        with open(save_matplotlib_object, 'wb') as f:
            pickle.dump(fig, f)

    # Display the plot if requested
    if show_plot:
        plt.show()

def load_and_show_pickle(pickle_file_path: str):
    """
    Load a pickled Matplotlib figure object and display it.

    Args:
        - pickle_file_path (str): Path to the pickled Matplotlib figure file.

    Returns:
        - None: Displays the loaded figure.
    """
    # Load and display the pickled figure
    try:
        with open(pickle_file_path, "rb") as f:
            loaded_fig = pickle.load(f)

        print(f"Figure successfully loaded and displayed from: {pickle_file_path}")
        plt.show(block=True)

    except FileNotFoundError:
        print(f"Error: File not found at {pickle_file_path}.")
    except Exception as e:
        print(f"Error loading the pickled figure: {e}")

def save_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save DataFrame to CSV.
    """
    df.to_csv(file_path)
    print(f"\nSuccessfully saved data with moving average to CSV: \n\t{file_path}\n")

def read_csv_file(file_path: str, preview_rows: int = 5, 
                  days_towards_end: int = None, 
                  days_from_start: int = None, description: str = ""):
    """
    Reads a CSV file and returns a pandas DataFrame filtered by date range.

    Args:
        - file_path (str): The path to the CSV file.
        - preview_rows (int): Number of rows to preview (default is 5).
        - days_towards_end (int, optional): Number of days from the most recent date.
        - days_from_start (int, optional): Number of days from the oldest date of filtered data.
        - description (str): A brief description of the dataset.
                           Explanation:
                           - To retrieve data from the **end**: Use `days_towards_end`.
                           - To retrieve data from the **start of the filtered range**: Use `days_from_start`.
                           - To retrieve data from the **middle**: Use both:
                             For example, if `days_towards_end=100` and `days_from_start=50`,
                             the function will first filter the last 100 days of the dataset,
                             and then filter the first 50 days from this range.
                             This results in data between the last 100th and the last 50th day.

    Returns:
        - pd.DataFrame: The loaded and filtered data from the CSV file.
    """
    try:
        if description:
            print(f"\nDescription: {description}")
        print(f"\nFile path: {file_path}")
        
        # Read the CSV file
        data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        
        # Filter by days towards the end
        if days_towards_end is not None:
            # Get the most recent date in the dataset
            last_date = data.index.max()
            end_cutoff_date = last_date - pd.Timedelta(days=days_towards_end)
            data = data[data.index >= end_cutoff_date]
            print(f"\nRetrieving data from the past {days_towards_end} days (from {end_cutoff_date.date()} onwards):")
        
        # Filter by days from the start (from the filtered data)
        if days_from_start is not None:
            # Get the earliest date in the filtered dataset
            first_date = data.index.min()
            start_cutoff_date = first_date + pd.Timedelta(days=days_from_start)
            data = data[data.index <= start_cutoff_date]
            print(f"\nRetrieving the first {days_from_start} days from the filtered data (up to {start_cutoff_date.date()}):")

        if preview_rows:
            # Print a preview of the data
            print(f"\nPreview of the first {preview_rows} rows:")
            display(data.head(preview_rows))
            print()

            print(f"\nPreview of the last {preview_rows} rows:")
            display(data.tail(preview_rows))
            print()

        return data
    
    except FileNotFoundError:
        print("Error: File not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: File parsing failed.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def downsample_minute_data(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Downsample minute data into N-minute intervals by retaining every Nth row.

    Args:
        - data (pd.DataFrame): The original DataFrame with a datetime index.
        - n (int): The number of minutes for the downsampling interval.

    Returns:
        - pd.DataFrame: Downsampled DataFrame.
    """
    print("\n========---> Downsampling the data! \n")
    data = data.copy()

    # Ensure index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("DataFrame index conversion to DatetimeIndex failed.") from e

    # Downsample by selecting rows where minute % N == 0
    return data[data.index.minute % n == 0]

def calculate_log_returns_all_columns(data: pd.DataFrame, exclude_columns: list = [], dropna: bool = True) -> pd.DataFrame:
    """
    Calculate log returns for all numeric columns in a pandas DataFrame,
    excluding specified columns, and removing excluded columns from the returned DataFrame.

    Args:
        - data (pd.DataFrame): Input DataFrame containing numeric data.
        - exclude_columns (list): List of columns to exclude from log return calculations and the result.
        - dropna (bool): Whether to drop rows with NaN values resulting from the calculation.

    Returns:
        - pd.DataFrame: DataFrame with log returns for numeric columns, excluding specified columns.
    """
    # Copy data and remove excluded columns
    data = data.copy().drop(columns=exclude_columns)
    
    # Select numeric columns for transformation
    columns_to_transform = data.select_dtypes(include=[np.number]).columns
    print(f"columns_to_transform = \n{columns_to_transform}, \nlen(columns_to_transform) = {len(columns_to_transform)}")

    # Calculate log returns for each numeric column
    for col in columns_to_transform:
        if (data[col] <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values. Log returns require strictly positive values.")
        data[col] = np.log(data[col] / data[col].shift(1))

    # Return data with or without NaN rows based on dropna
    return data.dropna() if dropna else data

def created_sequences_2(data: pd.DataFrame, sequence_length: int = 60, sliding_interval: int = 60) -> list:
    """
    Divide the dataset into sequences based on the sequence_length.
    Each sequence must fully cover the window size.

    Args:
    - data (pd.DataFrame): The input DataFrame.
    - sequence_length (int): The window size for sequences.

    Returns:
    - sequences (list): A list of sequences (as DataFrames).
    """
    sequences = []
    
    # Iterate over the data with a sliding window to create sequences
    for i in range(0, len(data) - sequence_length + 1, sliding_interval):
        # Extract a sequence of specified length from the DataFrame
        seq = data.iloc[i:i + sequence_length].copy()
        sequences.append(seq)

    return sequences

def gaussian_smoothing(data: pd.DataFrame, sigma=2) -> pd.DataFrame:
    """
    Applies Gaussian smoothing to numeric columns in a DataFrame.

    Args:
        - data (pd.DataFrame): Input DataFrame.
        - sigma (float): Standard deviation for the Gaussian kernel (default is 2).

    Returns:
        - pd.DataFrame: Smoothed DataFrame with sorted index.
    """
    # Sort data by index in ascending order and create a copy
    data = data.sort_index(ascending=True).copy()
    
    # Apply Gaussian smoothing to numeric columns
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = gaussian_filter1d(data[column].values, sigma=sigma)
    
    return data

def detect_trends_4(
    dataframe: pd.DataFrame, 
    column: str = 'close', 
    lower_threshold: float = 0.001, 
    upper_threshold: float = 0.02,
    reverse_steps: int = 7,
    trends_to_keep: set = {0, 1, 2, 3, 4}  # Default keeps all trends
) -> pd.DataFrame:
    """
    Detects trends based on log return data provided in a specified column and categorizes them into different strength levels.

    This function analyzes time-series data by evaluating cumulative trends in log return values provided in the input DataFrame. 
    It uses three dictionaries (`dic1`, `dic2`, `dic3`) to track different phases of trends, handles multi-step reversals, and 
    classifies trends dynamically based on cumulative product thresholds and specified thresholds for trend strengths.

    Args:
        - dataframe (pd.DataFrame): Input DataFrame with log return data.
        - column (str): Column name for log returns (default is 'close').
        - lower_threshold (float): Threshold for moderate trends (default is 0.001).
        - upper_threshold (float): Threshold for strong trends (default is 0.02).
        - reverse_steps (int): Steps to confirm trend reversal (default is 7).
        - trends_to_keep (set): Trends to retain, others set to 0 (default is {0, 1, 2, 3, 4}).

    Returns:
        pd.DataFrame: DataFrame with 'trend' column:
                        - 0: No trend
                        - 1: Moderate negative trend
                        - 2: Very strong negative trend
                        - 3: Moderate positive trend
                        - 4: Very strong positive trend
                      Any trends not included in `trends_to_keep` will be reset to 0.

    Function Details:
    1. **Input Assumption**:
    - The input DataFrame already contains log return data in the specified column (`column`).

    2. **Trend Tracking**:
    - Uses dictionaries to monitor trends:
        - `dic1`: Tracks the first phase of the trend.
        - `dic2`: Tracks the second phase if a reversal occurs.
        - `dic3`: Tracks the third phase if another reversal occurs.

    3. **Cumulative Product**:
    - Calculates the cumulative product of `(1 + log_return)` from the specified column to evaluate the strength of trends.

    4. **Reversal Handling**:
    - If a trend reversal persists beyond `reverse_steps`, labels are assigned based on the cumulative product tracked in `dic1`.
    - Subsequent reversals are merged or labeled independently if conditions are met.

    5. **Label Assignment**:
    - Labels are dynamically assigned based on cumulative product thresholds for positive and negative trends:
        - Positive trends are categorized as moderate (3) or strong (4).
        - Negative trends are categorized as moderate (1) or strong (2).

    6. **Trend Filtering**:
    - After detecting trends, only those specified in `trends_to_keep` remain unchanged.
    - Any trend category not included in `trends_to_keep` is reset to 0 (No Trend).

    7. **Edge Cases**:
    - Properly handles scenarios where data points are insufficient for trend analysis or when trend phases overlap, ensuring all data points are labeled.
    """
    # Copy to avoid modifying the original DataFrame
    df = dataframe.copy()
    df['trend'] = None  # Default value 

    dic1, dic2, dic3 = None, None, None # Initialize trend tracking dictionaries
    
    def assign_label(dictio_, lower_threshold, upper_threshold):
        cumulative = dictio_['cumulative']
        if cumulative > (1 + upper_threshold):
            df.iloc[dictio_['ids'], df.columns.get_loc('trend')] = 4  # Very strong positive
        elif (1 + lower_threshold) < cumulative <= (1 + upper_threshold):
            df.iloc[dictio_['ids'], df.columns.get_loc('trend')] = 3  # Moderate positive
        elif (1 - upper_threshold) < cumulative <= (1 - lower_threshold):
            df.iloc[dictio_['ids'], df.columns.get_loc('trend')] = 1  # Moderate negative
        elif cumulative <= (1 - upper_threshold):
            df.iloc[dictio_['ids'], df.columns.get_loc('trend')] = 2  # Very strong negative
        else:
            df.iloc[dictio_['ids'], df.columns.get_loc('trend')] = 0  # No trend
    
    # Process each log return to detect trends
    for idx, log_ret in enumerate(df[column]):
        sign = 1 if log_ret > 0 else -1

        if dic1 is None:  # Initialize dic1
            dic1 = {'ids': [idx], 'last_sign': sign, 'cumulative': (1 + log_ret)}
            continue

        last_sign = dic1['last_sign']
        if sign == last_sign and dic2 is None:  # Continue same trend
            dic1['ids'].append(idx)
            dic1['last_sign'] = sign
            dic1['cumulative'] *= (1 + log_ret)
            continue

        # 1st Reversal occuring
        if dic2 is None:  # Start dic2
            dic2 = {'ids': [idx], 'last_sign': sign, 'cumulative': (1 + log_ret)}
            continue

        last_sign = dic2['last_sign']
        if sign == last_sign and dic3 is None:  # Continue same trend
            dic2['ids'].append(idx)
            dic2['last_sign'] = sign
            dic2['cumulative'] *= (1 + log_ret)
            if len(dic2['ids']) == reverse_steps:
                assign_label(dic1, lower_threshold, upper_threshold) # Assign labels in the 'trend' column for ids of dic1
                dic1, dic2 = dic2, None
            continue

        # 2nd Reversal occuring
        if dic3 is None:  # Start dic3
            dic3 = {'ids': [idx], 'last_sign': sign, 'cumulative': (1 + log_ret)}
            continue

        last_sign = dic3['last_sign']
        if sign == last_sign: # Continue same trend, there is no dic4 to check if is None
            dic3['ids'].append(idx)
            dic3['last_sign'] = sign
            dic3['cumulative'] *= (1 + log_ret)
            dic_prod = dic2['cumulative'] * dic3['cumulative']
            if (sign == 1 and dic_prod > 1) or (sign == -1 and dic_prod < 1):
                dic1['ids'] += dic2['ids'] + dic3['ids']
                dic1['last_sign'] = dic3['last_sign']
                dic1['cumulative'] *= dic2['cumulative'] * dic3['cumulative']
                dic2, dic3 = None, None
                continue

            if len(dic3['ids']) == reverse_steps:      
                assign_label(dic1, lower_threshold, upper_threshold) # Assign labels in the 'trend' column for ids of dic1
                assign_label(dic2, lower_threshold, upper_threshold) # Assign labels in the 'trend' column for ids of dic1
                dic1, dic2, dic3 = dic3, None, None
            continue
            
        # 3rd Reversal occuring
        assign_label(dic1, lower_threshold, upper_threshold) # Assign labels in the 'trend' column for ids of dic1
        dic1, dic2, dic3 = dic2, dic3, {'ids': [idx], 'last_sign': sign, 'cumulative': (1 + log_ret)}

    # Assign remaining labels
    if dic1:
        assign_label(dic1, lower_threshold, upper_threshold)
    if dic2:
        assign_label(dic2, lower_threshold, upper_threshold)
    if dic3:
        assign_label(dic3, lower_threshold, upper_threshold)
    
    # Apply filtering: Keep only selected trends, set others to 0
    df['trend'] = df['trend'].where(df['trend'].isin(trends_to_keep), 0)

    return df

def split_X_y(sequences: list[pd.DataFrame], 
              target_column: str = 'trend',
              detect_trends_function: Callable[[pd.DataFrame, str, float, float, int, set], pd.DataFrame] = detect_trends_4, 
              column: str = 'close', 
              lower_threshold: float = 0.0009, 
              upper_threshold: float = 0.015,
              reverse_steps: int = 7,
              trends_to_keep: set = {0, 1, 2, 3, 4}) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process sequences to generate features (X) and labels (y) with trend detection.

    Args:
        - sequences (list[pd.DataFrame]): List of DataFrame sequences.
        - target_column (str): Column name for labels (default is 'trend').
        - detect_trends_function (Callable): Trend detection function (default is detect_trends_4).
        - column (str): Column for trend detection (default is 'close').
        - lower_threshold (float): Lower threshold for trends (default is 0.0009).
        - upper_threshold (float): Upper threshold for trends (default is 0.015).
        - reverse_steps (int): Steps for trend reversal (default is 7).
        - trends_to_keep (set): Trends to retain (default is {0, 1, 2, 3, 4}).

    Returns:
        - Tuple[np.ndarray, np.ndarray]: X (features), y (labels) as NumPy arrays.
    """
    # Initialize lists for features and labels
    X, y = [], []
    
    # Process each sequence
    for seq in sequences:
        # Apply trend detection
        seq = detect_trends_function(seq, column, lower_threshold, upper_threshold, reverse_steps, trends_to_keep)
        
        # Extract features and labels
        X.append(seq.drop(columns=[target_column]).values)
        y.append(seq[target_column].values)
    
    # Convert to NumPy arrays
    return np.array(X), np.array(y)

def process_and_return_splits(
    with_indicators_file_path: str,
    downsampled_data_minutes: int,
    exclude_columns: list[str],
    lower_threshold: float,
    upper_threshold: float,
    reverse_steps: int,
    sequence_length: int,
    sliding_interval: int,
    trends_to_keep: set = {0, 1, 2, 3, 4}  # Default keeps all trends
) -> tuple[
    list[list[float]],  # X_train: List of sequences, each containing a list of features
    list[list[int]],    # y_train: List of sequences, each containing a list of labels
    list[list[float]],  # X_val: List of sequences, each containing a list of features
    list[list[int]],    # y_val: List of sequences, each containing a list of labels
    list[list[float]],  # X_test: List of sequences, each containing a list of features
    list[list[int]]     # y_test: List of sequences, each containing a list of labels
]:
    """
    Processes time-series data from a CSV file and prepares it for machine learning.

    This function performs the following steps:
        1. Reads data from the specified CSV file and sorts it by date in descending order.
        2. Optionally downsamples the data to a lower frequency (e.g., 5-minute intervals).
        3. Applies Gaussian smoothing to reduce noise in the data.
        4. Calculates log returns for all numeric columns, excluding specified columns.
        5. Detects trends based on defined thresholds (`lower_threshold`, `upper_threshold`, and `reverse_steps`).
        6. Filters trends to keep only those specified in `trends_to_keep`, setting others to 0 (No Trend).
        7. Converts the processed data into sequences of a fixed length (`sequence_length`) with a sliding interval.
        8. Splits the sequences into training (80%), validation (10%), and test (10%) sets.
        9. Further splits the sequences into features (`X`) and labels (`y`) for supervised learning.

    Args:
        - with_indicators_file_path (str): Path to the CSV file with time-series data.
        - downsampled_data_minutes (int): Frequency for downsampling (e.g., 1 for no downsampling).
        - exclude_columns (list[str]): Columns to exclude from log return calculations.
        - lower_threshold (float): Lower threshold for trend detection.
        - upper_threshold (float): Upper threshold for trend detection.
        - reverse_steps (int): Steps for reversing trends in trend detection.
        - sequence_length (int): Length of sequences to create.
        - sliding_interval (int): Interval for sliding the window.
        - trends_to_keep (set): Trends to retain, others set to 0 (default is {0, 1, 2, 3, 4}).

    Returns:
        - tuple: X_train, y_train, X_val, y_val, X_test, y_test as lists of sequences.
    """
    def check_missing_timestamps(data: pd.DataFrame, stage: str):
        """
        Checks for missing timestamps and prints diagnostic info.
        """
        missing_timestamps = pd.date_range(
            start=data.index.min(),
            end=data.index.max(),
            freq='1min',  # Checking 1-minute frequency
            tz=data.index.tz,
        ).difference(data.index)

        print(f"\n{stage} - Missing timestamps: \n{missing_timestamps}")

        if not missing_timestamps.empty:
            for timestamp in missing_timestamps[:5]:  # Show only first 5 missing timestamps
                print(f"\nMissing timestamp: {timestamp}")

                before = data[data.index < timestamp].tail(5)  # 5 data points before
                after = data[data.index > timestamp].head(5)  # 5 data points after

                print("\nData before missing timestamp:")
                display(before) if not before.empty else print("No data available before.")

                print("\nData after missing timestamp:")
                display(after) if not after.empty else print("No data available after.")

    print("\n======== Processing Time-Series Data ========")

    # Step 1: Read & Sort Data
    data = read_csv_file(with_indicators_file_path, preview_rows=0).sort_index(ascending=False)

    # Step 2: Downsample Data
    if downsampled_data_minutes != 1:
        print("\n---> Downsampling Data")
        data = downsample_minute_data(data, downsampled_data_minutes)

    check_missing_timestamps(data, "Data Retrieved")

    # Step 3: Gaussian Smoothing
    data = gaussian_smoothing(data, sigma=7)
    check_missing_timestamps(data, "Gaussian Smoothed Data")

    # Step 4: Compute Log Returns
    data = calculate_log_returns_all_columns(data, exclude_columns=exclude_columns)
    check_missing_timestamps(data, "Log Returns Computed")

    # Step 5: Create Sequences
    sequences = created_sequences_2(data, sequence_length, sliding_interval)

    # Step 6: Train / Validation / Test Split
    train_size = int(len(sequences) * 0.8)
    val_size = int(len(sequences) * 0.1)

    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:train_size + val_size]
    test_sequences = sequences[train_size + val_size:]

    print(f"\nNumber of sequences:\n"
          f"  - Total sequences: {len(sequences)}\n"
          f"  - Train: {len(train_sequences)}\n"
          f"  - Validation: {len(val_sequences)}\n"
          f"  - Test: {len(test_sequences)}\n")

    # Step 7: Convert Sequences to X, y
    def split_and_format_data(sequences):
        X, y = split_X_y(
            sequences, target_column='trend',
            detect_trends_function=detect_trends_4,
            column='close', lower_threshold=lower_threshold,
            upper_threshold=upper_threshold, reverse_steps=reverse_steps,
            trends_to_keep=trends_to_keep
        )
        return np.array(X), np.array(y)

    X_train, y_train = split_and_format_data(train_sequences)
    X_val, y_val = split_and_format_data(val_sequences)
    X_test, y_test = split_and_format_data(test_sequences)

    # Step 8: Data Integrity Check (Ensuring Proper Types)
    def check_data_types(X: np.ndarray, y: np.ndarray, label: str):
        """
        Checks if all values in X are float and y are integer.
        """
        unexpected_X = [(i, j, k, type(v)) for i, seq in enumerate(X)
                        for j, row in enumerate(seq)
                        for k, v in enumerate(row) if not isinstance(v, (float, np.float32))]
        unexpected_y = [(i, j, type(v)) for i, seq in enumerate(y)
                        for j, v in enumerate(seq) if not isinstance(v, (int, np.int64))]

        if unexpected_X:
            print(f"\n⚠️ Unexpected type in {label} X:")
            for i, j, k, t in unexpected_X[:5]:  # Show first 5 errors
                print(f"  Sequence {i}, Row {j}, Feature {k}: {t}")

        if unexpected_y:
            print(f"\n⚠️ Unexpected type in {label} y:")
            for i, j, t in unexpected_y[:5]:  # Show first 5 errors
                print(f"  Sequence {i}, Label {j}: {t}")

    check_data_types(X_train, y_train, "Train")
    check_data_types(X_val, y_val, "Validation")
    check_data_types(X_test, y_test, "Test")

    # Step 9: Convert y types if needed
    def convert_dtype(y: np.ndarray):
        return np.array(y, dtype=np.int64) if isinstance(y, np.ndarray) and y.dtype == np.object_ else y

    y_train, y_val, y_test = map(convert_dtype, [y_train, y_val, y_test])

    # Get feature info
    Number_features = X_train.shape[-1]
    close_col_index = data.columns.get_loc('close')
    
    print(f"\nFeature Info:\n  - close_col_index = {close_col_index}\n  - Number_features = {Number_features}")

    return X_train, y_train, X_val, y_val, X_test, y_test, Number_features

def print_class_distribution(y, var_name: str) -> None:
    """
    Prints the class distribution of a label array.

    Args:
        y: Tensor, array, or list of class labels.
        var_name: Name of the variable (for display).
    """
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    flattened = np.array(y).flatten()

    unique_classes, counts = np.unique(flattened, return_counts=True)
    total = counts.sum()

    header = f"Class Distribution for '{var_name}':"
    line_parts = [
        f"Class {int(c):<3} Percent: {(count / total) * 100:>6.2f}%"
        for c, count in zip(unique_classes, counts)
    ]
    print(header.ljust(40) + " || ".join(line_parts))

def print_model_info(model):
    """
    Print total number of parameters and estimated model size in MB (float32).
    """
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = total_params * 4  # Assuming float32 (4 bytes per param)
    param_size_MB = param_size_bytes / (1024**2)

    print(f"🧠 Total Parameters        : {total_params}")
    print(f"📦 Model Size (float32)   : {param_size_MB:.2f} MB")

def compute_fwt_fixed_verbose(previous_model, init_model, X_val, y_val, known_classes, batch_size=64):
    """
    Compute Forward Transfer (FWT) for sequence-level inputs with known class filtering.

    Args:
        previous_model (nn.Module): Well-trained model after previous period.
        init_model (nn.Module): Newly initialized model before current training.
        X_val (Tensor): Validation input, shape [B, L, F].
        y_val (Tensor): Validation labels, shape [B, L].
        known_classes (list[int]): List of classes considered known before current period.
        batch_size (int): Evaluation batch size.

    Returns:
        fwt_value (float): Acc(prev) - Acc(init) over known classes.
        acc_prev (float): Accuracy from previous model.
        acc_init (float): Accuracy from init model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous_model.to(device).eval()
    init_model.to(device).eval()

    B, L, F = X_val.shape
    y_val_flat = y_val.view(-1)  # [B * L]
    mask = torch.isin(y_val_flat, torch.tensor(known_classes, device=y_val.device))
    indices_flat = mask.nonzero(as_tuple=False).squeeze()

    # Map token indices back to sequence-level batch indices
    batch_indices = indices_flat // L
    batch_indices = batch_indices.unique()

    X_known = X_val[batch_indices]
    y_known = y_val[batch_indices]

    if len(y_known) == 0:
        print(f"⚠️ No validation samples for known classes {known_classes}.")
        return None, None, None

    print(f"📋 Total matching sequences for known classes {known_classes}: {len(y_known)}")

    loader = DataLoader(TensorDataset(X_known, y_known), batch_size=batch_size)
    correct_prev, correct_init, total = 0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if xb.dim() == 2:
                xb = xb.unsqueeze(0)
                yb = yb.unsqueeze(0)

            out_prev = previous_model(xb).view(-1, previous_model.output_size)
            out_init = init_model(xb).view(-1, init_model.output_size)
            yb_flat = yb.view(-1)

            mask = torch.isin(yb_flat, torch.tensor(known_classes, device=yb.device))
            total_batch = mask.sum().item()

            correct_prev += (torch.argmax(out_prev, dim=-1)[mask] == yb_flat[mask]).sum().item()
            correct_init += (torch.argmax(out_init, dim=-1)[mask] == yb_flat[mask]).sum().item()
            total += total_batch

    acc_prev = correct_prev / total
    acc_init = correct_init / total
    fwt_value = acc_prev - acc_init

    print(f"\n### 🔍 FWT Debug Info:")
    print(f"- Total evaluated tokens   : {total}")
    print(f"- Acc (Prev Model)         : {acc_prev:.4f}")
    print(f"- Acc (Init Model)         : {acc_init:.4f}")
    print(f"- ➕ FWT = Acc_prev - Acc_init = {fwt_value:.4f}")

    return fwt_value, acc_prev, acc_init

def compute_classwise_accuracy(student_logits_flat, y_batch, class_correct, class_total):
    """
    Computes per-class accuracy by accumulating correct and total samples for each class using vectorized operations.
    
    Args:
        student_logits_flat (torch.Tensor): Model predictions (logits) in shape [batch_size * seq_len, output_size]
        y_batch (torch.Tensor): True labels in shape [batch_size * seq_len]
        class_correct (dict): Dictionary to store correct predictions per class
        class_total (dict): Dictionary to store total samples per class
    """
    # Ensure inputs are on the same device
    if student_logits_flat.device != y_batch.device:
        raise ValueError("student_logits_flat and y_batch must be on the same device")

    # Convert logits to predicted class indices
    predictions = torch.argmax(student_logits_flat, dim=-1)  # Shape: [batch_size * seq_len]

    # Compute correct predictions mask
    correct_mask = (predictions == y_batch)  # Shape: [batch_size * seq_len], boolean

    # Get unique labels in this batch
    unique_labels = torch.unique(y_batch)

    # Update class_total and class_correct using vectorized operations
    for label in unique_labels:
        label = label.item()  # Convert tensor to scalar
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        
        # Count total samples for this label
        label_mask = (y_batch == label)
        class_total[label] += label_mask.sum().item()
        
        # Count correct predictions for this label
        class_correct[label] += (label_mask & correct_mask).sum().item()

def setup_file_paths(pair='BTCUSD', base_dir='Data', days=190):
    """
    Set up file paths for cryptocurrency data across multiple periods.

    Args:
        pair (str): Trading pair (e.g., 'BTCUSD').
        base_dir (str): Base directory for data storage (default 'Data').
        days (int): Number of days for each period (default 190).

    Returns:
        tuple: (base_folder_path, with_indicators_file_path, list_period_files_full_path)
    """
    # Define base file name and folder structure
    file_name = f"Polygon_{pair}_4Y_1min"
    base_folder_path = os.path.normpath(os.path.join(base_dir, file_name))
    
    # Check if folder exists
    if not os.path.isdir(base_folder_path):
        raise FileNotFoundError(f"Directory '{base_folder_path}' does not exist.")

    # Define file path with indicators for Period 1
    with_indicators_file_path = os.path.normpath(
        os.path.join(base_folder_path, f"_{file_name}_{days}_days_with_indicators.csv")
    )

    # Define file paths for all periods
    list_period_files_full_path = [
        # Period 1
        with_indicators_file_path,
        # Period 2: 2020-11-11 to 2021-05-20
        os.path.normpath(os.path.join(
            base_folder_path, f"{file_name}_{days}_days__2020-11-11__2021-05-20__with_indicators.csv"
        )),
        # Period 3: 2021-05-20 to 2021-11-26
        os.path.normpath(os.path.join(
            base_folder_path, f"{file_name}_{days}_days__2021-05-20__2021-11-26__with_indicators.csv"
        )),
        # Period 4: 2021-11-26 to 2022-06-04
        os.path.normpath(os.path.join(
            base_folder_path, f"{file_name}_{days}_days__2021-11-26__2022-06-04__with_indicators.csv"
        )),
        # Period 5: 2022-06-04 to 2022-12-11
        os.path.normpath(os.path.join(
            base_folder_path, f"{file_name}_{days}_days__2022-06-04__2022-12-11__with_indicators.csv"
        )),
    ]

    return base_folder_path, with_indicators_file_path, list_period_files_full_path

def print_folder_contents(folder_path):
    """Print all files in the specified folder."""
    print("\n📂 Folder Contents:")
    for file in os.listdir(folder_path):
        print(f"Found file: {file}")

if __name__ == "__main__":
    # Set up paths
    base_folder_path, with_indicators_file_path, list_period_files_full_path = setup_file_paths()

    # Print results
    print("=" * 70)
    print("File Path Configuration".center(70))
    print("=" * 70)
    
    print(f"{'Base Folder Path':<25}: {base_folder_path}")
    print(f"{'Period 1 File Path':<25}: {with_indicators_file_path}")
    print("-" * 70)
    
    print("List of Period Files:")
    for i, path in enumerate(list_period_files_full_path, 1):
        print(f"{'Period ' + str(i):<25}: {path}")
    
    print("=" * 70)

    # Print folder contents
    print_folder_contents(base_folder_path)


def check_gpu_config():
    """
    Check GPU availability and display detailed configuration information.
    """
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    
    # Print header
    print("=" * 50)
    print("GPU Configuration Check".center(50))
    print("=" * 50)
    
    # Basic GPU availability
    print(f"{'PyTorch Version':<25}: {torch.__version__}")
    print(f"{'GPU Available':<25}: {'Yes' if gpu_available else 'No'}")
    
    # If GPU is available, print detailed info
    if gpu_available:
        print("-" * 50)
        print("GPU Details".center(50))
        print("-" * 50)
        
        # Device info
        print(f"{'Device Name':<25}: {torch.cuda.get_device_name(0)}")
        print(f"{'Number of GPUs':<25}: {torch.cuda.device_count()}")
        print(f"{'Current Device Index':<25}: {torch.cuda.current_device()}")
        
        # Compute capability and CUDA cores
        props = torch.cuda.get_device_properties(0)
        print(f"{'Compute Capability':<25}: {props.major}.{props.minor}")
        print(f"{'Total CUDA Cores':<25}: {props.multi_processor_count * 128}")  # Approx. 128 cores per SM
        
        # Memory info
        total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"{'Total Memory (GB)':<25}: {total_memory:.2f}")
        print(f"{'Allocated Memory (GB)':<25}: {memory_allocated:.2f}")
        print(f"{'Reserved Memory (GB)':<25}: {memory_reserved:.2f}")
    else:
        print("-" * 50)
        print("No GPU detected. Running on CPU.".center(50))
        print("-" * 50)
    
    print("=" * 50)

if __name__ == "__main__":
    check_gpu_config()


def print_torch_config():
    """Print PyTorch and CUDA configuration in a formatted manner."""
    print("=" * 50)
    print("PyTorch Configuration".center(50))
    print("=" * 50)
    
    # Basic PyTorch and CUDA info
    print(f"{'PyTorch Version':<25}: {torch.__version__}")
    print(f"{'CUDA Compiled Version':<25}: {torch.version.cuda}")
    print(f"{'CUDA Available':<25}: {'Yes' if torch.cuda.is_available() else 'No'}")
    print(f"{'Number of GPUs':<25}: {torch.cuda.device_count()}")

    # GPU details if available
    if torch.cuda.is_available():
        print(f"{'GPU Name':<25}: {torch.cuda.get_device_name(0)}")

    print("-" * 50)
    
    # Seed setting
    torch.manual_seed(42)
    print(f"{'Random Seed':<25}: 42 (Seeding successful!)")
    
    print("=" * 50)

if __name__ == "__main__":
    print_torch_config()

# Configuration Parameters
ticker = 'BTC-USD'
downsampled_data_minutes = 1  # No downsampling (1-minute interval retained)

# === Trend Detection Parameters (for 1,000-point sequences) ===
lower_threshold = 0.0009  # Lower threshold: even small price movements may be considered trends
upper_threshold = 0.015   # Upper threshold: only stronger movements are marked as strong trends
reverse_steps = 13        # Minimum number of consecutive reversals to confirm a trend change

# === Features to exclude from log return and trend calculation ===
exclude_columns = ['MACD', 'MACD_signal', 'ROC_10', 'OBV', 'AD_Line']

# === Feature Selection by Correlation with 'trend' column ===
# Strongly correlated (correlation > 0.6): Keep
strongly_correlated = ['close', 'open', 'SMA_5', 'high', 'low', 'EMA_10', 'SMA_10']

# Moderately correlated (correlation between 0.3 and 0.6): Exclude
moderately_correlated = ['BB_middle', 'BB_lower', 'BB_upper', 'RSI_14']

# Weakly or uncorrelated (correlation <~ 0.3): Exclude
weakly_correlated = ['SMA_50', 'volume', 'BBW', 'ATR_14']

# Extend the exclude list to include all weakly and moderately correlated features
exclude_columns += weakly_correlated + moderately_correlated

# === Sequence Generation Parameters ===
sequence_length = 1000       # Window size for each time-series sample
sliding_interval = 60        # Stride between two consecutive sequences


# Model
class LoRA(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int):
        """
        LoRA module applied to a specified linear layer.

        Args:
            linear_layer (nn.Linear): The linear layer to adapt (e.g., attention_fc or fc).
            rank (int): The rank of the LoRA adjustment matrices (e.g., 8).
        """
        super(LoRA, self).__init__()
        self.linear = linear_layer
        self.rank = rank
        
        # Get input and output dimensions from the linear layer
        in_features, out_features = linear_layer.weight.shape
        
        # Create LoRA matrices A and B
        self.A = nn.Parameter(torch.zeros(in_features, rank))  # Shape: (in_features, rank)
        self.B = nn.Parameter(torch.zeros(rank, out_features))  # Shape: (rank, out_features)
        
        # Initialize A with normal distribution, B with zeros
        nn.init.normal_(self.A, mean=0, std=1)
        nn.init.zeros_(self.B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adjustment applied to the linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with LoRA-adapted weights.
        """
        lora_delta = self.A @ self.B
        adapted_weight = self.linear.weight + lora_delta
        return nn.functional.linear(x, adapted_weight, self.linear.bias)
    
    def parameters(self, recurse=True):
        """
        Override parameters() to return only LoRA-specific parameters (A and B).

        Args:
            recurse (bool): Ignored, included for compatibility with nn.Module.

        Returns:
            list: List of LoRA parameters (self.A and self.B).
        """
        return [self.A, self.B]
    

class BiGRUWithAttention_LoRA(nn.Module):
    """
    A BiGRU model with attention mechanism and optional LoRA adapter.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of GRU units per direction.
        output_size (int): Number of output classes.
        num_layers (int): Number of stacked GRU layers.
        dropout (float): Dropout rate.
        lora_rank (int): Rank of the low-rank adaptation module (LoRA).
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0, lora_rank=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lora_rank = lora_rank

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0)

        self.attention_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.lora_adapter = None  # Initialized from Period 2 onward
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def init_lora(self):
        """Initialize the LoRA adapter (only once, from Period 2 onward)."""
        if self.lora_adapter is None:
            self.lora_adapter = LoRA(self.attention_fc, self.lora_rank).to(next(self.parameters()).device)
            print("Initialized LoRA adapter")

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)

        attn_out = self.lora_adapter(out) if self.lora_adapter else self.attention_fc(out)
        attn_weights = torch.tanh(attn_out)
        out = attn_weights * out
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def get_trainable_parameters(self):
        """
        Returns trainable parameters for optimization (LoRA + FC).
        Prints a summary of parameter statistics.
        """
        lora_params = []
        lora_names = []
        fc_params = []
        fc_names = []

        total_params = sum(p.numel() for p in self.parameters())

        # Collect LoRA parameters (A and B)
        if self.lora_adapter:
            lora_params += [self.lora_adapter.A, self.lora_adapter.B]
            lora_names += ['lora_adapter.A', 'lora_adapter.B']

        # Collect FC layer parameters
        for name, param in self.fc.named_parameters():
            fc_params.append(param)
            fc_names.append(f"fc.{name}")

        trainable_params = lora_params + fc_params
        lora_param_count = sum(p.numel() for p in lora_params)
        fc_param_count = sum(p.numel() for p in fc_params)
        trainable_param_count = lora_param_count + fc_param_count
        frozen_params = total_params - trainable_param_count

        # Print statistics
        print(f"📊 Parameter Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_param_count:,} ({trainable_param_count / total_params * 100:.2f}%)")
        print(f"    - LoRA parameters: {lora_param_count:,} ({lora_param_count / total_params * 100:.2f}%)")
        print(f"    - FC parameters: {fc_param_count:,} ({fc_param_count / total_params * 100:.2f}%)")
        print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")

        print(f"🧠 Trainable parameter names:")
        for name in lora_names:
            print(f"  ✅ {name} (LoRA)")
        for name in fc_names:
            print(f"  ✅ {name} (FC)")

        return trainable_params

# Training Function
def train_lora_baseline(model, output_size, criterion, optimizer,
                        X_train, y_train, X_val, y_val, scheduler=None,
                        num_epochs=10, batch_size=64, model_saving_folder=None,
                        model_name=None, stop_signal_file=None):
    print("\n🚀 'train_lora_baseline' started.")

    # Prepare saving folder
    if model_saving_folder:
        if os.path.exists(model_saving_folder):
            shutil.rmtree(model_saving_folder)
            print(f"✅ Removed existing folder: {model_saving_folder}")
        os.makedirs(model_saving_folder, exist_ok=True)

    model_name = model_name or 'model'
    model_saving_folder = model_saving_folder or './saved_models'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tensor to device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    print("\n✅ Data Overview:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

    best_results = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        class_correct, class_total = {}, {}

        if stop_signal_file and os.path.exists(stop_signal_file):
            print("\n🛑 Stop signal detected. Exiting training loop.")
            break

        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1, output_size)
            y_batch = y_batch.view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            compute_classwise_accuracy(outputs, y_batch, class_correct, class_total)

        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = {int(c): f"{(class_correct[c] / class_total[c]) * 100:.2f}%" if class_total[c] > 0 else "0.00%"
                     for c in sorted(class_total.keys())}

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_class_correct, val_class_total = {}, {}
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).view(-1, output_size)
                y_batch = y_batch.view(-1)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
                predictions = torch.argmax(outputs, dim=-1)
                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.size(0)
                compute_classwise_accuracy(outputs, y_batch, val_class_correct, val_class_total)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_acc_cls = {int(c): f"{(val_class_correct[c] / val_class_total[c]) * 100:.2f}%" if val_class_total[c] > 0 else "0.00%"
                       for c in sorted(val_class_total.keys())}

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Train-Class-Acc: {train_acc},")
        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc * 100:.2f}%, Val-Class-Acc: {val_acc_cls}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        model_path = os.path.join(model_saving_folder, f"{model_name}_epoch_{epoch+1}.pth")
        current = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'train_classwise_accuracy': train_acc,
            'val_classwise_accuracy': val_acc_cls,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'model_path': model_path
        }

        if len(best_results) < 5 or val_acc > best_results[-1]['val_accuracy']:
            if len(best_results) == 5:
                to_remove = best_results.pop()
                if os.path.exists(to_remove['model_path']):
                    os.remove(to_remove['model_path'])
                    print(f"🗑 Removed: {to_remove['model_path']}")
            best_results.append(current)
            best_results.sort(key=lambda x: (x['val_accuracy'], x['epoch']), reverse=True)
            torch.save(current, model_path)
            print(f"✅ Saved model: {model_path}")

        if scheduler: scheduler.step(val_loss)

    if best_results:
        best = best_results[0]
        best_model_path = os.path.join(model_saving_folder, f"{model_name}_best.pth")
        torch.save(best, best_model_path)
        print(f"\n🏆 Best model saved as: {best_model_path} (Val Accuracy: {best['val_accuracy'] * 100:.2f}%)")

    final_model_path = os.path.join(model_saving_folder, f"{model_name}_final.pth")
    torch.save(current, final_model_path)
    print(f"\n📌 Final model saved as: {final_model_path}")

    print("\n🎯 Top 5 Best Models:")
    for res in best_results:
        print(f"Epoch {res['epoch']}, Train Loss: {res['train_loss']:.6f}, Train-Acc: {res['train_classwise_accuracy']},\n"
              f"Val Loss: {res['val_loss']:.6f}, Val Acc: {res['val_accuracy']*100:.2f}%, Val-Acc: {res['val_classwise_accuracy']},"
              f" Model Path: {res['model_path']}")

    del X_train, y_train, X_val, y_val, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


device = auto_select_cuda_device()

# Period 1
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    X_train, y_train, X_val, y_val, X_test, y_test, Number_features = process_and_return_splits(
        with_indicators_file_path = "path/to/your/period1_file.csv",
        downsampled_data_minutes = downsampled_data_minutes,
        exclude_columns = exclude_columns,
        lower_threshold = lower_threshold,
        upper_threshold = upper_threshold,
        reverse_steps = reverse_steps,
        sequence_length = sequence_length,
        sliding_interval = sliding_interval,
        trends_to_keep = {0, 1}
    )

input_size = Number_features
hidden_size = 64
output_size = len(np.unique(y_val))
num_layers = 4
dropout = 0.0
lora_r = 4
num_epochs = 1000
batch_size = 64

stop_signal_file = "path/to/your/stop_signal.txt"
model_saving_folder = "path/to/your/period1_folder"
ensure_folder(model_saving_folder)

model = BiGRUWithAttention_LoRA(input_size, hidden_size, output_size, num_layers, dropout, lora_r).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Full parameter training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10)

# === Train ===
train_lora_baseline(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    scheduler=scheduler,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name='BiGRUWithAttention',
    stop_signal_file=stop_signal_file
)

# Period 2
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    X_train, y_train, X_val, y_val, X_test, y_test, Number_features = process_and_return_splits(
        with_indicators_file_path = "path/to/your/period2_file.csv",
        downsampled_data_minutes = downsampled_data_minutes,
        exclude_columns = exclude_columns,
        lower_threshold = lower_threshold,
        upper_threshold = upper_threshold,
        reverse_steps = reverse_steps,
        sequence_length = sequence_length,
        sliding_interval = sliding_interval,
        trends_to_keep = {0, 1, 2}
    )

input_size = Number_features
hidden_size = 64
output_size = len(np.unique(y_val))
num_layers = 4
dropout = 0.0
lora_r = 4
num_epochs = 1000
batch_size = 64

stop_signal_file = "path/to/your/stop_signal.txt"
model_saving_folder = "path/to/your/period2_folder"
ensure_folder(model_saving_folder)

prev_path = "path/to/your/period1_folder/BiGRUWithAttention_best.pth"
checkpoint = torch.load(prev_path, map_location=device)
prev_state = checkpoint["model_state_dict"]

model = BiGRUWithAttention_LoRA(input_size, hidden_size, output_size, num_layers, dropout, lora_r).to(device)
model.load_state_dict({k: v for k, v in prev_state.items() if not k.startswith("fc.") and not k.startswith("lora_adapter.")}, strict=False)
model.init_lora()

if any(k.startswith("lora_adapter.") for k in prev_state.keys()):
    model.lora_adapter.load_state_dict({k.replace("lora_adapter.", ""): v for k, v in prev_state.items() if k.startswith("lora_adapter.")})

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.get_trainable_parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10)

train_lora_baseline(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    scheduler=scheduler,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name='BiGRUWithAttention',
    stop_signal_file=stop_signal_file
)

# Period 3
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    X_train, y_train, X_val, y_val, X_test, y_test, Number_features = process_and_return_splits(
        with_indicators_file_path = "path/to/your/period3_file.csv",
        downsampled_data_minutes = downsampled_data_minutes,
        exclude_columns = exclude_columns,
        lower_threshold = lower_threshold,
        upper_threshold = upper_threshold,
        reverse_steps = reverse_steps,
        sequence_length = sequence_length,
        sliding_interval = sliding_interval,
        trends_to_keep = {0, 1, 2, 3}
    )

input_size = Number_features
hidden_size = 64
output_size = len(np.unique(y_val))
num_layers = 4
dropout = 0.0
lora_r = 4
num_epochs = 1000
batch_size = 64

stop_signal_file = "path/to/your/stop_signal.txt"
model_saving_folder = "path/to/your/period3_folder"
ensure_folder(model_saving_folder)

prev_path = "path/to/your/period2_folder/BiGRUWithAttention_best.pth"
checkpoint = torch.load(prev_path, map_location=device)
prev_state = checkpoint["model_state_dict"]

model = BiGRUWithAttention_LoRA(input_size, hidden_size, output_size, num_layers, dropout, lora_r).to(device)
model.init_lora()
model.load_state_dict({k: v for k, v in prev_state.items() if not k.startswith("fc.")}, strict=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.get_trainable_parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10)

train_lora_baseline(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    scheduler=scheduler,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name='BiGRUWithAttention',
    stop_signal_file=stop_signal_file
)

# Period 4
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    X_train, y_train, X_val, y_val, X_test, y_test, Number_features = process_and_return_splits(
        with_indicators_file_path = list_period_files_full_path[3],
        downsampled_data_minutes = downsampled_data_minutes,
        exclude_columns = exclude_columns,
        lower_threshold = lower_threshold,
        upper_threshold = upper_threshold,
        reverse_steps = reverse_steps,
        sequence_length = sequence_length,
        sliding_interval = sliding_interval,
        trends_to_keep = {0, 1, 2, 3, 4}
    )

input_size = Number_features
hidden_size = 64
output_size = len(np.unique(y_val))
num_layers = 4
dropout = 0.0
lora_r = 4
num_epochs = 1000
batch_size = 64

stop_signal_file = "path/to/your/stop_signal.txt"
model_saving_folder = "path/to/your/period4_folder"
ensure_folder(model_saving_folder)

prev_path = "path/to/your/period3_folder/BiGRUWithAttention_best.pth"
checkpoint = torch.load(prev_path, map_location=device)
prev_state = checkpoint["model_state_dict"]

model = BiGRUWithAttention_LoRA(input_size, hidden_size, output_size, num_layers, dropout, lora_r).to(device)
model.init_lora()
model.load_state_dict({k: v for k, v in prev_state.items() if not k.startswith("fc.")}, strict=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.get_trainable_parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10)

train_lora_baseline(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    scheduler=scheduler,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name='BiGRUWithAttention',
    stop_signal_file=stop_signal_file
)
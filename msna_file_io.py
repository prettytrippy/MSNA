import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_burst_idxs(msna):
    return np.nonzero(msna['BURST'].to_numpy())[0]
    
# Function to read and parse the data file
def read_msna(file_path):
    # NOTE: I did not write the majority of this function, I need to find out who did
    
    # Dictionary to hold metadata
    metadata = {}
    # List to hold data values
    data = []
    
    # Read file
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    data = []
    channel_titles = []

    for line in lines:
        if line.startswith("ChannelTitle="):
            # Extract channel titles
            channel_titles = line.strip().split('\t')[1:]
        elif line.strip() and not line.startswith(("Interval=", "ExcelDateTime=", "TimeFormat=", "DateFormat=", "ChannelTitle=", "Range=", "UnitName=", "TopValue=", "BottomValue=")):
            # Split the line by tab
            parts = line.strip().split('\t')
            # Extract BURST comment if present
            burst_comment = ""

            # TODO: CHECK OTHER BURST NAMES
            if "BURST" in line:
                burst_comment = list(parts[-1])[4:]
                burst_comment = "".join(burst_comment)
                burst_flag = 1
            else:
                burst_flag = 0
            
            # Extract relevant data
            timestamp = parts[0]
            ecg = parts[1]  # ECG is the first channel after Timestamp
            nibp = parts[2]
            handgrip = parts[3]
            respiratory_waveform = parts[4]
            systolic = parts[5]
            diastolic = parts[6]
            heart_rate = parts[7]
            raw_msna = parts[8]
            stimulator = parts[9]
            filtered_msna = parts[10]
            integrated_msna = parts[11]  # Integrated MSNA is the 12th channel
            respiratory_rate = parts[12]
            filter_other = parts[13]
            percent_mvc = parts[14]
            
            data.append([timestamp, ecg, nibp, integrated_msna, burst_flag])

    if not channel_titles:
        print("Error: Channel titles were not found in the metadata.")
        return None
  
    # Convert to DataFrame
    new_channels = ['Timestamp', 'ECG', 'NIBP', 'Integrated MSNA', 'BURST']
    df = pd.DataFrame(data, columns=new_channels)
    df = df.apply(pd.to_numeric, errors='coerce')

    for channel in new_channels:
        df[channel] = pd.to_numeric(df[channel], errors='coerce')
    if len(get_burst_idxs(df)) == 0:
        print("No bursts found.")
        return None
    
    return df

def get_dataframes(glob_regex="../MSNAS/MSNA*/MSNA*burstcomments*.txt"):
    file_list = glob.glob(glob_regex) 
    dfs = []

    for filename in tqdm(file_list):
        try:
            df = read_msna(filename)
            if 1 in df['BURST']:
                dfs.append(df)
        except Exception as e:
            print(e, filename)
    
    return dfs

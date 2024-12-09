import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from msna_pipeline import MSNA_pipeline
from msna_file_io import get_dataframes

dfs = get_dataframes()
pipeline = MSNA_pipeline(sampling_rate=250)
pipeline.load("../pretrained")

df = dfs[0]

# Extract indices of the actual BURST events
actual_bursts = df[df['BURST'] != 0].index

peaks = pipeline.predict(df)

plot_start = 0 #np.random.randint(1000, 2000)
plot_end = len(df) #plot_start + np.random.randint(20000, 30000)

# Plot the results
plt.figure(figsize=(14, 8))
plt.plot(df['Integrated MSNA'][plot_start:plot_end], label='Filtered Signal')
plt.plot(actual_bursts[(actual_bursts<plot_end) & (actual_bursts>plot_start)], 
         df['Integrated MSNA'][actual_bursts[(actual_bursts<plot_end) & (actual_bursts>plot_start)]], 
         "o", label='Actual BURST Events', color='blue')
plt.plot(peaks[(peaks<plot_end) & (peaks>plot_start)], 
         df['Integrated MSNA'][peaks[(peaks<plot_end) & (peaks>plot_start)]], 
         "x", label='Detected Peaks', color='red')
plt.title('Detected Peaks vs Actual BURST Events')
plt.xlabel('Sample Index')
plt.ylabel('Filtered Signal')
plt.legend()
plt.show()
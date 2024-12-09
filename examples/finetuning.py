import sys
sys.path.append("..")
from msna_file_io import read_msna
from msna_pipeline import MSNA_pipeline

pipeline = MSNA_pipeline()
pipeline.load("../pretrained")

df = read_msna("new_msna_file.txt")

# Finetune on new file
# Use much smaller parameters than training.py
pipeline.train(df, threshold_train_max_iter=8, num_epochs=4, learning_rate=0.00001)

# Save new parameters
pipeline.save("../pretrained")
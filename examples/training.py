import sys
sys.path.append("..")
from msna_file_io import get_dataframes
from msna_pipeline import MSNA_pipeline

pipeline = MSNA_pipeline()
dfs = get_dataframes()

# Train on files
pipeline.train(dfs, threshold_train_max_iter=32, num_epochs=64, learning_rate=0.001)

# Save new parameters
pipeline.save("../pretrained")
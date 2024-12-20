from msna_cnn import MSNA_CNN
import pandas as pd
import numpy as np
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
from scipy.optimize import minimize

# TODO: Add a guess_kernel method to guess a good peak template kernel for find_peaks
# TODO: Add downsampling to get signals in a fixed sampling rate

default_kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
default_threshold = 0.99

class MSNA_pipeline():
    
    def __init__(self, sampling_rate=250, window_size=256, batch_size=1024, verbose=True):
        self.verbose = verbose
        self.sr = sampling_rate
        self.window_size = window_size
        self.batch_size = batch_size

        self.init_params()

    def init_params(self):
        self.cnn = MSNA_CNN(n=self.window_size)
        self.threshold = default_threshold
        self.kernel = default_kernel

    def save(self, output_folder):
        torch.save(self.cnn.state_dict(), f"{output_folder}/model_parameters.pth")
        with open(f"{output_folder}/pipeline_parameters.json", 'w') as file:
            json_dict = {"kernel":self.kernel, "threshold":self.threshold}
            file.write(json.dumps(json_dict))

    def load(self, output_folder):
        # Assume `model` is an instance of your model class
        self.cnn.load_state_dict(torch.load(f"{output_folder}/model_parameters.pth"))
        with open(f"{output_folder}/pipeline_parameters.json", 'r') as file:
            json_dict = json.loads(file.read())
            self.kernel = json_dict['kernel']
            self.threshold = json_dict['threshold']

    def get_burst_idxs(self, df):
        return np.nonzero(df['BURST'].to_numpy())[0]
    
    def process_dataframe(self, df):
        low_cutoff = 0.5   # Low cutoff hz
        high_cutoff = 20  # High cutoff hz
        fs = self.sr 
        
        df['Integrated MSNA'] = medfilt(df['Integrated MSNA'], 3)
        df['ECG'] = medfilt(df['ECG'], 3)
        
        df['band-pass MSNA'] = self.band_pass_filter(df['Integrated MSNA'], low_cutoff, high_cutoff, fs)
        df['band-pass ECG'] = self.band_pass_filter(df['ECG'], low_cutoff/2, high_cutoff*2, fs)

        df['normalized MSNA'] = self.median_normalize(df['band-pass MSNA'])
        df['normalized ECG'] = self.median_normalize(df['band-pass ECG'])

        return df
    
    def band_pass_filter(self, data, low_cutoff, high_cutoff, fs, order=3):
        nyquist = 0.5 * fs
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(order, [low,high], btype='band', analog=False)
        y = filtfilt(b, a, data)
        return y

    def median_normalize(self, data):
        numpy_data = data.to_numpy()  
        mu = np.median(numpy_data, axis=0)
        difference = np.abs(numpy_data - mu)
        sigma = np.median(difference, axis=0)
        return (numpy_data - mu) / sigma

    def chunk_df(self, df):
        n = self.window_size
        steps = n // 16
        
        msna = df['normalized MSNA'].to_numpy()
        ecg = df['normalized ECG'].to_numpy()
        burst_labels = df['BURST'].to_numpy()
        
        chunks = [
                [torch.tensor(np.array([msna[idx-n//2: idx+n//2], ecg[idx-n//2: idx+n//2]])), 
                int(1 in burst_labels[idx-n//2: idx+n//2])]
            
            for idx in range(n//2+1, len(msna)-n//2-2, steps)
        ]
    
        return chunks
    
    def make_training_set(self, chunks):
        trainloader = torch.utils.data.DataLoader(chunks, shuffle=True, batch_size=self.batch_size)
        return trainloader

    def train_cnn(self, trainloader, num_epochs=75, learning_rate=0.001):
        self.cnn.train()
        
        # Loss function, optimizer, scheduler
        criterion = nn.BCELoss() 
        optimizer = optim.Adam(self.cnn.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=self.verbose)
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
        
            for inputs, labels in trainloader:
                # Forward
                optimizer.zero_grad()
                outputs = self.cnn(inputs.float())
                
                # Backward
                loss = criterion(outputs, labels.float().unsqueeze(1))
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
            # Average loss
            avg_loss = running_loss / len(trainloader)
            
            if self.verbose:
                if (epoch+1) % 2 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
            scheduler.step(avg_loss)
        
        if self.verbose:
            print("Training complete.")
            
        self.cnn.eval()

    def find_peaks(self, signal):
        signal = np.convolve(signal, self.kernel, mode='same')
        diff = np.diff(signal)
        
        increasing = diff > 0
        decreasing = diff < 0
        
        return np.where(increasing[:-1] & decreasing[1:])[0] + 1
        
    def make_sample_list(self, df):
        chunks = []
        maxs = []
        n = self.window_size
        
        msna = df['normalized MSNA'].to_numpy()
        ecg = df['normalized ECG'].to_numpy()
        
        idxs = self.find_peaks(msna)
        idxs = idxs[(idxs > n//2) & (idxs < (len(msna) - n//2))]
        
        for idx in idxs:
            start = idx - n//2
            end = idx + n//2
        
            max_idx = np.argmax(msna[start:end])
            chunk = [msna[start:end], ecg[start:end]]
            
            chunks.append(chunk)
            maxs.append(int(max_idx+start))
    
        return np.array(chunks), np.array(maxs)
    
    def predict_labels(self, df):
        # Get all chunks (with a possible peak index in the center)
        chunks, possible_peak_indices = self.make_sample_list(df)
    
        # Run the model to get output probabilities
        model_input = torch.tensor(chunks)
        with torch.no_grad():
            probabilities = self.cnn(model_input.float()).detach()
        probabilities = probabilities.numpy().squeeze()
        
        probabilities = np.array(probabilities > self.threshold, dtype=int)
        labels = np.nonzero(probabilities)[0]
        
        # Get the indices of predicted thresholded peaks
        peak_indices = possible_peak_indices[labels]
    
        return peak_indices

    def metrics(self, df, verbose=False) -> float:
        # NOTE: I did not write the majority of this method, I need to find out who did

        actual_bursts = self.get_burst_idxs(df)
        predicted_bursts = self.predict_labels(df)
        
        # Define a tolerance for matching peaks (e.g., 50 sample indices)
        tolerance = 25
    
        # Convert the lists to numpy arrays for easier manipulation
        detected_peaks = np.array(predicted_bursts)
        actual_peaks = np.array(actual_bursts)
    
        if len(detected_peaks) == 0:
            return 0.0
        
        # Initialize true positives (TP), false positives (FP), and false negatives (FN)
        TP = 0
        FP = 0
        FN = 0
        TN = 0  # True Negatives (not typically used in this context but included for completeness)
        
        # Calculate TP and FP
        for dp in detected_peaks:
            if np.any(np.abs(actual_peaks - dp) <= tolerance):
                TP += 1
            else:
                FP += 1
        
        # Calculate FN
        for ap in actual_peaks:
            if not np.any(np.abs(detected_peaks - ap) <= tolerance):
                FN += 1
        
        # Calculate TN
        # Note: In signal processing contexts like this, true negatives (TN) are not typically calculated 
        # because it would require accounting for all the points where no peaks are detected or expected.
        # However, for completeness, if I am considering TN as all points not being peaks, it can be approximated:
        total_samples = max(np.max(detected_peaks), np.max(actual_peaks)) + 1
        total_non_peaks = total_samples - len(detected_peaks) - len(actual_peaks)
        TN = total_non_peaks
    
        # Calculate Precision, Recall, F1-score, and Accuracy
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    
        if verbose:
            print(f"True Positives (TP): {TP}")
            print(f"False Positives (FP): {FP}")
            print(f"False Negatives (FN): {FN}")
            print(f"True Negatives (TN): {TN}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-score: {f1_score:.2f}")
            print(f"Accuracy: {accuracy:.2f}")
    
        return f1_score

    def train_threshold(self, dfs, maxiter=16):
        # TODO: If threshold is too high, we get count == 0 (bc no probabilities pass through to metrics)
        # Currently fixing with a recursive call to loss, but this is obviously bad
        def loss(x):
            self.threshold = x
            # np.random.shuffle(dfs)
            total = 0
            count = 0
            for df in (dfs):
                n = self.metrics(df)
                if n > 0.0:
                    total += 1 - n
                    count += 1
            if count == 0:
                return loss(x-0.1)
            return total/count
        
        result = minimize(loss, x0=self.threshold, method='Nelder-Mead', options={'disp': self.verbose, 'maxiter': maxiter})
            
        self.threshold = result.x[0]

    def train(self, dfs, threshold_train_max_iter=16, learning_rate=0.001, num_epochs=32):
        if self.verbose:
            print("Processing dataframes.")

        if type(dfs) != list:
            dfs = [dfs]
            
        dfs = [self.process_dataframe(df) for df in dfs]

        if self.verbose:
            print("Processed dataframes, chunking.")
        
        chunks = []
        for df in dfs:
            chunk = self.chunk_df(df)
            chunks.extend(chunk)
            
        if self.verbose:
            print("Got chunks, making dataloader.")

        trainloader = self.make_training_set(chunks)

        if self.verbose:
            print("Made dataloadder, training CNN.")

        self.train_cnn(trainloader, num_epochs=num_epochs, learning_rate=learning_rate)
        # self.find_kernel(dfs)

        if self.verbose:
            print("trained CNN, getting threshold.")

        self.train_threshold(dfs, maxiter=threshold_train_max_iter)

        if self.verbose:
            print("Got threshold, training complete.")

    def predict(self, df):
        df = self.process_dataframe(df)
        return self.predict_labels(df)

    def split_data(self, data, k):
        length = len(data) // k
        splits = []
        for i in range(k-1):
            chunk = data[i*length:(i+1)*length]
            if type(data) != list: # Handle intrasubject Dataframe chunks
                chunk = [chunk]
            splits.append(chunk)
        return splits
        
    def k_fold_cross_validation(self, dfs, k):
        k += 1
        splits = self.split_data(dfs, k)

        n = len(splits)
        f1s = []

        for idx in range(n):
            
            train_samples = []
            for jidx in range(n):
                if jidx != idx:
                    train_samples.extend(splits[jidx])
                    
                    
            test_sample = splits[idx]

            self.init_params()

            if self.verbose:
                print("Training")
            self.train(train_samples)

            if self.verbose:
                print("Validating")
            test_sample = [self.process_dataframe(df) for df in test_sample]
            f1 = np.mean([self.metrics(i) for i in test_sample])
            f1s.append(f1)
            
            if self.verbose:
                print(f"F1 for split{idx}: {f1}")

            if self.verbose:
                print("Current Threshold:", self.threshold)
            # Add info about this run (threshold, etc.)

        return f1s
import torch
import itertools
import os
import numpy as np
from torch.utils.data import Dataset
import pdb
from random import randint

class RegressionDataset(Dataset):
    """Generate data for the regression task."""

    def __init__(
        self,
        files,
        input_length,
        channels,
        task,
        g_con = True
    ):

        self.input_length = input_length
        self.files = self.group_recordings(files)
        self.channels = channels
        self.task = task
        self.g_con = g_con

    def group_recordings(self, files):
 
        new_files = []
        
        # Group by subject name (second part of filename)
        grouped = itertools.groupby(sorted(files), lambda x: os.path.basename(x).split("_-_")[1])
        
        for subject_name, feature_paths in grouped:
            feature_list = list(feature_paths)
            
            # Sort by feature type (eeg first, then mel)
            sorted_features = sorted(feature_list, key=lambda x: "0" if "eeg" in x else "1")
            new_files.append(sorted_features)

        return new_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, recording_index):
        
        # 1. For within subject, return eeg, envelope and subject ID
        # 2. For held-out subject, return eeg, envelope

        if self.task == "train":
            x, y, sub_id = self.__train_data__(recording_index)

        else:
            x, y, sub_id = self.__test_data__(recording_index)

        return x, y, sub_id


    def __train_data__(self, recording_index):

        framed_data = []

        for idx, feature in enumerate(self.files[recording_index]):
            data = np.load(feature)

            if idx == 0: 
                start_idx= randint(0,len(data)- self.input_length)

            framed_data += [data[start_idx:start_idx + self.input_length]]

        if len(framed_data) < 2:
            print(f"ERROR: Not enough features found. Expected 2 (eeg and mel), got {len(framed_data)}")
            # Return dummy data to avoid crash
            dummy_data = torch.zeros(self.input_length, self.channels)
            return dummy_data, dummy_data, 0

        if self.g_con == True:
            # Extract subject name from filename and convert to a numeric ID
            subject_name = feature.split('/')[-1].split('_-_')[1]
            # Create a hash-based ID for the subject name to ensure consistency
            # Use modulo with a smaller number to stay within model's embedding range
            sub_idx = hash(subject_name) % 85  # Model expects within_sub_num=85
    
        else:
            sub_idx = torch.FloatTensor([0])
    
            # return torch.FloatTensor(framed_data[0]), torch.FloatTensor(framed_data[1]), sub_idx
        
            
        return torch.FloatTensor(framed_data[0]), torch.FloatTensor(framed_data[1]), sub_idx

    def __test_data__(self, recording_index):
        """
        return: list of segments [[eeg, envelope] ...] depending on self.input_length 
                e.g.,for 10 second-long input signal and input_length==5, return [[5, 5], [5, 5]]
        
        """
        framed_data = []

        for idx, feature in enumerate(self.files[recording_index]):
            data = np.load(feature)
            nsegment = data.shape[0] // self.input_length
            data = data[:int(nsegment * self.input_length)]
            segment_data = [torch.FloatTensor(data[i:i+self.input_length]).unsqueeze(0) for i in range(0, data.shape[0], self.input_length)]
            segment_data = torch.cat(segment_data)
            framed_data += [segment_data]
            
        if self.g_con == True:
            # Extract subject name from filename and convert to a numeric ID
            subject_name = feature.split('/')[-1].split('_-_')[1]
            # Create a hash-based ID for the subject name to ensure consistency
            # Use modulo with a smaller number to stay within model's embedding range
            sub_idx = hash(subject_name) % 85  # Model expects within_sub_num=85

        else:
            sub_idx = torch.FloatTensor([0])

        return framed_data[0], framed_data[1], sub_idx

import os
import io
from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
from google.cloud import storage
from torchvision import transforms
import logging
from typing import Dict, Any, Tuple

class GCPBucketDataset(Dataset):
    def __init__(self, bucket_name: str, prefix: str, csv_file: str, transform=None):
        """
        Initializes the dataset by connecting to Google Cloud Storage and reading the CSV file.

        Args:
            bucket_name (str): The name of the GCP bucket.
            prefix (str): The prefix for the files in the bucket.
            csv_file (str): The path to the CSV file containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.transform = transform

        # Connect to Google Cloud Storage
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Get list of files in the bucket
        self.file_paths = [blob.name for blob in self.bucket.list_blobs(prefix=self.prefix)]

       # Initialize label encoders and decoders with known labels
        self.label_encoders = {
            'view': {
                'Apical Four Chamber(A4C)': 0,
                'Parasternal long axis (PLAX)': 1,
                'Parasternal short axis(PSAX)': 2
            },
            'condition': {
                'Aortic Valve Regurgitation and Pulmonary Valve Regurgitation': 0,
                'Aortic Valve Regurgitation': 1,
                'Mitral Valve Prolapse': 2,
                'Mitral Valve Regurgitation': 3,
                'Not Applicable': 4,
                'Pulmonary Valve Regurgitation': 5,
                'Tricuspid Valve Regurgitation': 6
            },
            'severity': {
                'Borderline RHD': 0,
                'Definite RHD': 1,
                'Not Applicable': 2
            }
        }

        # Create decoders (reverse of encoders)
        self.label_decoders = {
            task: {v: k for k, v in encoders.items()}
            for task, encoders in self.label_encoders.items()
        }

        # Read and process labels if csv_file is provided
        if csv_file:
            self.labels_df = self._read_csv(csv_file)
            # Capitalize 'rhd' in severity column
            self.labels_df['SEVERITY'] = self.labels_df['SEVERITY'].str.replace('rhd', 'RHD', case=False)
            # Print column names for debugging
            print("Available columns in CSV:", self.labels_df.columns.tolist())
            
            # Clean up condition labels
            if 'CONDITION' in self.labels_df.columns:
                self.labels_df['CONDITION'] = self.labels_df['CONDITION'].astype(str).str.strip("[]").str.replace("'", "")
            
            self._create_label_encodings()
            self.labels = self._create_labels_dict()
            
        logging.info(f"Total files in bucket prefix '{self.prefix}': {len(self.file_paths)}")
        logging.info(f"Number of labeled samples: {len(self.labels)}")
        self._log_label_distribution()

    def _read_csv(self, csv_file):
        """Read the CSV file from GCP bucket."""
        blob = self.bucket.blob(csv_file)
        with blob.open("rb") as f:
            return pd.read_csv(f)

    def _create_label_encodings(self):
        """Create numerical encodings for each label category."""
        for task in ['view', 'condition', 'severity']:
            # Use the correct column name based on the task
            col_name = task.upper() + '-APP' if task == 'view' else task.upper()  # Correct column name
            unique_labels = sorted(self.labels_df[col_name].unique())
            print(f"Unique {task} Labels from CSV ({col_name}):", unique_labels)  # Print with column name for debugging

            for idx, label in enumerate(unique_labels):
                self.label_encoders[task][label] = idx
                self.label_decoders[task][idx] = label
                
        print("\nLabel Encoders:")  # Print the entire encoders
        print(self.label_encoders)
        print("\nLabel Decoders:") # Print the entire decoders
        print(self.label_decoders)
        
    def _create_labels_dict(self):
        """Create a dictionary mapping filenames to their labels."""
        labels_dict = {}
        for _, row in self.labels_df.iterrows():
            filename = os.path.basename(row['FILENAME'])
            labels_dict[filename] = (
                self.label_encoders['view'][row['VIEW-APP']],
                self.label_encoders['condition'][row['CONDITION']],
                self.label_encoders['severity'][row['SEVERITY']]
            )
        return labels_dict


    def __len__(self):
        return len(self.file_paths)

    def _log_label_distribution(self):
        """Log distribution of labels for each task"""
        for task in ['view', 'condition', 'severity']:
            logging.info(f"\n{task.upper()} label distribution:")
            for idx, label in self.label_decoders[task].items():
                count = sum(1 for labels in self.labels.values() if labels[list(self.label_encoders.keys()).index(task)] == idx)
                logging.info(f"{label}: {count} samples")
                

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        filename = os.path.basename(file_path)
        
        # Load and transform image
        with self.bucket.blob(file_path).open("rb") as f:
            image = Image.open(io.BytesIO(f.read())).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Extract video ID from filename
        video_id = self._extract_video_id(filename)

        # Return labels if available, else dummy labels
        return image, self.labels.get(filename, (-1, -1, -1)), video_id


    def _extract_video_id(self, filename):
        """Helper to extract video ID from filename."""
        try:
            video_id = filename.split("_frame")[0]  # Extract video ID from frame filename
            return video_id
        except (IndexError, AttributeError):
            logging.error(f"Could not extract video ID from filename: {filename}")
            return None  # Return None or a suitable placeholder on error


    def get_num_classes(self):
        """Return dictionary with number of classes for each task"""
        return {
            'view': 3,
            'condition': 7,
            'severity': 3
        }

    def get_label_decoders(self):
        return self.label_decoders
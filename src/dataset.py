import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from text_to_tensor import text_to_tensor
from mfcc_pipeline import extract_mfcc_pipeline
import vocab





class NepaliAudioDataset(Dataset):
    def __init__(self, tsv_filepath, audio_base_path, audio_extension=".flac"):

        try:
            self.data = pd.read_csv(tsv_filepath, sep='\t', header=None)
            self.data.columns = ['file_id', 'transcription']

            # ✅ Add filtering step here
            allowed_folders = {
                "00", "0a", "0b", "0c", "0d", "0e",
                "0f", "02", "03", "04", "05", "06", "01",#"07", "08", "09"
            }

            # Only keep rows where the file_id's folder is in the allowed list
            self.data = self.data[self.data['file_id'].str[:2].isin(allowed_folders)].reset_index(drop=True)

            #for checking
            self.data = self.data.head(4000)

            print(f"Filtered dataset: {len(self.data)} entries in allowed folders.")

        except FileNotFoundError:
            print(f"Error:tsv file not found '{tsv_filepath}'. check file path!")
            self.data = pd.DataFrame()
        except Exception as e:
            print(f"error occured during reading dataset: {e}")
            self.data = pd.DataFrame()




        self.audio_base_path = audio_base_path
        self.audio_extension = audio_extension

        self.mfccs = []
        self.labels = []
        self.input_lengths = []
        self.label_lengths = []

        print(f"Processing {len(self.data)} entries from TSV...")
        for idx, row in self.data.iterrows():
          if idx % 100 == 0:
            print(f"Processed {idx}/{len(self.data)} files...")


          file_id = str(row['file_id']).strip()
          label_text = str(row['transcription']).strip() #splitting the transcription into elementary pieces.

            # --- KEY CHANGE: Constructing the path based on first two characters ---
            # Get the first two characters of the file_id for the subdirectory name
          if len(file_id) < 2:
              print(f"Skipping {file_id}: File ID is too short to determine subdirectory.")
              continue

          subdirectory = file_id[:2]

            # Construct the full path: base_path / subdirectory / file_id.flac
          full_audio_path = os.path.join(self.audio_base_path, subdirectory, f"{file_id}{self.audio_extension}")
            # --- END KEY CHANGE ---

          if os.path.exists(full_audio_path):
              try:
                  mfcc = extract_mfcc_pipeline(full_audio_path)

                  if mfcc is None:
                      print(f"Skipping {full_audio_path}: MFCC extraction failed or returned None.")
                      continue

                  if not isinstance(mfcc, np.ndarray):
                      mfcc = np.array(mfcc)

                  mfcc = (mfcc - mfcc.mean()) / (mfcc.std()) #normalizing mfcc

                  if not isinstance(mfcc, torch.Tensor):
                      mfcc = torch.tensor(mfcc, dtype=torch.float32)
                  else:
                      mfcc = mfcc.detach().float()

                  label=text_to_tensor(label_text,char2idx)

                  self.mfccs.append(mfcc)
                  self.labels.append(label)
                  self.input_lengths.append(mfcc.shape[0])
                  self.label_lengths.append(len(label))

              except Exception as e:
                  print(f"process failed! {full_audio_path}: {e}")
          else:
              print(f"File not found: {full_audio_path}")

        print(f"Finished processing. Loaded {len(self.mfccs)} valid audio entries.")

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, idx):
        return self.mfccs[idx], self.labels[idx], self.input_lengths[idx], self.label_lengths[idx]


import vocab





# Hyperparameters
input_dim = 13   # MFCC features (13 for each time step)
hidden_dim = 256  # Hidden dimension for LSTM

output_dim=len(vocab.NEPALI_CHARS)
#print("output:",output_dim)
num_layers = 2  # Number of LSTM layers
batch_size = 32
num_epochs = 100
tsv_path = "/content/drive/MyDrive/openslrds/updated_df.tsv"
audio_dir = "/content/drive/MyDrive/openslrds/data"
checkpoint_dir = "/content/drive/MyDrive/openslrds/checkpoints"

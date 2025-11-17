from src.dataset import NepaliAudioDataset
from torch.utils.data import Dataset, DataLoader, random_split
from src.collate import nepali_collate_fn
from src.param import batch_size


tsv_path = "/content/drive/MyDrive/openslrds/updated_df.tsv"
audio_dir = "/content/drive/MyDrive/openslrds/data"


# Defining the splitting ratio
train_ratio=0.8
test_ratio=0.2

dataset = NepaliAudioDataset(tsv_path, audio_dir)


#Calculating the sizes for each split
dataset_size=len(dataset)
train_size=int(train_ratio*dataset_size)
test_size=int(dataset_size-train_size)

#Performing the split
train_dataset, test_dataset=random_split(dataset,[train_size,test_size])

#Creating dataloader
train_loader=DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=nepali_collate_fn,
    num_workers=2
)

test_loader=DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=nepali_collate_fn,
    num_workers=2)




if __name__=="__main__":

    # Verifing the splits
    print(f'Total dataset size: {dataset_size}')
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(test_dataset)}')
    print()
    print()

    # check after padding:
    for batch in train_loader:
        mfccs_padded, labels_concat, input_lengths, label_lengths = batch
        print(f"MFCC batch shape: {mfccs_padded.shape}")
        print(f"Labels concat length: {labels_concat.shape}")
        print(f"Input lengths: {input_lengths}")
        print(f"Label lengths: {label_lengths}")


        break

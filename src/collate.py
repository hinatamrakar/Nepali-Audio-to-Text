import torch
from torch.nn.utils.rnn import pad_sequence


def nepali_collate_fn(batch):#batch:- list of tupples,[(samp1),(samp2),(samp3)]

  # Eg. batch:- [(mfcc_1, label_1, input_length_1, label_length_1), (mfcc_2, label_2, input_length_2, label_length_2)]

    """Pad the MFCCs and concatenate the labels for batching."""
    mfccs, labels, input_lengths, label_lengths = zip(*batch) # separate arguments and aggrgates elements in each column

    # max_input_length = max(input_lengths)
    # input_dim=mfccs[0].shape[1]
    # # print("max mfcc leng",max_mfcc_length)
    # padded_mfccs = torch.stack([torch.cat([mfcc, torch.zeros(max_input_length - mfcc.shape[0], input_dim)], dim=0)
    #                             for mfcc in mfccs])
    padded_mfccs = pad_sequence(mfccs, batch_first=True, padding_value=0.0)  # [B, T, D]

    # # Concatenate labels into a single tensor (assuming labels are lists of integers)
    #max_label_length = max(label_lengths)

    flattened_labels = torch.cat(labels)


    # Convert input_lengths and label_lengths to tensors
    input_lengths = torch.tensor(input_lengths,dtype=torch.long)
    label_lengths = torch.tensor(label_lengths,dtype=torch.long)



    return padded_mfccs, flattened_labels, input_lengths, label_lengths

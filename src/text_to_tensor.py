from import_module import torch

def text_to_tensor(label_text, char2idx, unk_idx=1):
    """
    Converts a label string into a tensor of character indices.
    Unknown characters are mapped to `unk_idx`.
    CTC blank '-' is never added to the label.
    """
    return torch.tensor(
        [char2idx.get(c, unk_idx) for c in label_text if c != '-'],
        dtype=torch.long
    )

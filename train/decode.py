def decode_prediction(pred, idx2char):
    decoded = []
    prev = None
    for k in pred:
        k = k.item()  # Convert tensor to int
        if k != prev and k != char2idx['-']:
            if k in idx2char:
                decoded.append(idx2char[k])
        prev = k
    return ''.join(decoded)

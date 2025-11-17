import editdistance  # If you're offline. Install via: pip install editdistance

def calculate_cer(predictions, references):
    total_chars = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        total_errors += editdistance.eval(pred, ref)
        total_chars += len(ref)

    if total_chars == 0:
        return 0.0
    return total_errors / total_chars

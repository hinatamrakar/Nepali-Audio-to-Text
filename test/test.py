model.eval()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
def ctc_greedy_decoder(output, idx2char):
    # output shape: (batch_size, seq_len, vocab_size)
    # take argmax over vocab dimension
    argmax_preds = output.argmax(dim=2)  # (batch, seq_len)
    results = []
    for pred in argmax_preds:
        # collapse repeating characters and remove blanks
        prev = None
        sentence = []
        for p in pred:
            if p.item() != prev and idx2char[p.item()] != '-':
                sentence.append(idx2char[p.item()])
            prev = p.item()
        results.append(''.join(sentence))
    return results




all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        inputs, targets, input_lengths, target_lengths = batch
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)  # shape: (batch, seq_len, vocab_size)

        # Decode
        decoded_texts = ctc_greedy_decoder(outputs.cpu(), idx2char)
        all_preds.extend(decoded_texts)

        # Optional: for comparison
        for target in targets:
            true_text = ''.join([idx2char[idx.item()] for idx in target if idx.item() != char2idx['-']])
            all_targets.append(true_text)



for pred, target in zip(all_preds, all_targets):
    print(f'Prediction: {pred}')
    print(f'Ground Truth: {target}')
    print('---')

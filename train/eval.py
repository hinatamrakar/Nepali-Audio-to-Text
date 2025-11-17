def model_eval(model, data_loader,ctc_loss,device):
    model.eval()
    total_loss = 0
    predictions = []
    references=[]

    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for batch in tk:
            # Unpack based on what your nepali_collate_fn returns
            inputs, targets, input_lengths, target_lengths = batch

            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            output = model(inputs)  # (B, T, C)

            log_probs = output.permute(1, 0, 2)  # (T, B, C)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            # decode predictions
            decoded = log_probs.argmax(2).permute(1, 0)  # (B, T)
            predictions.extend(decoded.cpu().numpy())

            #for CER
            i = 0
            for length in target_lengths:
                label_seq = targets[i:i+length]
                references.append(decode_prediction(label_seq, idx2char))
                i += length

    avg_loss = total_loss / len(data_loader)
    return predictions, avg_loss,references

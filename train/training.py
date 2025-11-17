








os.makedirs(checkpoint_dir, exist_ok=True)
csv_file_path = os.path.join(checkpoint_dir, "losses_and_cers.csv")
epoch_loss_file = os.path.join(checkpoint_dir, "epoch_losses.csv")

#data loader
train_dataset = NepaliAudioDataset(tsv_filepath=tsv_path, audio_base_path=audio_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=nepali_collate_fn)
model.to(device)
#resuming from checkpoint
resume_from = os.path.join(checkpoint_dir, "checkpoint_latest.pt")  # ← change as needed
start_epoch = 0

if os.path.exists(resume_from):
    checkpoint = torch.load(resume_from, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed from checkpoint: {resume_from}, starting at epoch {start_epoch + 1}")

#main train toop
train_losses=[]
valid_losses=[]
cers=[]
for epoch in range(start_epoch, num_epochs):
    print(f"\n Epoch {epoch + 1}/{num_epochs}")
    train_loss,batch_losses = train(model, train_loader, optimizer, ctc_loss, device)
    print(f"Average Loss: {train_loss:.4f}")
    # scheduler.step(avg_loss)
    valid_preds,valid_loss,valid_refs=model_eval(model,test_loader,ctc_loss,device)
    valid_text_preds=[]
    for vp in valid_preds:
      current_preds=decode_prediction(vp,idx2char)
      valid_text_preds.append(current_preds)

    #for Cer
    cer = calculate_cer(valid_text_preds, valid_refs)

    pprint(list(zip(valid_text_preds))[6:11])
    print(f"Epoch:{epoch},train_loss{train_loss},valid_loss:{valid_loss},CER:{cer:.4f}")
    scheduler.step(valid_loss)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    cers.append(cer)

    with open(epoch_loss_file, mode='a', newline='') as f:
      writer = csv.writer(f)
      for i, loss in enumerate(batch_losses):
          writer.writerow([epoch + 1, i, loss])  # Epoch, Batch index, Loss


    with open(csv_file_path, mode='a', newline='') as f:
      writer = csv.writer(f)
      writer.writerow([epoch + 1, train_loss, valid_loss, cer])



    # Save checkpoint
    # checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    # latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")

    # torch.save({
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': train_loss,
    # }, checkpoint_path)

    # torch.save({
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': train_loss,
    # }, latest_path)

    # print(f"✅ Saved checkpoint: {checkpoint_path}")

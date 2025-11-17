checkpoint_path = os.path.join(checkpoint_dir, 'lstm_ctc_checkpoint.pth')
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'valid_loss': valid_loss,
    'char2idx': char2idx,
    'idx2char': idx2char,
    'config': {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_layers': num_layers,
        'dropout': dropout
    }
}

torch.save(checkpoint, checkpoint_pth)

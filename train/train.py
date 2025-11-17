import torch.optim as optim
from model import LSTMCTCModel
from param import input_dim, hidden_dim, ouput_dim, num_layers
import torch.nn as nn
import src.vocab
import torch
from src.vocab import char2idx


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Instantiate the model
asrmodel = LSTMCTCModel(input_dim, hidden_dim, output_dim, num_layers)
model =asrmodel.to(device)
# Define CTC loss
ctc_loss = nn.CTCLoss(blank=char2idx['-'], zero_infinity=True)
#optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)
print(model)


def train(model, train_loader, optimizer, ctc_loss, device):
    model.train()
    total_loss = 0
    batch_losses=[]

    for batch_idx, (mfcc, labels, input_len, label_len) in enumerate(train_loader):
        mfcc = mfcc.to(device)
        labels = labels.to(device)
        input_len = input_len.to(device)
        label_len = label_len.to(device)

        output = model(mfcc)  # (batch, seq_len, vocab)/log_probs
        output = output.permute(1, 0, 2)



        loss = ctc_loss(output, labels, input_len, label_len)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()


        # total_loss += loss.item()
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader),batch_losses





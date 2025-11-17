import torch.nn as nn
import torch



class LSTMCTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3,intermediate_dim=256,intermediate_dim2=128,dropout=0.15):
        super(LSTMCTCModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)


        #for debugging
         # Intermediate hidden layer
        self.fc1 = nn.Linear(hidden_dim * 2, intermediate_dim)
        self.activation = nn.ReLU()
        self.fc2=nn.Linear(intermediate_dim,intermediate_dim2)
        self.dropout = nn.Dropout(dropout)

        self.bn1=nn.BatchNorm1d(intermediate_dim)
        self.ln1=nn.LayerNorm(hidden_dim*2)


        # Final output layer to vocab
        self.fc = nn.Linear(intermediate_dim2, output_dim)


    def forward(self, x):

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        #lstm_out,_=self.lstm(x)
        lstm_out=self.ln1(lstm_out)

        # Hidden layer
        hidden = self.fc1(lstm_out)  # (batch_size, seq_len, intermediate_dim)
        #print(hidden.size())
        hidden=self.bn1(hidden.transpose(1,2)).transpose(1,2)
        #hidden = self.fc2(lstm_out)  # (batch_size, seq_len, intermediate_dim)
        #print(hidden.size())
        hidden = self.activation(hidden)
        #print(hidden.size())
        hidden = self.fc2(hidden)
        #print(hidden.size())
        hidden = self.activation(hidden)
        #print(hidden.size())
        # hidden = self.dropout(hidden)
        #print(hidden.size())

        # Output layer
        output = self.fc(hidden)  # (batch_size, seq_len, output_dim)
        #print(output.size())
        output=torch.nn.functional.log_softmax(output,dim=2) #(batch, seq_len/time, output_dim)
        # CTC loss requires the output to be of shape (seq_len, batch_size, vocab_size)
        #output = output.permute(1, 0, 2)  # (seq_len, batch, vocab) => T/seq_len=max(input_len)
        # Log softmax for CTC
        return output

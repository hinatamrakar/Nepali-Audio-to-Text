import pandas as pd
import matplotlib.pyplot as plt

# Define column names manually
column_names = ['Epoch', 'Train Loss', 'Valid Loss', 'CER']

# Load CSV without header
csv_path = 'path/to/losses_and_cers.csv'  # 🔁 Replace with your file path
df = pd.read_csv(csv_file_path, header=None, names=column_names)

# Plot Loss vs Epoch
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
plt.plot(df['Epoch'], df['Valid Loss'], label='Validation Loss', marker='s')

plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Optional: Save plot
plt.savefig('loss_vs_epoch.png')
plt.show()

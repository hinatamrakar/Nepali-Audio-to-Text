import torch

NEPALI_CHARS = [
    ' ', 'ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ',
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह','१','२','३','४','५','६','७','८','९', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', '्',
    '\u200c', '\u200d', '।'
]


# Add padding, unknown, and blank tokens
NEPALI_CHARS = ['u'] + NEPALI_CHARS + ['-']

NUM_NEPALI_CHARS = len(NEPALI_CHARS)  # Vocabulary size
#print(len(NEPALI_CHARS))

char2idx = {char: idx for idx, char in enumerate(NEPALI_CHARS)}
idx2char = {idx: char for char, idx in char2idx.items()}

# char2idx['-'] < NUM_NEPALI_CHARS
# print(char2idx['-'])
# print(len(NEPALI_CHARS))
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

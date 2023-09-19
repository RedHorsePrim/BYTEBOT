import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from aimindmodel import NeuralNet
from tqdm import tqdm
from torch.optim import lr_scheduler

# Constants
DATA_FILE = 'intents.json'
MODEL_FILE = 'data.pth'
NUM_EPOCHS = 10000
BATCH_SIZE = 64
LEARNING_RATE = 0.01
HIDDEN_SIZE = 32

# Load data from intents.json
with open(DATA_FILE, 'r') as json_data:
    intents = json.load(json_data)

# Data preprocessing
all_words = set()
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern.lower())  # Tokenize and lowercase
        all_words.update(words)
        xy.append((words, tag))

all_words = sorted(all_words)
tags = sorted(set(tags))

# Split data into training and testing
from sklearn.model_selection import train_test_split

train_xy, test_xy = train_test_split(xy, test_size=0.2, random_state=42)

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in train_xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Create DataLoader for training data
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(X_train, dtype=torch.float32)
        self.y_data = torch.tensor(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create DataLoader for testing data
X_test = []
y_test = []

for (pattern_sentence, tag) in test_xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_test.append(bag)
    y_test.append(tags.index(tag))

X_test = np.array(X_test)
y_test = np.array(y_test)

class TestDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_test)
        self.x_data = torch.tensor(X_test, dtype=torch.float32)
        self.y_data = torch.tensor(y_test)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

test_dataset = TestDataset()
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define and train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(len(all_words), HIDDEN_SIZE, len(tags)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define the learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Create a progress bar
progress_bar = tqdm(total=NUM_EPOCHS, desc='Training Progress', position=0, leave=True)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(words)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}')

    # Update the learning rate
    scheduler.step()

    # Update the progress bar
    progress_bar.update(1)

# Close the progress bar after completion
progress_bar.close()

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the test dataset
test_loss = 0.0
correct_predictions = 0

with torch.no_grad():
    for (test_words, test_labels) in test_loader:
        test_words = test_words.to(device)
        test_labels = test_labels.to(dtype=torch.long).to(device)

        test_outputs = model(test_words)
        test_loss += criterion(test_outputs, test_labels).item()

        _, predicted = torch.max(test_outputs, 1)
        correct_predictions += (predicted == test_labels).sum().item()

# Calculate the average test loss and accuracy
average_test_loss = test_loss / len(test_loader)
accuracy = correct_predictions / len(test_loader.dataset)

print(f'Validation Loss: {average_test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

# Set the model back to training mode
model.train()

# Save the trained model and related data
data = {
    "model_state": model.state_dict(),
    "input_size": len(all_words),
    "hidden_size": HIDDEN_SIZE,
    "output_size": len(tags),
    "all_words": list(all_words),
    "tags": tags
}

torch.save(data, MODEL_FILE)
print(f'Training complete. Model saved to {MODEL_FILE}')

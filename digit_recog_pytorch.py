from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from helper_functions import accuracy_fn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from torch import nn
import numpy as np
import random
import torch
import os


class TrainDataset(Dataset):
    def __init__(self, img, labels, transform=None):
        self.labels = torch.LongTensor(labels)
        self.img = img.reshape(-1, 1, 28, 28)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item]
        img = self.img[item]
        if self.transform:
            img = Image.fromarray(self.img[item].astype(np.uint8))
            img = self.transform(img)
        return img, label


data_transform = transforms.Compose([transforms.ToTensor()])

file_path = r'datasets/digit-recognizer'
train_path = os.path.join(file_path, 'train.csv')
test_path = os.path.join(file_path, 'test.csv')


def csv2tensor(path):
    to_np = np.loadtxt(path, dtype=np.float32, delimiter=',', skiprows=1)
    labels = to_np[:, 0]
    to_np = to_np[:, 1:]
    to_tensor = torch.from_numpy(to_np)
    return to_tensor, labels


# Visualization
def visual(train_data, train_labels):
    fig = plt.figure(figsize=(8, 8))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        idx = random.randint(0, len(train_labels))
        image, label = train_data[idx], train_labels[idx]
        image = image.reshape(28, 28)
        fig.add_subplot(rows, cols, i)
        plt.imshow(image, cmap='binary')
        plt.title(int(label))
        plt.axis(False)
    plt.show()
    plt.close()


all_tensor, all_label = csv2tensor(train_path)
train_tensor, test_tensor, train_label, test_label = train_test_split(all_tensor, all_label)

train_ds = TrainDataset(train_tensor, train_label)
test_ds = TrainDataset(test_tensor, test_label)

visual(train_tensor, train_label)

# Divide into batches
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Create train functions
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optim: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        if batch % 200 == 0:
            print(batch)
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optim.zero_grad()
        loss.backward()
        optim.step()

    # Calculate loss and accuracy per epoch
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


# Create a CNN model
class CNNModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1),
            nn.Sigmoid(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


model = CNNModel(1, 10, 10)
# Setup hyper parameters
optim = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

# Training and testing
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n-------')
    train_step(model, train_dataloader, loss_fn, optim, accuracy_fn, device)
    test_step(model, test_dataloader, loss_fn, accuracy_fn, device)

print("Model parameter", model.parameters())
print("model state dict", model.state_dict())


# Save model
MODEL_PATH = Path('digit_recognizer_models')
MODEL_NAME = 'digit_recognizer.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print("Saving model")
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

test_np = np.loadtxt(test_path, dtype=np.float32, delimiter=',', skiprows=1)
test_tensor = torch.from_numpy(test_np)
test_ds = test_tensor.reshape(-1, 1, 28, 28)
test_data = DataLoader(test_ds, BATCH_SIZE, shuffle=False)


model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
model.eval()
y_pred = []
with torch.inference_mode():
    with open('answers.csv', 'w') as f:
        f.write('ImageId,Label\n')
        for j, X in enumerate(test_data):
            y_logits = model(X)
            pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            for i in range(len(pred)):
                f.write(str((j*32)+(i+1))+','+str(int(pred[i]))+'\n')
                print((j*32)+(i+1), '| prediction', int(pred[i]))

print(pred)

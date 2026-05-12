import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

# my modules
from model import DeepFM
from dataset import CTRDataset


# ============================================
# Device
# ============================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================
# dataset and dataloader
# ============================================
dataset = CTRDataset()

dataloader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle = True
)

# ============================================
# Model
# ============================================
model = DeepFM(
    field_dims = [100, 100, 100],
    embedding_dim = 8
).to(device)

criterion = nn.BCEWithLogitsLoss()  # already have sigmoid in the the loss function

optimizer = optim.Adam(
    model.parameters(),
    lr = 0.0005
)

# ============================================
# Training loop
# ============================================
num_epochs = 200

for epoch in range(num_epochs):
    # tell the model is in training mode
    # so the model will calculate the gradient and update the parameters
    model.train()

    total_loss = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        # forward pass
        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# ============================================
# Inference
# ============================================
model.eval()

test = torch.tensor(
    [
        [1, 20, 3],
        [2, 30, 4],
        [20, 3, 15]
    ]
)

with torch.no_grad():
    test = test.to(device)
    output = model(test)
    # because the output is the logit(original score), we need to add sigmoid to get the probability
    # bcewithlogitsloss already have sigmoid in the loss function
    prob = torch.sigmoid(output)

print("\nPrediction Score:")
print(prob.tolist())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import numpy as np
import os
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_dir = './data/lfw'
batch_size = 32
num_workers = 0 if os.name == 'nt' else 4

transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    fixed_image_standardization
])

lfw_dataset = datasets.LFWPeople(
    root=data_dir,
    split='train',
    download=True,
    transform=transform
)

train_loader = DataLoader(
    lfw_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

model = InceptionResnetV1(
    pretrained='vggface2',
    classify=False,
    num_classes=None
).to(device)

for param in model.parameters():
    param.requires_grad = False

for param in model.last_linear.parameters():
    param.requires_grad = True
for param in model.last_bn.parameters():
    param.requires_grad = True


def create_triplets(embeddings, labels, num_triplets=10):
    anchors = []
    positives = []
    negatives = []

    unique_labels = torch.unique(labels)

    for label in unique_labels:
        pos_indices = torch.where(labels == label)[0]
        neg_indices = torch.where(labels != label)[0]

        if len(pos_indices) < 2 or len(neg_indices) < 1:
            continue

        for _ in range(num_triplets):
            anchor_idx, positive_idx = np.random.choice(pos_indices.cpu(), 2, replace=False)

            negative_idx = np.random.choice(neg_indices.cpu(), 1)[0]

            anchors.append(embeddings[anchor_idx])
            positives.append(embeddings[positive_idx])
            negatives.append(embeddings[negative_idx])

    if len(anchors) == 0:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            anchors, positives, negatives = create_triplets(embeddings, labels)

            if anchors is None:
                continue

            optimizer.zero_grad()
            loss = criterion(anchors, positives, negatives)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1)})

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

    return model


print("Starting training...")
trained_model = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10
)

torch.save(trained_model.state_dict(), 'lfw_face_embedding_model.pth')
print("Model saved to 'lfw_face_embedding_model.pth'")
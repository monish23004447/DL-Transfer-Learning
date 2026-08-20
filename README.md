# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Problem Statement:
Develop an image classification system using transfer learning with the VGG19 convolutional neural network to automatically classify chip images into two categories: defect and notdefect. The pretrained VGG19 model is used to extract useful visual features from the images, while its final fully connected layer is modified to perform binary classification. The model is trained and evaluated using training and testing images, and its performance is measured using loss, accuracy, confusion matrix, and classification report.

Dataset:
The dataset contains chip images divided into two classes: defect and notdefect. The images are organized into separate training and testing folders. The training data is used to train the modified VGG19 model, while the testing data is used to evaluate its classification performance.


## Neural Network Model
<img width="1145" height="908" alt="image" src="https://github.com/user-attachments/assets/373e1881-1615-4bde-9761-014d2377e94e" />


## DESIGN STEPS
### STEP 1: 

Load and preprocess the image dataset and resize the images to 224 × 224 pixels.

### STEP 2: 

Load the pretrained VGG19 model with ImageNet weights.

### STEP 3: 

Freeze the pretrained layers to preserve the learned features.

### STEP 4: 

Replace the final fully connected layer with a new layer containing two output classes, defect and notdefect.

### STEP 5: 

Define the Cross-Entropy loss function and Adam optimizer and train the model.

### STEP 6: 

Evaluate the trained model using loss curves, confusion matrix, classification report, and new sample prediction.



## PROGRAM

### Name: RAGASUDHA R

### Register Number: 212224230215

```

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report


# ==========================================
# Step 1: Load Dataset
# ==========================================

# Dataset ZIP file
zip_path = "chip_data.zip"
extract_path = "chip_data"

# Extract dataset
if not os.path.exists(extract_path):

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("Dataset extracted successfully!")


# Find train and test folders
def find_folders(path):

    for root, dirs, files in os.walk(path):

        if "train" in dirs and "test" in dirs:

            train_path = os.path.join(root, "train")
            test_path = os.path.join(root, "test")

            return train_path, test_path

    return None, None


train_path, test_path = find_folders(extract_path)


print("Train folder:", train_path)
print("Test folder:", test_path)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load dataset
train_dataset = torchvision.datasets.ImageFolder(
    root=train_path,
    transform=transform
)


test_dataset = torchvision.datasets.ImageFolder(
    root=test_path,
    transform=transform
)


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# ==========================================
# Step 2: Load Pretrained VGG19
# ==========================================

model = models.vgg19(
    weights=models.VGG19_Weights.IMAGENET1K_V1
)


# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False


# Modify final layer
num_classes = len(train_dataset.classes)

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    num_classes
)


# ==========================================
# Move Model to GPU / CPU
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

print("Using Device:", device)


# ==========================================
# Step 3: Loss Function and Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


# ==========================================
# Step 4: Train Model
# ==========================================

def train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=5):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # --------------------------
        # Training
        # --------------------------

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        train_loss = (
            running_loss /
            len(train_loader)
        )

        train_losses.append(train_loss)


        # --------------------------
        # Validation
        # --------------------------

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                validation_loss += loss.item()


        val_loss = (
            validation_loss /
            len(test_loader)
        )

        val_losses.append(val_loss)


        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f}"
        )


    # --------------------------
    # Loss Graph
    # --------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "Training Loss and Validation Loss"
    )

    plt.legend()

    plt.show()


# Train
train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=5
)


# ==========================================
# Step 5: Test Model
# ==========================================

def test_model(
        model,
        test_loader):

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    accuracy = correct / total


    print(
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )


    # --------------------------
    # Confusion Matrix
    # --------------------------

    cm = confusion_matrix(
        all_labels,
        all_preds
    )


    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title(
        "Confusion Matrix"
    )

    plt.show()


    # --------------------------
    # Classification Report
    # --------------------------

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=train_dataset.classes
        )
    )


# Test model
test_model(
    model,
    test_loader
)


# ==========================================
# Step 6: Predict Single Image
# ==========================================

def predict_image(
        model,
        image_index,
        dataset):

    model.eval()

    image, label = dataset[image_index]


    with torch.no_grad():

        image_tensor = (
            image.unsqueeze(0)
            .to(device)
        )

        output = model(
            image_tensor
        )

        _, predicted = torch.max(
            output,
            1
        )

        predicted = predicted.item()


    class_names = dataset.classes


    # Display image
    img = image.permute(
        1, 2, 0
    )


    # Undo normalization for display
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    img = img * std + mean

    img = torch.clamp(
        img,
        0,
        1
    )


    plt.figure(figsize=(4, 4))

    plt.imshow(img)

    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted]}"
    )

    plt.axis("off")

    plt.show()


    print(
        "Actual:",
        class_names[label]
    )

    print(
        "Predicted:",
        class_names[predicted]
    )


# Example prediction
predict_image(
    model,
    image_index=0,
    dataset=test_dataset
)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1118" height="602" alt="Screenshot 2026-08-20 155526" src="https://github.com/user-attachments/assets/509be02b-8656-4acc-b48d-444d452a5c39" />


## Confusion Matrix
<img width="943" height="636" alt="Screenshot 2026-08-20 155508" src="https://github.com/user-attachments/assets/811d75fe-3427-40aa-9ec2-1bdcb197eeb2" />



## Classification Report


<img width="637" height="250" alt="Screenshot 2026-08-20 155050" src="https://github.com/user-attachments/assets/dd651af9-c3b6-4949-ba71-b1e58771a857" />

### New Sample Data Prediction

<img width="772" height="627" alt="Screenshot 2026-08-20 155425" src="https://github.com/user-attachments/assets/dd559ec1-cc5b-46dd-a68a-b62c4ad4544d" />


## RESULT
Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.

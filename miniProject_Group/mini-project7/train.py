import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)    # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # Conv layer 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Conv layer 3
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)        # FC layer 1
        self.fc2 = nn.Linear(1024, 256)                # FC layer 2
        self.fc3 = nn.Linear(256, 4)                   # Output layer (adjust number of classes if needed)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv layer 1 and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv layer 2 and pooling
        x = self.pool(F.relu(self.conv3(x)))  # Conv layer 3 and pooling
        x = torch.flatten(x, 1)               # Flatten the tensor
        x = F.relu(self.fc1(x))               # FC layer 1
        x = F.relu(self.fc2(x))               # FC layer 2
        x = self.fc3(x)                       # Output layer
        return x

# Function to show an image (optional for visualization)
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define your data directories
    train_dir = './data0/train'
    test_dir = './data0/test'

    # Define image transformations
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),              # Ensure all images are 64x64
        transforms.RandomHorizontalFlip(),        # Data augmentation
        transforms.RandomRotation(10),            # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),     # Normalize images
                             (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),              # Ensure all images are 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),     # Normalize images
                             (0.5, 0.5, 0.5))
    ])

    batch_size = 32  # Batch size

    # Load your training data
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Load your testing data
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Get class names from the dataset
    classes = trainset.classes  # ['cellphone', 'lamp', 'remote_control', 'tv']
    print('Classes:', classes)

    # Optionally, save class names to a file for consistency
    with open('classes.txt', 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    # Get some random training images (optional for visualization)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images (optional)
    # imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

    net = Net().to(device)  # Move the model to the appropriate device

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Learning rate scheduler

    # Training loop
    num_epochs = 20
    best_accuracy = 0.0  # For saving the best model

    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        net.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # Get inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = net(inputs)             # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward()                   # Backward pass
            optimizer.step()                  # Optimize

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        scheduler.step()  # Update the learning rate

        # Calculate training accuracy
        training_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {training_accuracy:.2f}%')

        # Evaluate on test data after each epoch
        net.eval()  # Set the model to evaluation mode
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%')

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(net.state_dict(), './best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.2f}%')

    print('Finished Training')

    # Load the best model for final evaluation
    net.load_state_dict(torch.load('./best_model.pth'))

    # Evaluate the network
    net.eval()  # Set the network to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for evaluation
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # Get predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Final Test Accuracy: {100 * correct / total:.2f}%')

    # Calculate accuracy for each class
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of {classes[i]:5s} : {accuracy:.2f}%')
        else:
            print(f'Accuracy of {classes[i]:5s} : N/A (no samples)')

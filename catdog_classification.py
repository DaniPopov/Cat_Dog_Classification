# Import 
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
#path: directory address where the images are stored
#images are then resized to 100×100 and converted to RGB
#return: np.array(images)
def load_images(path):

    images = []
    filenames = os.listdir(path)
    
    for filename in tqdm(filenames): 
        if filename == '_DS_Store':
            continue
        image = cv2.imread(os.path.join(path, filename))
        image = cv2.resize(image, dsize=(100,100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    
    return np.array(images)


# Sotre the images in cats_train,dogs_train,cats_test and dogs_test
cats_train = load_images('./data/dogcat_data/training_set/training_set/cats')
dogs_train = load_images('./data/dogcat_data/training_set/training_set/dogs')

cats_test = load_images('./data/dogcat_data/test_set/test_set/cats')
dogs_test = load_images('./data/dogcat_data/test_set/test_set/dogs')


print(f'cats_train.shape : {cats_train.shape}')
print(f'dogs_train.shape : {dogs_train.shape}')
print(f'cats_test.shape : {cats_test.shape}')
print(f'dogs_test.shape : {dogs_test.shape}')


X_train = np.append(cats_train, dogs_train, axis=0)
X_test = np.append(cats_test, dogs_test, axis=0)

print(f'X_train.shape: {X_train.shape}')
print(f'X_test.shape: {X_test.shape}')


# Creating Labels
# label cats with 0 and dogs with 1

y_train = np.array([0] * len(cats_train) + [1] * len(dogs_train))
y_test = np.array([0] * len(cats_test) + [1] * len(dogs_test))


print(f'y_train.shape: {y_train.shape}')
print(f'y_test.shape: {y_test.shape}')

# Displaying Several Images
def show_images(images, labels, start_index):
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20,12))
    counter = start_index

    for i in range(4):
        for j in range(8):
            axes[i,j].set_title(labels[counter].item())
            axes[i,j].imshow(images[counter], cmap='gray')
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
            counter +=1
    
    plt.show()        

show_images(X_train, y_train, 0)
show_images(X_train, y_train, 4001)


print(y_train[:10])

y_train = torch.from_numpy(y_train.reshape(len(y_train),1))
y_test = torch.from_numpy(y_test.reshape(len(y_test),1))

print(y_train[:10])

# Image Preprocessing and Augmentation
transforms_train = transforms.Compose([transforms.ToTensor(),
                                       transforms.RandomRotation(degrees=20),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomVerticalFlip(p=0.005),
                                       transforms.RandomGrayscale(p=0.2),
                                       transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                                       ])

transforms_test = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                                       ])

# Custom Dataset Class & Data Loader
class Cat_Dog_Dataset():
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return (image, label)

train_dataset = Cat_Dog_Dataset(images=X_train, labels=y_train, transform=transforms_train)
test_dataset = Cat_Dog_Dataset(images=X_test, labels=y_test, transform=transforms_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, drop_last=True)

iterator = iter(train_loader)
image_batch, label_batch = next(iterator)

print(image_batch.shape)


image_batch_permuted = image_batch.permute(0, 2, 3, 1)
print(image_batch_permuted.shape)
show_images(image_batch_permuted, label_batch, 0)

# Model 
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        #self.maxpool

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        #self.maxpool

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        #self.maxpool

        self.dropout = nn.Dropout(p=0.5)
        self.fc0 = nn.Linear(in_features=128*6*6, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x
    
model = CNN().to(device)
summary(model, input_size=(4,3,100,100))

loss_function = nn.BCELoss()
optimizier = optim.Adam(params=model.parameters(), lr=0.001)

def predict_test_data(model, test_loader):

    n_correct = 0
    n_samples = 0

    model.eval()

    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.float().to(device)
            y_test = y_test.float().to(device)

            # Calculate loss (forward propagation)
            test_preds = model(X_test)
            test_loss = loss_function(test_preds, y_test)

            # Calculate accuracy
            rounded_test_preds = torch.round(test_preds)
            n_correct += torch.sum(rounded_test_preds == y_test)
            n_samples += len(y_test)

    model.train()

    test_acc = n_correct/n_samples

    return test_loss, test_acc

# Training Loop
# Training and testing loss was calculated based on the last batch of each epoch.
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(100):
    n_correct_train = 0
    n_samples_train = 0
    for batch, (X_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        X_train = X_train.float().to(device)
        y_train = y_train.float().to(device)

        # Foward pass
        train_preds = model(X_train)
        train_loss = loss_function(train_preds, y_train)

        # Calculate train accuracy
        with torch.no_grad():
            rounded_train_preds = torch.round(train_preds)
            n_correct_train += torch.sum(rounded_train_preds == y_train)
            n_samples_train += len(y_train)

        # Backward pass
        optimizier.zero_grad()
        train_loss.backward()

        # Gradient descent
        optimizier.step()

    train_acc = n_correct_train/n_samples_train
    test_loss,  test_acc = predict_test_data(model,test_loader=test_loader)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accs.append(train_acc.item())
    test_accs.append(test_acc.item())
        
    print(f'Epoch: {epoch} \t|' \
            f' Train loss: {np.round(train_loss.item(),3)} \t|' \
            f' Test loss: {np.round(test_loss.item(),3)} \t|' \
            f' Train acc: {np.round(train_acc.item(),2)} \t|' \
            f' Test acc: {np.round(test_acc.item(),2)}')


# Evaluatuion
plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train_losses', 'test_losses'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.figure(figsize=(10,6))
plt.grid()
plt.plot(train_accs)
plt.plot(test_accs)
plt.legend(['train_accs', 'test_accs'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# Predicting Images on Test Set

# Load test images
iter_test = iter(test_loader)
img_test, lbl_test = next(iter_test)

# Predict labels
preds_test = model(img_test.to(device))
img_test_permuted = img_test.permute(0, 2, 3, 1)
rounded_preds = preds_test.round()

# Show test images and the predicted labels
show_images(img_test_permuted, rounded_preds, 0)


def pred_and_show(path,model):
    image  = cv2.imread(path)
    image  = cv2.resize(image, dsize=(100,100))
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Preprocess and convert to a PyTorch tensor
    preprocessed_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        model.eval()
        prediction  = model(preprocessed_image.to(device))

    # Interpret the prediction
    probability = prediction.item()
    if probability < 0.5:
        prediction_label = "Cat"
    else:
        prediction_label = "Dog"

    plt.imshow(image)
    plt.title(f"Prediction: {prediction_label} (Probability: {probability:.2f})")
    plt.axis('off')  # Hide axes
    plt.show()


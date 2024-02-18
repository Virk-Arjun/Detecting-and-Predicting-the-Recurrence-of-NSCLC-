import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import streamlit as st
from PIL import Image
# one-hot encoding
# [1, 0, 0, 0] = adenocarcinoma
# [0, 1, 0, 0] = largecellcarcinoma
# [0, 0, 1, 0] = squamouscell carcinoma
# [0, 0, 0, 1] = normal
# 128x128 pixels
img_size = 128
# locations of training image files
adenocarcinoma_training_folder = "NSCLC_Dataset/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/"
largecellcarcinoma_training_folder = "NSCLC_Dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa"
squamouscellcarcinoma_training_folder = "NSCLC_Dataset/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"
normal_training_folder = "NSCLC_Dataset/train/normal"
# locations of testing image files
adenocarcinoma_testing_folder = "NSCLC_Dataset/test/adenocarcinoma"
largecellcarcinoma_testing_folder = "NSCLC_Dataset/test/large.cell.carcinoma"
squamouscellcarcinoma_testing_folder = "NSCLC_Dataset/test/squamous.cell.carcinoma"
normal_testing_folder = "NSCLC_Dataset/test/normal"
classes = ["Adenocarcinoma", "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Normal"]
# Load and process images function
def load_and_process_images(folder, label):
    data = [[np.array(cv2.resize(cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE), (img_size, img_size))), np.array(label)] for filename in os.listdir(folder)]
    return data
# Load training and testing data
adenocarcinoma_training_data = load_and_process_images(adenocarcinoma_training_folder, [1, 0, 0, 0])
largecellcarcinoma_training_data = load_and_process_images(largecellcarcinoma_training_folder, [0, 1, 0, 0])
squamouscellcarcinoma_training_data = load_and_process_images(squamouscellcarcinoma_training_folder, [0, 0, 1, 0])
normal_training_data = load_and_process_images(normal_training_folder, [0, 0, 0, 1])
adenocarcinoma_testing_data = load_and_process_images(adenocarcinoma_testing_folder, [1, 0, 0, 0])
largecellcarcinoma_testing_data = load_and_process_images(largecellcarcinoma_testing_folder, [0, 1, 0, 0])
squamouscellcarcinoma_testing_data = load_and_process_images(squamouscellcarcinoma_testing_folder, [0, 0, 1, 0])
normal_testing_data = load_and_process_images(normal_testing_folder, [0, 0, 0, 1])
# Convert lists to NumPy arrays
adenocarcinoma_training_data = np.array(adenocarcinoma_training_data, dtype=object)
largecellcarcinoma_training_data = np.array(largecellcarcinoma_training_data, dtype=object)
squamouscellcarcinoma_training_data = np.array(squamouscellcarcinoma_training_data, dtype=object)
normal_training_data = np.array(normal_training_data, dtype=object)
adenocarcinoma_testing_data = np.array(adenocarcinoma_testing_data, dtype=object)
largecellcarcinoma_testing_data = np.array(largecellcarcinoma_testing_data, dtype=object)
squamouscellcarcinoma_testing_data = np.array(squamouscellcarcinoma_testing_data, dtype=object)
normal_testing_data = np.array(normal_testing_data, dtype=object)
# Combine and shuffle training and testing data
training_data = np.concatenate((adenocarcinoma_training_data, largecellcarcinoma_training_data, squamouscellcarcinoma_training_data, normal_training_data), axis=0)
np.random.shuffle(training_data)
testing_data = np.concatenate((adenocarcinoma_testing_data, largecellcarcinoma_testing_data, squamouscellcarcinoma_testing_data, normal_testing_data), axis=0)
np.random.shuffle(testing_data)
# Save the processed data
np.save("NSCLC_training_data.npy", training_data)
np.save("NSCLC_testing_data.npy", testing_data)
print()
print(f"Adenocarcinoma training count: {len(adenocarcinoma_training_data)}")
print(f"Large Cell Carcinoma training count: {len(largecellcarcinoma_training_data)}")
print(f"Squamous Cell Carcinoma training count: {len(squamouscellcarcinoma_training_data)}")
print(f"Normal training count: {len(normal_training_data)}")
print()
print(f"Adenocarcinoma testing count: {len(adenocarcinoma_testing_data)}")
print(f"Large Cell Carcinoma testing count: {len(largecellcarcinoma_testing_data)}")
print(f"Squamous Cell Carcinoma testing count: {len(squamouscellcarcinoma_testing_data)}")
print(f"Normal testing count: {len(normal_testing_data)}")
# Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # linear layers
        self.fc1 = nn.Linear(16 * (img_size // 4) * (img_size // 4), 256)  # Adjust the input size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        # dropout
        self.dropout = nn.Dropout(p=0.2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flattening the image
        x = x.view(-1, 16 * (img_size // 4) * (img_size // 4))
        # linear layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
net = Net()
print (net)
# Streamlit app
st.title("NSCLC Image Classification")
uploaded_file = st.file_uploader("Upload a CT image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded CT Image.', use_column_width=True)
    # Detect button
    detect_button = st.button("Detect")
    if detect_button:
        # Preprocess the image for prediction
        try:
            image = cv2.resize(np.array(image), (img_size, img_size))
            image = torch.Tensor(image / 255).view(1, 1, img_size, img_size)
            # Make prediction with raw output logging
            net.eval()
            with torch.no_grad():
                output = net(image)
                print("Raw model output:", output)
                predicted_class = torch.argmax(output).item()
                print("Predicted class index:", predicted_class)
            # Check if the predicted_class is within the valid range
            if 0 <= predicted_class < len(classes):
                st.write(f"Prediction: {classes[predicted_class]}")
            else:
                st.write("Invalid prediction index.")
        except Exception as e:
            st.write(f"Error processing the image: {e}")
# Training code
training_data = np.load("NSCLC_training_data.npy", allow_pickle=True)
# Convert the list of numpy arrays to a single numpy array
train_X = torch.Tensor(np.array([item[0] for item in training_data])) / 255
# One-hot vector labels tensor
train_y = torch.LongTensor(np.array([item[1] for item in training_data]))
net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()
batch_size = 64
epochs = 30
for epoch in range(epochs):
    for i in range(0, len(train_X), batch_size):
        print(f"EPOCH {epoch+1}, fraction complete: {i/len(train_X)}")
        batch_X = train_X[i: i+batch_size].view(-1, 1, img_size, img_size)
        batch_y = train_y[i: i+batch_size]
        optimizer.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, torch.argmax(batch_y, dim=1))
        loss.backward()
        optimizer.step()
# Save the trained model
torch.save(net.state_dict(), "saved_model.pth")
# Load the trained model
net = Net()
net.load_state_dict(torch.load('saved_model.pth'))
net.eval()
# Testing code
testing_data = np.load("NSCLC_testing_data.npy", allow_pickle=True)
# Putting all the image arrays into this tensor
test_X = torch.Tensor(np.array([item[0] for item in testing_data]) / 255)
# One-hot vector labels tensor
test_y = torch.LongTensor(np.array([item[1] for item in testing_data]))
correct = 0
total = 0
with torch.no_grad():
    # Tells PyTorch not to automatically keep track of gradients
    for i in range(len(test_X)):
        output = net(test_X[i].view(1, 1, img_size, img_size))
        # Use torch.argmax to get the predicted class index
        predicted_class = torch.argmax(output).item()
        real_class = torch.argmax(test_y[i]).item()
        if predicted_class == real_class:
            correct += 1
        total += 1
accuracy = correct / total
print(f"Accuracy: {accuracy:.3f}")


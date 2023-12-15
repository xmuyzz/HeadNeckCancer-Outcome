import torch
import torch.nn as nn
import torch.optim as optim
#from torchvision.models import densenet121  # You can use other variants like densenet169, densenet201
from models import DenseNet 
import torch.nn.functional as F


class CustomDenseNet(nn.Module):
    def __init__(self, model_depth, n_classes, num_clinical_features, in_channels):
        super(CustomDenseNet, self).__init__()

        # Load pre-trained DenseNet model
        self.densenet = DenseNet.generate_model(
            model_depth=model_depth,
            num_classes=n_classes,
            n_input_channels=in_channels)

        # Modify the classifier to accept clinical variables
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features + num_clinical_features, n_classes)

        # Additional fully connected layer for clinical data
        self.fc_clinical = nn.Linear(num_clinical_features, 30)  # Adjust the size as needed

    #def forward(self, image_input, clinical_input):
    def forward(self, image_input, clinical_input):
        # Forward pass through DenseNet for image processing
        # print('img_input:', image_input.shape)
        # print('clinical_input:', clinical_input.shape)
        # print('img_input:', image_input)
        # print('clinical_input:', clinical_input)
        x_image = self.densenet.features(image_input)
        x_image = F.relu(x_image, inplace=True)
        # average pooling and flatten
        x_image = F.adaptive_avg_pool3d(x_image, output_size=(1, 1, 1)).view(x_image.size(0), -1)
        # print('x_image shape:', x_image.shape)
        # print('x_image:', x_image)

        #x_clinical = F.adaptive_avg_pool3d(clinical_input, output_size=(1, 1, 1)).view(x_image.size(0), -1)
        #x_clinical = self.densenet.features(clinical_input)
        # feature extraction
        #x_image = F.relu(x_image, inplace=True)
        # average pooling and flatten
        x_clinical = clinical_input.float().view(x_image.size(0), -1)
        # print('x_clinical:', x_clinical, x_clinical.shape)
        # print("Weight shape:", self.fc_clinical.weight.shape)
        # print("Bias shape:", self.fc_clinical.bias.shape)

        # weight_tensor = self.fc_clinical.weight
        # weight_dtype = weight_tensor.dtype
        # #print("Weight Data Type:", weight_dtype)
        # # convert clincal input to the dtype of weight_tensor
        # x_clinical = x_clinical.to(weight_dtype)
        #x_clinical = self.fc_clinical(x_clinical)
        #print('x_clinical shape:', x_clinical.shape)
        #print('x_clinical:', x_clinical)

        # Concatenate image and clinical features
        x_combined = torch.cat((x_image, x_clinical), dim=1)
        # print('last layer shape:', x_combined.shape, x_combined)

        # Forward pass through modified DenseNet classifier
        output = self.densenet.classifier(x_combined)

        return output


"""
# Instantiate the model
num_classes = 10  # Adjust according to your task
num_clinical_features = 5  # Adjust according to the number of clinical features
model = CustomDenseNet(num_classes, num_clinical_features)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for images, clinical_data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images, clinical_data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Testing loop
model.eval()
with torch.no_grad():
    for images, clinical_data, labels in test_loader:
        outputs = model(images, clinical_data)
        # Evaluate the model as needed
"""
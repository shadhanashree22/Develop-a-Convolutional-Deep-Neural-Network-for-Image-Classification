# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   THEORY

Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset Fashion-MNIST and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model

<img width="1039" height="754" alt="image" src="https://github.com/user-attachments/assets/60b76dac-5d2c-49c3-a18f-35285600a6ce" />


## DESIGN STEPS

1.Load and Preprocess Data

2.Get the shape of the first image in the training dataset

3.Get the shape of the first image in the test dataset

4.Train the Model

5.Test the Model

6.Predict on a Single Image

7.Display the image



## PROGRAM

### Name: S V SHADHANASHREE

### Register Number: 212223230202

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x

model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)


## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here
    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

      print('Name: S V SHADHANASHREE')
      print('Register Number:212223230202')
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
```

### OUTPUT

## Training Loss per Epoch

<img width="550" height="307" alt="image" src="https://github.com/user-attachments/assets/dd74a12a-0d90-4237-935d-8fd9e70eb029" />


## Confusion Matrix

<img width="741" height="678" alt="image" src="https://github.com/user-attachments/assets/b0ba9566-7a79-4cb2-afe9-e780b6ec1171" />


## Classification Report

<img width="620" height="452" alt="image" src="https://github.com/user-attachments/assets/7081e28c-98ac-4b6d-9f0c-efe1d45b057f" />


### New Sample Data Prediction

<img width="544" height="620" alt="image" src="https://github.com/user-attachments/assets/8d85401a-3b0b-43b2-82bc-3f6988f20d8a" />


## RESULT
Thus , a convolutional deep neural network (CNN) for image classification and to verify the response for new images is successfully developed.

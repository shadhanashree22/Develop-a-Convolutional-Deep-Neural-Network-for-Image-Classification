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

### Name: Syed Mohamed Raihan M

### Register Number: 212224240167

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

<img width="383" height="164" alt="image" src="https://github.com/user-attachments/assets/5e57c6a3-181e-4714-b9a5-48c6b50f0275" />


## Confusion Matrix

<img width="918" height="804" alt="image" src="https://github.com/user-attachments/assets/0b52a373-b2b1-4150-839e-86f1410f2781" />


## Classification Report

<img width="590" height="421" alt="image" src="https://github.com/user-attachments/assets/6b82dddc-e8a8-4462-88b6-d30ba6b846ee" />

### New Sample Data Prediction

<img width="614" height="612" alt="image" src="https://github.com/user-attachments/assets/b3c4954f-df69-4cbf-81c6-9f3f14540da2" />


## RESULT
Thus , a convolutional deep neural network (CNN) for image classification and to verify the response for new images is successfully developed.

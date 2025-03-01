# plant_disease_detection

**Project Overview**

The Plant Disease Detection project uses machine learning to classify plant leaf images as either healthy or diseased. It leverages TensorFlow for model creation and Flask for the web-based interface. The dataset is automatically downloaded from Kaggle using the Kaggle API.

**End-to-End Execution Steps**

1. Install the required dependencies using the requirements.txt file.
2. Place kaggle.json in the ./Plant_disease/secrets/.kaggle folder.
3. Run train_model.py to download the dataset and train the model.
4. Start the Flask app by running app.py.
5. Upload plant leaf images through the web interface to get the disease detection result.
6. You can select between a pretrained model and your custom trained one for prediction.

**Use Case**
This system helps farmers and researchers quickly identify plant diseases, reducing the time required for diagnosis and improving crop yield.

**Model Architecture Explanation**
The model is a Convolutional Neural Network (CNN) designed for binary image classification.
_Conv2D Layers (32, 64 filters):_ These layers extract features from the input image by applying 3x3 filters.
_Activation Function (ReLU):_ It introduces non-linearity, helping the model learn complex patterns.
_MaxPooling2D Layers: _These layers reduce the spatial dimensions, lowering computational cost and preventing overfitting.
_Flatten Layer:_ Converts the 2D feature maps into a 1D feature vector.
_Dense Layer (128 units):_ Fully connected layer for high-level feature representation.
_Output Layer (1 unit, sigmoid activation):_ Outputs a probability score for binary classification.
_Optimizer (Adam):_ Chosen for its adaptive learning rate and efficiency in training deep networks.
_Loss Function (Binary Crossentropy):_ Suitable for binary classification problems.

**Improvements**

  . Use a more extensive dataset with multiple plant types.

  . Implement multi-class classification.

  . Integrate mobile app support.

  . Deploy the model on cloud services for remote access.
  
  . Add more image preprocessing techniques.

**Conclusion**
This project provides a foundational system for plant disease detection. Further improvements can make the system more robust and scalable for agricultural use.

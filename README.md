# **CIFAR-10 Image Classification using Multi-Layer Perceptron (MLP)**

![image](https://github.com/prajwal-cn/Deep-Learning---Multilayer-Perceptron-/assets/127007794/5fb42daa-eedf-4af0-8ec7-114bfe89d6cc) 

This project implements a Multi-Layer Perceptron (MLP) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to train an MLP model that can accurately classify these images into their respective classes.

### **Requirements**
To run this code, you need the following libraries and frameworks:

numpy (v1.0 or higher)
matplotlib (v3.0 or higher)
tensorflow (v2.0 or higher)
You can install these dependencies using pip:

### **1. Parameters**
This section defines the number of classes (NUM_CLASSES) which is set to 10 in this case.

### **2. Prepare the Data**
The CIFAR-10 dataset is loaded using the datasets.cifar10.load_data() function provided by the TensorFlow Keras library. The dataset is then preprocessed by scaling the pixel values between 0 and 1. Additionally, the labels are one-hot encoded using utils.to_categorical().

### **3. Build the Model**
The MLP model is built using the Keras functional API. The model architecture consists of an input layer, a flatten layer to convert the 32x32 images into a 1D vector, two dense layers with ReLU activation, and an output layer with softmax activation. The model summary is printed to provide an overview of the model's architecture.

### **4. Train the Model**
The model is compiled with the Adam optimizer and categorical cross-entropy loss function. It is then trained using the training data (x_train and y_train) with a batch size of 32 and for 10 epochs. The training process also shuffles the training data.

### **5. Evaluation**
The trained model is evaluated on the test data (x_test and y_test) using the evaluate() function, which calculates the loss and accuracy of the model on the test set.

### **6. Visualization**
This section visualizes the model's predictions on a random selection of images from the test set. The predicted class labels and the actual class labels are displayed below each image.

**To run the code, execute the following command:**

python your_script_name.py
Make sure to replace your_script_name.py with the actual name of your script file.

### **Results**
After running the code, the MLP model will be trained on the CIFAR-10 dataset, and its performance will be evaluated on the test set. The evaluation results will be printed, showing the test loss and accuracy of the model.

Additionally, a visualization of the model's predictions will be displayed, showing a selection of images from the test set along with their predicted and actual class labels.

### **Conclusion**
This project demonstrates the implementation of an MLP for image classification using the CIFAR-10 dataset. By following the steps outlined in the code, you can train and evaluate the MLP model on the dataset. Feel free to modify the code and experiment with different architectures, hyperparameters, and datasets to further improve the model's performance.

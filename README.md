# ACCURATE DETECTION AND PREDICTION OF ANXIETY AND SLEEP APNEA USING PHYSIOLOGICAL SINGALS 

The proposed framework utilizes physiological signals such as electrocardiogram (ECG), to accurately identify anxiety and sleep apnea events. The framework employs advanced Deep Learning techniques to analyze and classify the physiological signals, enabling the prediction of anxiety and sleep apnea events with high accuracy.

# Dataset
ANXIETY
The WESAD (WEarable Stress and Affect Detection) [7] is a dataset created for stress detection using physiological signals (ECG). 
Among the measures, the dataset contains ECG measures of 15 subjects during 2 hours of stressing, amusing, relaxing and neutral situations. The ECG is measured with an ECG sensor placed on the chest with a frequency of 700 Hz. 

SLEEP APNEA
The Apnea ECG-dataset used consits of PSG recordings of 25 patients out of which 21 have sleep apnea and 4 do not have sleep apnea, segmented in a window of 11 seconds each. The raw data obtained is cleaned and filtered and converted MATLAB files (.mat) for further processing. This data is then split in the ratio 10:1:1 into training, validation and test sets. All the patient records have been used.

![image](https://github.com/mahi1492/Accurate-Detection-and-Prediction-if-Anxiety-and-Sleep-Apnea-using-Physiological-Signals/assets/83654923/35bff82b-8139-4248-aa58-d25733af6aa2)

# Model Implementation 
ANXIETY 
The model is a Full Connected Neural Network (FCNN) implemented using a Deep Neural Network (DNN) classifier. Each Fully Connected (FC) layer is followed by a Batch Normalization layer, a Dropout(p = 0.5) layer, and a LeakyRelu (a = 0.2) layer.
The size of these layers decreases from 128 → 64 → 16 → 4 → 1. The final FC layer is followed by a Sigmoid function to obtain an output ∈ [0;1].
The input size is 12 and the output size is 1. An output > a-given-threshold is considered as a stress state.

![image](https://github.com/mahi1492/Accurate-Detection-and-Prediction-if-Anxiety-and-Sleep-Apnea-using-Physiological-Signals/assets/83654923/fc2e63cd-a69e-491e-a3b5-8b6ea33fcd13)

For each fold (of the 91-fold), the model has been trained with :
1. Loss Function = Binary Cross Entropy
2. Epochs = 15
3. Batch size = 32
4. Learning Rate = 0.0001
5. Optimizer = Adam(learning rate, beta1 = 0.9, beta2 = 0.999)

SLEEP APNEA 
The proposed method for detecting sleep apnea on a per-second basis uses a 1D-CNN that has 1408 nodes in its input layer, which corresponds to 128 samples per second for an 11-second window. Prior to the input layer, the samples are standardized using a standard normalization technique. The input layer is then followed by a batch normalization stage. The 1D-CNN model has 3 convolution layers, where all pooling layers use the maximum pooling method, and all activations use the Rectified Linear Unit (ReLU) activation function.

![image](https://github.com/mahi1492/Accurate-Detection-and-Prediction-if-Anxiety-and-Sleep-Apnea-using-Physiological-Signals/assets/83654923/dc2cbab9-973b-4c23-aca4-f36d4dede416)

To prevent overfitting during training, the fully connected layers implement weight dropout with a probability of 0.25. The binary cross-entropy loss function is used for model optimization with the ADAM optimizer, and the learning rate is fixed at 0.001. The output layer applies the softmax activation function.
The model considered in this study has a total of 50,909 parameters, which are the values that the model uses to adjust its weights during training. These parameters are learned through the optimization process and are critical for the model's ability to make accurate predictions.

# Performance Evaluation 
ANXIETY 
1. Confusion Matrix

![image](https://github.com/mahi1492/Accurate-Detection-and-Prediction-if-Anxiety-and-Sleep-Apnea-using-Physiological-Signals/assets/83654923/0bd0cd13-3458-44a3-9cf2-73e769a9144f)

Accuracy = 0.957
Precision = 0.851
Recall = 1.00
F1 Score = 0.920

SLEEP APNEA 
1. Confusion Matrix

![image](https://github.com/mahi1492/Accurate-Detection-and-Prediction-if-Anxiety-and-Sleep-Apnea-using-Physiological-Signals/assets/83654923/905b7f7d-b004-4e64-bcdc-806813e98c1b)

Accuracy = 0.99009
Precision = 0.99
Recall = 0.88
F1 Score = 0.79



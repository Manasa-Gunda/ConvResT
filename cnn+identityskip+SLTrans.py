#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Conv1D,Add
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization,MaxPool1D,Concatenate
from tensorflow.keras.layers import GlobalAvgPool1D
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical


# In[325]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:\\New_arithmetic_task - Copy\\TRANSFORMER\\subject_35.csv")

x = df.iloc[:, 0:21]
y = df.iloc[:, 21]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .90)


# In[326]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform (x_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[327]:


x_train_reshaped = tf.reshape(x_train, (-1, 21, 1))
x_test_reshaped = tf.reshape(x_test, (-1, 21, 1))
print(x_train_reshaped.shape)
print(x_test_reshaped.shape)


# In[328]:


##skip connection
def resnet_block1(input_tensor, filters, kernel_size=3, stride=1):
    # Shortcut
    shortcut = input_tensor

    # First convolutional layer
    x = Conv1D(filters, 1, strides=stride, padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third convolutional layer
    x = Conv1D(filters * 4, 1)(x)  # 1x1 Conv with 4 times the number of filters
    x = BatchNormalization()(x)

    # Identity skip connection with 1x1 convolution
    if stride != 1 or shortcut.shape[-1] != filters * 4:
        shortcut = Conv1D(filters * 4, 1, strides=stride)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add the shortcut to the output
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


# In[329]:


import keras
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, SeparableConv1D, MaxPooling1D,Reshape,
    Flatten, Dense, Bidirectional, LSTM, MultiHeadAttention, LayerNormalization, Dropout, Add
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.5):
        super(TransformerBlock, self).__init__()

        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)

        self.dense1 = Dense(dff, activation='relu')
        self.dense2 = Dense(d_model)
        self.dropout2 = Dropout(rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        #inputs = tf.keras.layers.Reshape((112, 112, 64))(inputs)
        #inputs = tf.keras.layers.Reshape((56*56, 64))(inputs)
        attn_output = self.multi_head_attention1(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ff_output = self.dense1(out1)
        ff_output = self.dense2(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.layernorm2(out1 + ff_output)

        return out2


# In[330]:


from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, MaxPooling1D,Add,
    Flatten, Dense, GlobalAveragePooling1D, DepthwiseConv1D, SeparableConv1D,Concatenate
)
from tensorflow.keras.models import Model


def create_eeg_model(input_shape, num_classes, num_repeats):
    #input_layer = Input(shape=input_shape)
    #x = input_layer
    input_shape = (21, 1)
    inputs = tf.keras.layers.Input(shape=input_shape)
    #input_shape = (62000, 21)
    #inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    for _ in range(num_repeats):
        # Add Convolutional Layer
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Add a Residual Block here
        x = resnet_block1(x, filters=32, kernel_size=3, stride=1)  # You can define this function

        # Add Depthwise Separable Convolution Layer
        #input_shape = (62000, 1, 32)  # Adjust the shape to match your data

        # Reshape the input data to (batch_size, sequence_length, input_dim)
        #x = Reshape(target_shape=input_shape)(x)
        x = tf.keras.layers.DepthwiseConv1D(kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Add Pointwise Convolution Layer
        x = SeparableConv1D(filters=32, kernel_size=1, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # Add Max Pooling Layer
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)        
        x = MaxPooling1D(pool_size=2)(x)
        #x = Flatten()(x)


        # Calculate transformer_output here if you want it within the loop
        transformer_output1 = TransformerBlock(d_model=32, num_heads=4, dff=128)(x)
        transformer_output2 = TransformerBlock(d_model=32, num_heads=4, dff=128)(transformer_output1)
        transformer_output3 = TransformerBlock(d_model=32, num_heads=4, dff=128)(transformer_output2)
        transformer_output4 = TransformerBlock(d_model=32, num_heads=4, dff=128)(transformer_output3)

    # Flatten or Global Average Pooling (Choose one)
    
    x = GlobalAveragePooling1D()(transformer_output4)
    x = Flatten()(x)
    dense1 = Dense(32, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(dense1)

    model = Model(inputs=inputs, outputs=output_layer)
    return model
#model.summary()


# In[331]:


#input_shape = (224, 224, 3)  # Example input shape (height, width, channels)
input_shape = (21,)
# Create an input layer with the defined shape
inputs = tf.keras.layers.Input(shape=input_shape)
num_classes = 1  # Number of classes
num_repeats = 2  # Number of times to repeat the model architecture
model = create_eeg_model(input_shape=(62000, 21), num_classes=num_classes, num_repeats=num_repeats)
model.summary()


# In[332]:


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score  # Choose your desired metrics
#from your_model import create_model  # Import your model creation function


# Define the number of folds for cross-validation
num_folds = 5  # Adjust as needed

# Initialize lists to store evaluation metrics
roc_auc_scores = []
accuracy_scores = []
roc_curve_data = []

# Initialize k-fold cross-validation
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate through each fold
for train_index, val_index in kf.split(x_train,y_train):
    X_train_fold, X_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Create your neural network model here, compile it, and fit it to the fold's data
    model = create_eeg_model(input_shape=(62000, 21), num_classes=1, num_repeats=2)  # Adjust this function to create your model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-05),
              loss="binary_crossentropy",
              metrics=['accuracy'])  # Adjust loss and metrics as needed

    # Train the model on this fold
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=16, validation_data=(X_val_fold, y_val_fold))

    # Evaluate the model on the test set for this fold
    accuracy = model.evaluate(x_test_reshaped, y_test, verbose=0)[1]
    accuracy_scores.append(accuracy)


    # Evaluate the model on the validation set and compute ROC AUC and accuracy
    y_pred_prob = model.predict(X_val_fold)
    roc_auc = roc_auc_score(y_val_fold, y_pred_prob)
    accuracy = accuracy_score(y_val_fold, (y_pred_prob > 0.5).astype(int))
    
     # Append evaluation metrics to lists
    roc_auc_scores.append(roc_auc)
    accuracy_scores.append(accuracy)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_val_fold, y_pred_prob)
    roc_curve_data.append((fpr, tpr))

  
 #Calculate mean and standard deviation of evaluation metrics
mean_roc_auc = np.mean(roc_auc_scores)
std_roc_auc = np.std(roc_auc_scores)
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

# Print and analyze the results
print(f'Mean ROC AUC: {mean_roc_auc:.2f} (std: {std_roc_auc:.2f})')
print(f'Mean Accuracy: {mean_accuracy:.2f} (std: {std_accuracy:.2f})')


# In[17]:


#no
#from sklearn.metrics import auc

# Plot ROC curves
#plt.figure(figsize=(6, 3))
#for i, (fpr, tpr) in enumerate(roc_curve_data):
    #plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC = {roc_auc_scores[i]:.2f})')

# Compute and plot the mean ROC curve
#mean_fpr = np.linspace(0, 1, 100)
#mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curve_data], axis=0)
#mean_auc = auc(mean_fpr, mean_tpr)
#plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', lw=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')

# Calculate and display the mean accuracy and its standard deviation
#mean_accuracy = np.mean(accuracy_scores)
#std_accuracy = np.std(accuracy_scores)
#plt.axvline(x=mean_accuracy, color='r', linestyle='--', label=f'Mean Accuracy (std: {std_accuracy:.2f})')

#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) and Accuracy')
#plt.legend(loc='lower right')
#plt.show()


# In[333]:


import os
import pickle

# Define the directory path
results_dir = "results6/"

# Create the directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# In[334]:


# Inside the subject loop
subject_id = "subject_35"  # Replace with the actual subject ID
subject_dir = os.path.join(results_dir, subject_id)

# Create the subject's directory if it doesn't exist
if not os.path.exists(subject_dir):
    os.makedirs(subject_dir)


# In[335]:


# Save ROC AUC and accuracy to CSV files
roc_auc_file = os.path.join(subject_dir, "roc_auc_scores.csv")
accuracy_file = os.path.join(subject_dir, "accuracy_scores.csv")

np.savetxt(roc_auc_file, roc_auc_scores, delimiter=",")
np.savetxt(accuracy_file, accuracy_scores, delimiter=",")

# Save ROC curve data as a pickle file
roc_curve_file = os.path.join(subject_dir, "roc_curve_data.pkl")
with open(roc_curve_file, 'wb') as file:
    pickle.dump(roc_curve_data, file)


# In[207]:


#YES,yes
import os
import numpy as np
import pandas as pd

# Define the directory containing subject folders
subjects_dir = "C:\\Users\\Manasa\\results6\\"

# Initialize lists to store scores
#roc_auc_scores_list = []
accuracy_scores_list = []

# Iterate through subject folders
for subject_folder in os.listdir(subjects_dir):
    if os.path.isdir(os.path.join(subjects_dir, subject_folder)):
        # Load ROC AUC scores CSV
        #roc_auc_file = os.path.join(subjects_dir, subject_folder, "roc_auc_scores.csv")
        #roc_auc_scores = np.loadtxt(roc_auc_file, delimiter=",")
        #roc_auc_scores_list.append(roc_auc_scores)

        # Load accuracy scores CSV
        accuracy_file = os.path.join(subjects_dir, subject_folder, "accuracy_scores.csv")
        accuracy_scores = np.loadtxt(accuracy_file, delimiter=",")
        accuracy_scores_list.append(accuracy_scores)

# Calculate mean and standard deviation for ROC AUC scores
#mean_roc_auc = np.mean(roc_auc_scores_list, axis=0)
#std_roc_auc = np.std(roc_auc_scores_list, axis=0)

# Calculate mean and standard deviation for accuracy scores
mean_accuracy = np.mean(accuracy_scores_list, axis=0)
std_accuracy = np.std(accuracy_scores_list, axis=0)

# Create a DataFrame to store mean and std values
#results_df = pd.DataFrame({
    #"Mean ROC AUC": mean_roc_auc,
    #"Std ROC AUC": std_roc_auc
    
#})
results_df = pd.DataFrame({
    "Mean Accuracy": mean_accuracy,
    "Std Accuracy": std_accuracy
    
})
# Save the results to a CSV file
results_file = "mean_std_scores_ACC3.csv"
results_df.to_csv(results_file, index=False)


# In[201]:


#YES,YES
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

# Aggregate ROC AUC and accuracy scores across all subjects
all_roc_auc_scores = np.array(roc_auc_scores)
all_accuracy_scores = np.array(accuracy_scores)

mean_roc_auc = np.mean(all_roc_auc_scores, axis=0)
std_roc_auc = np.std(all_roc_auc_scores, axis=0)
mean_accuracy = np.mean(all_accuracy_scores, axis=0)
std_accuracy = np.std(all_accuracy_scores, axis=0)

# Calculate the mean ROC curve
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curve_data], axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

# Plot the final ROC curve with standard deviation
plt.figure(figsize=(8, 6))
for i, (fpr, tpr) in enumerate(roc_curve_data):
    plt.plot(fpr, tpr, lw=2, alpha=0.5, label=f'Subject {i+1} (AUC = {roc_auc_scores[i]:.2f})')

plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', lw=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')

plt.show()


# In[205]:


#import numpy as np
#import os
#import pickle
#import matplotlib.pyplot as plt
#from sklearn.metrics import auc

# Define the directory path
#results_dir = "C:\\Users\\Manasa\\results3\\"

# Define lists to store evaluation metrics for all subjects
#all_roc_auc_scores = []
#all_roc_curve_data = []

# List all the subject folders in the results directory
#subject_folders = [folder for folder in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, folder))]

# Iterate through each subject folder
#for subject_id in subject_folders:
    #subject_dir = os.path.join(results_dir, subject_id)

    # Load ROC AUC scores
    #roc_auc_file = os.path.join(subject_dir, "roc_auc_scores.csv")
    #roc_auc_scores = np.loadtxt(roc_auc_file, delimiter=",")

    # Append evaluation metrics to the list
    #all_roc_auc_scores.append(roc_auc_scores)

    # Load ROC curve data
    #roc_curve_file = os.path.join(subject_dir, "roc_curve_data.pkl")
    #with open(roc_curve_file, 'rb') as file:
        #roc_curve_data = pickle.load(file)

    # Append ROC curve data to the list
    #all_roc_curve_data.extend(roc_curve_data)

# Calculate the mean ROC curve
#mean_fpr = np.linspace(0, 1, 100)
#mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in all_roc_curve_data], axis=0)
#mean_auc = auc(mean_fpr, mean_tpr)

# Plot ROC curves for all subjects in a single plot
#plt.figure(figsize=(8, 6))
#for i, (fpr, tpr) in enumerate(all_roc_curve_data):
    #plt.plot(fpr, tpr, lw=2, alpha=0.5, label=f'Subject {subject_folders[i]} (AUC = {all_roc_auc_scores[i]:.2f})')

# Plot the mean ROC curve
#plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', lw=2, label=f'Mean ROC (AUC = {mean_auc:.2f})')

#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC)')
#plt.legend(loc='lower right')
#plt.show()


# In[208]:


#yes,yes
import os
import pandas as pd

# Define the directory containing subject folders
main_folder = "C:\\Users\\Manasa\\results3\\"# Replace with the actual path

# Initialize empty DataFrames to store the combined data
roc_auc_combined = pd.DataFrame()
accuracy_combined = pd.DataFrame()

# Iterate through subject folders
for subject_folder in os.listdir(main_folder):
    subject_path = os.path.join(main_folder, subject_folder)
    if os.path.isdir(subject_path):
        # Load ROC AUC scores CSV
        roc_auc_file = os.path.join(subject_path, "roc_auc_scores.csv")
        if os.path.exists(roc_auc_file):
            roc_auc_df = pd.read_csv(roc_auc_file)
            roc_auc_combined = pd.concat([roc_auc_combined, roc_auc_df], ignore_index=True)

        # Load accuracy scores CSV
        accuracy_file = os.path.join(subject_path, "accuracy_scores.csv")
        if os.path.exists(accuracy_file):
            accuracy_df = pd.read_csv(accuracy_file)
            accuracy_combined = pd.concat([accuracy_combined, accuracy_df], ignore_index=True)

# Save the combined DataFrames to CSV files
roc_auc_combined.to_csv(os.path.join(main_folder, "all_subjects_roc_auc_scores.csv"), index=False)
accuracy_combined.to_csv(os.path.join(main_folder, "all_subjects_accuracy_scores.csv"), index=False)


# In[5]:


import os
import pickle
import matplotlib.pyplot as plt

# Define the directory containing subject folders
subjects_dir = "C:\\Users\\Manasa\\results3\\"

# Initialize a list to store all ROC curve data
all_roc_curve_data = []

# Iterate through subject folders
for subject_folder in os.listdir(subjects_dir):
    if os.path.isdir(os.path.join(subjects_dir, subject_folder)):
        # Load the ROC curve data for each subject
        roc_curve_file = os.path.join(subjects_dir, subject_folder, "roc_curve_data.pkl")
        with open(roc_curve_file, 'rb') as file:
            roc_curve_data = pickle.load(file)
        all_roc_curve_data.extend(roc_curve_data)

# Create a single plot for all subjects
plt.figure(figsize=(6, 5))

for fpr, tpr in all_roc_curve_data:
    plt.plot(fpr, tpr, lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Plot the diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)',fontsize=15)
plt.ylabel('True Positive Rate (TPR)',fontsize=15)
plt.title('ROC Curves for All Subjects',fontsize=15)
plt.xticks(fontsize=15)  # Increase the font size of x-axis ticks
plt.yticks(fontsize=15)  # Increase the font size of y-axis ticks
plt.show()


# In[9]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your main folder
main_folder = "C:\\Users\\Manasa\\results3\\"

# Initialize lists to store subject names and mean accuracy scores
subject_names = []
mean_accuracy_scores = []

# Iterate through subject folders
for i, subject_folder in enumerate(os.listdir(main_folder)):
    subject_dir = os.path.join(main_folder, subject_folder)
    accuracy_scores_file = os.path.join(subject_dir, "accuracy_scores.csv")

    # Check if the accuracy scores file exists in the subject folder
    if os.path.exists(accuracy_scores_file):
        # Load the accuracy scores from the CSV file without header
        df = pd.read_csv(accuracy_scores_file, header=None)
        
        # Calculate the mean accuracy score for this subject
        mean_accuracy = df.mean().values[0]
        
        # Append subject name and mean accuracy score to lists
        subject_names.append(subject_folder)
        mean_accuracy_scores.append(mean_accuracy)

# Create a DataFrame from the lists
mean_accuracy_df = pd.DataFrame({"Subject": range(0, len(subject_names)), "Mean Accuracy Score": mean_accuracy_scores})

# Save the mean accuracy scores to a CSV file
mean_accuracy_df.to_csv("mean_accuracy_scores.csv", index=False)

# Plot the mean accuracy scores for all subjects
plt.figure(figsize=(9, 3))
plt.bar(range(len(mean_accuracy_scores)), mean_accuracy_scores)
plt.xlabel("Subject")
plt.ylabel("Mean Accuracy Score")
plt.title("Mean Accuracy Scores for All Subjects")
plt.xticks(range(len(mean_accuracy_scores)), range(len(subject_names)), rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


#import os
#import pickle
#from sklearn.metrics import roc_auc_score

# Define the path to your main folder
#main_folder ="C:\\Users\\Manasa\\results3" 
# Initialize a list to store AUC scores
#auc_scores = []

# Iterate through subject folders
#for subject_folder in os.listdir(main_folder):
    #subject_dir = os.path.join(main_folder, subject_folder)
    #roc_curve_file = os.path.join(subject_dir, "roc_curve_data.pkl")
    
    # Check if the pickle file exists in the subject folder
    #if os.path.exists(roc_curve_file):
        # Load the ROC curve data from the pickle file
       # with open(roc_curve_file, "rb") as file:
           # roc_curve_data = pickle.load(file)
        
        # Calculate AUC for each ROC curve in the data
        #for fpr, tpr in roc_curve_data:
            #auc = roc_auc_score(true_labels, predicted_probs)  # Replace with your true labels and predicted probabilities
            #auc_scores.append(auc)

# Print or use the AUC scores as needed
#print(auc_scores)


# In[ ]:


#import os
#import pickle
#from sklearn.metrics import roc_auc_score

# Define the path to your main folder
#main_folder ="C:\\Users\\Manasa\\results3" 
# Initialize a list to store AUC scores
#auc_scores = []

# Iterate through subject folders
#for subject_folder in os.listdir(main_folder):
    #subject_dir = os.path.join(main_folder, subject_folder)
    #roc_curve_file = os.path.join(subject_dir, "roc_curve_data.pkl")
    
    # Check if the pickle file exists in the subject folder
   # if os.path.exists(roc_curve_file):
        # Load the ROC curve data from the pickle file
       # with open(roc_curve_file, "rb") as file:
          #  roc_curve_data = pickle.load(file)
        
        # Calculate AUC for each ROC curve in the data
        #for fpr, tpr in roc_curve_data:
           # auc = roc_auc_score(true_labels, predicted_probs)  # Replace with your true labels and predicted probabilities
            #auc_scores.append(auc)

# Print or use the AUC scores as needed
#print(auc_scores)


# In[ ]:


#import os
#import pickle
#from sklearn.metrics import roc_auc_score

# Define the path to your main folder
#main_folder ="C:\\Users\\Manasa\\results3" 
# Initialize a list to store AUC scores
#auc_scores = []

# Iterate through subject folders
#for subject_folder in os.listdir(main_folder):
    #subject_dir = os.path.join(main_folder, subject_folder)
   # roc_curve_file = os.path.join(subject_dir, "roc_curve_data.pkl")
    
    # Check if the pickle file exists in the subject folder
    #if os.path.exists(roc_curve_file):
        # Load the ROC curve data from the pickle file
       # with open(roc_curve_file, "rb") as file:
           # roc_curve_data = pickle.load(file)
        
        # Calculate AUC for each ROC curve in the data
        #for fpr, tpr in roc_curve_data:
           # auc = roc_auc_score(true_labels, predicted_probs)  # Replace with your true labels and predicted probabilities
           # auc_scores.append(auc)

# Print or use the AUC scores as needed
#print(auc_scores)


# In[3]:


#yes,yes
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Get the absolute path of the main folder
main_folder_path = os.path.abspath('C:\\Users\\Manasa\\results3\\')  # Replace 'MainFolder' with your actual main folder name

# Change the current working directory to the main folder
os.chdir(main_folder_path)


# Now you can proceed with the rest of the code

# Define the number of subjects (e.g., 36)
num_subjects = 36

# Create an empty DataFrame to store all ROC AUC scores
all_roc_auc_scores = pd.DataFrame()
results_list = []
for subject_id in range(num_subjects):
    # Navigate to the subject's folder
    subject_folder = f'subject_{subject_id:02d}'  # Adjust the folder naming convention if needed
    if not os.path.exists(subject_folder):
        continue  # Skip folders that don't exist
    os.chdir(subject_folder)

    # Load ROC AUC scores for the current subject from the CSV file
    roc_auc_df = pd.read_csv('roc_auc_scores.csv', header=None)  # Specify header=None to indicate no column names

    # Calculate mean ROC AUC score
    mean_roc_auc = roc_auc_df.mean().values[0]  # Calculate the mean of all values in the DataFrame

    # Print the mean ROC AUC score
    print(f'Mean ROC AUC for {subject_folder}: {mean_roc_auc:.2f}')

    # Perform a paired t-test against a hypothetical mean (e.g., 0.5 for random classification)
    t_statistic, p_value = stats.ttest_1samp(roc_auc_df[0], 0.5)  # Use the correct column name (e.g., [0])

    # Calculate 95% confidence interval for ROC AUC
    confidence_interval = stats.t.interval(0.95, len(roc_auc_df) - 1, loc=mean_roc_auc, scale=stats.sem(roc_auc_df[0]))  # Use the correct column name (e.g., [0])

    # Store the results in a DataFrame
    result_dict = {
        'Subject ID': subject_id,
        'Mean ROC AUC': mean_roc_auc,
        'T-Statistic': t_statistic,
        'P-Value': p_value,
        'CI Lower': confidence_interval[0],
        'CI Upper': confidence_interval[1]
    }
    all_roc_auc_scores = all_roc_auc_scores.append(result_dict, ignore_index=True)

    # Navigate back to the main folder
    os.chdir('..')
    # Create a DataFrame from the list of results
    #all_roc_auc_scores = pd.DataFrame(results_list)

# Print the results for all subjects
print(all_roc_auc_scores)

# Specify the path where you want to save the CSV file
csv_file_path = 'all_roc_auc_scores.csv'

# Save the DataFrame to a CSV file
all_roc_auc_scores.to_csv(csv_file_path, index=False)

# Print a message to confirm that the DataFrame has been saved
print(f'DataFrame saved to {csv_file_path}')

# Create a line graph of Mean ROC AUC scores for all subjects
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_subjects + 1), all_roc_auc_scores['Mean ROC AUC'], marker='o', linestyle='-')
plt.title('Mean ROC AUC Scores for All Subjects',fontsize=15)
plt.xlabel('Subject ID',fontsize=15)
plt.ylabel('Mean ROC AUC',fontsize=15)
plt.xticks(fontsize=15)  # Increase the font size of x-axis ticks
plt.yticks(fontsize=15)  # Increase the font size of y-axis ticks
plt.grid(True)  # Add a grid to the plot
plt.show()


# In[344]:


#1.correct
import os
import pandas as pd

# Define the directory where subject folders are located
base_directory = r'C:\\Users\\Manasa\\my_results\\results6\\'

# Create a directory to store the separated files
output_directory = r'C:\\Users\\Manasa\\my_results\\output6\\'
os.makedirs(output_directory, exist_ok=True)

# Function to separate odd and even rows and save to separate files
def separate_odd_even_rows(subject_folder):
    subject_directory = os.path.join(base_directory, subject_folder)
    accuracy_scores_file = os.path.join(subject_directory, 'accuracy_scores.csv')
    
    # Check if the file exists
    if os.path.exists(accuracy_scores_file):
        df = pd.read_csv(accuracy_scores_file)
        
        # Separate odd and even rows
        odd_rows = df.iloc[::2]
        even_rows = df.iloc[1::2]
        
        # Save odd and even rows to separate files
        odd_file = os.path.join(output_directory, f'{subject_folder}_odd_rows.csv')
        even_file = os.path.join(output_directory, f'{subject_folder}_even_rows.csv')
        
        odd_rows.to_csv(odd_file, index=False)
        even_rows.to_csv(even_file, index=False)

# Iterate through subject folders and separate rows
for subject_folder in os.listdir(base_directory):
    separate_odd_even_rows(subject_folder)

print("Separation of odd and even rows completed.")


# In[345]:


#2.correct-even-separate
import os
import pandas as pd

# Define the directory containing the even row files
even_rows_directory = "C:\\Users\\Manasa\\my_results\\output6\\even_odd_sep"  # Replace with the actual path

# Create an output directory to save the mean files
output_directory = "C:\\Users\\Manasa\\my_results\\output6\\even\\"  # Replace with the desired path

# Function to calculate the mean of a single file and save only the mean value
def calculate_and_save_mean(input_file, output_file):
    df = pd.read_csv(input_file)
    mean = df.mean()
    mean.to_csv(output_file, header=False, index=False)  # Exclude headers and index

# Iterate through even row files
for file_name in os.listdir(even_rows_directory):
    if file_name.endswith("_even_rows.csv"):
        file_path = os.path.join(even_rows_directory, file_name)
        
        # Define the output file path for the mean of this file
        output_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_mean.csv")
        
        # Calculate and save the mean for this file
        calculate_and_save_mean(file_path, output_file_path)

print("Calculation and saving of mean files completed.")


# In[346]:


#3.correct-odd-separate
import os
import pandas as pd

# Define the directory containing the even row files
odd_rows_directory = "C:\\Users\\Manasa\\my_results\\output6\\even_odd_sep"  # Replace with the actual path

# Create an output directory to save the mean files
output_directory = "C:\\Users\\Manasa\\my_results\\output6\\odd\\"  # Replace with the desired path

# Function to calculate the mean of a single file and save only the mean value
def calculate_and_save_mean(input_file, output_file):
    df = pd.read_csv(input_file)
    mean = df.mean()
    mean.to_csv(output_file, header=False, index=False)  # Exclude headers and index

# Iterate through even row files
for file_name in os.listdir(odd_rows_directory):
    if file_name.endswith("_odd_rows.csv"):
        file_path = os.path.join(odd_rows_directory, file_name)
        
        # Define the output file path for the mean of this file
        output_file_path = os.path.join(output_directory, f"{os.path.splitext(file_name)[0]}_mean.csv")
        
        # Calculate and save the mean for this file
        calculate_and_save_mean(file_path, output_file_path)

print("Calculation and saving of mean files completed.")


# In[347]:


#4.correct-odd-allcombined mean
import os
import pandas as pd

# Define the directory containing the even row files
odd_rows_directory = "C:\\Users\\Manasa\\my_results\\output6\\even_odd_sep"  # Replace with the actual path

# Create an output directory to save the mean file
output_directory = "C:\\Users\\Manasa\\my_results\\output6\\"  # Replace with the desired path

# Create an empty list to store all mean values
all_mean_values = []

# Function to calculate the mean of a single file and return the mean value
def calculate_mean(input_file):
    df = pd.read_csv(input_file)
    mean = df.mean().values[0]  # Extract the mean value and convert it to a scalar
    return mean

# Iterate through even row files
for file_name in os.listdir(odd_rows_directory):
    if file_name.endswith("_odd_rows.csv"):
        file_path = os.path.join(odd_rows_directory, file_name)
        
        # Calculate the mean for this file
        mean_value = calculate_mean(file_path)
        
        # Append the mean value to the list
        all_mean_values.append(mean_value)

# Define the output file path to save all mean values in one CSV file
output_file_path = os.path.join(output_directory, "all_mean_values_odd.csv")

# Save all mean values in one CSV file
with open(output_file_path, "w") as output_file:
    for mean_value in all_mean_values:
        output_file.write(f"{mean_value}\n")

print("Calculation and saving of mean values completed.")


# In[348]:


#5.correct-even-allcombined mean
import os
import pandas as pd

# Define the directory containing the even row files
even_rows_directory = "C:\\Users\\Manasa\\my_results\\output6\\even_odd_sep"  # Replace with the actual path

# Create an output directory to save the mean file
output_directory = "C:\\Users\\Manasa\\my_results\\output6\\"  # Replace with the desired path

# Create an empty list to store all mean values
all_mean_values = []

# Function to calculate the mean of a single file and return the mean value
def calculate_mean(input_file):
    df = pd.read_csv(input_file)
    mean = df.mean().values[0]  # Extract the mean value and convert it to a scalar
    return mean

# Iterate through even row files
for file_name in os.listdir(even_rows_directory):
    if file_name.endswith("_even_rows.csv"):
        file_path = os.path.join(even_rows_directory, file_name)
        
        # Calculate the mean for this file
        mean_value = calculate_mean(file_path)
        
        # Append the mean value to the list
        all_mean_values.append(mean_value)

# Define the output file path to save all mean values in one CSV file
output_file_path = os.path.join(output_directory, "all_mean_values_even.csv")

# Save all mean values in one CSV file
with open(output_file_path, "w") as output_file:
    for mean_value in all_mean_values:
        output_file.write(f"{mean_value}\n")

print("Calculation and saving of mean values completed.")


# In[349]:


#6.correct
import pandas as pd
import matplotlib.pyplot as plt

# Load the mean values from the CSV file
mean_values_df = pd.read_csv("C:\\Users\\Manasa\\my_results\\output6\\all_mean_values_odd.csv", header=None, names=["Mean"])

# Create a list of subject numbers based on the number of rows in the DataFrame
subject_numbers = list(range(1, len(mean_values_df) + 1))

# Plot the mean values
plt.figure(figsize=(19, 4))  # Adjust the figure size
plt.plot(subject_numbers, mean_values_df["Mean"], marker='o', linestyle='-')
plt.title('Mean Values for All Subjects')
plt.xlabel('Subject')
plt.ylabel('Mean Value')
plt.grid(True)
plt.xticks(subject_numbers)  # Display all subject numbers on the x-axis
plt.show()


# In[350]:


#7.correct-evenplot
import pandas as pd
import matplotlib.pyplot as plt

# Load the mean values from the CSV file
mean_values_df = pd.read_csv("C:\\Users\\Manasa\\my_results\\output6\\all_mean_values_even.csv", header=None, names=["Mean"])

# Create a list of subject numbers based on the number of rows in the DataFrame
subject_numbers = list(range(1, len(mean_values_df) + 1))

# Plot the mean values
plt.figure(figsize=(19, 4))  # Adjust the figure size
plt.plot(subject_numbers, mean_values_df["Mean"], marker='o', linestyle='-')
plt.title('Mean Values for All Subjects')
plt.xlabel('Subject')
plt.ylabel('Mean Value')
plt.grid(True)
plt.xticks(subject_numbers)  # Display all subject numbers on the x-axis
plt.show()


# In[351]:


import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
directory = "C:\\Users\\Manasa\\my_results\\output6\\"  # Replace with the actual path

# Load data from the CSV files without headers
even_data = pd.read_csv(os.path.join(directory, "all_mean_values_even.csv"), header=None)
odd_data = pd.read_csv(os.path.join(directory, "all_mean_values_odd.csv"), header=None)

# Create a list of subjects (e.g., subject IDs)
subjects = range(1, len(even_data) + 1)  # Assuming subjects are numbered sequentially

# Plot mean values for even and odd subjects on a single graph
plt.figure(figsize=(20, 6))
plt.plot(subjects, even_data.iloc[:, 0], marker='o', linestyle='-', label='Test')
plt.plot(subjects, odd_data.iloc[:, 0], marker='*', linestyle='-', label='Validation')

plt.title('Mean Accuracy Values of All Subjects')
plt.xlabel('Subject')
plt.ylabel('Mean Value')
plt.xticks(subjects)  # Set subject IDs as x-axis ticks
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


#yes
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the CSV files
directory = "C:\\Users\\Manasa\\my_results\\output6\\"  # Replace with the actual path

# Load data from the CSV files without headers
even_data = pd.read_csv(os.path.join(directory, "all_mean_values_even.csv"), header=None)
odd_data = pd.read_csv(os.path.join(directory, "all_mean_values_odd.csv"), header=None)

# Create a list of subjects (e.g., subject IDs)
subjects = range(0, len(even_data))  # Start from 0 to 35

# Plot mean values for even and odd subjects on a single graph
plt.figure(figsize=(8, 5))
plt.plot(subjects, even_data.iloc[:, 0], marker='o', linestyle='-', label='Test')
plt.plot(subjects, odd_data.iloc[:, 0], marker='*', linestyle='-', label='Validation')

plt.title('Mean of Test, Validation Values of All Subjects',fontsize=15)
plt.xlabel('Subject',fontsize=15)
plt.ylabel('Mean Value',fontsize=15)
plt.xticks(fontsize=15)  # Increase the font size of x-axis ticks
plt.yticks(fontsize=15)  # Increase the font size of y-axis ticks # Set subject IDs as x-axis ticks
plt.legend()
#plt.grid(True)
plt.show()


# In[8]:


#yesfinal-roc and auc
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder paths where the dataframes are located
folder_path1 = 'C:\\Users\\Manasa\\my_results'  # Replace 'path_to_folder1' with the actual path to the first folder
folder_path2 = 'C:\\Users\\Manasa\\my_results'  # Replace 'path_to_folder2' with the actual path to the second folder
folder_path3 = 'C:\\Users\\Manasa\\my_results'  # Replace 'path_to_folder3' with the actual path to the third folder

# Specify the actual names of the CSV files in each folder
csv_file1 = 'all_roc_auc_scores21.csv'  # Replace with the actual file name
csv_file2 = 'all_roc_auc_scores22.csv'  # Replace with the actual file name
csv_file3 = 'all_roc_auc_scores23.csv'  # Replace with the actual file name

# Load dataframes from different folders
all_roc_auc_scores21 = pd.read_csv(os.path.join(folder_path1, csv_file1))
all_roc_auc_scores22 = pd.read_csv(os.path.join(folder_path2, csv_file2))
all_roc_auc_scores23 = pd.read_csv(os.path.join(folder_path3, csv_file3))

# Extract the mean ROC AUC scores from each dataframe
mean_roc_auc_scores21 = all_roc_auc_scores21['Mean ROC AUC']
mean_roc_auc_scores22 = all_roc_auc_scores22['Mean ROC AUC']
mean_roc_auc_scores23 = all_roc_auc_scores23['Mean ROC AUC']

# Create a list of labels for the legend (optional)
labels = ['Normal', 'Pre-activation', 'Skip connection']

# Create a list of colors for the lines
colors = ['b', 'g', 'r']

# Create a figure and axis
plt.figure(figsize=(8, 6))
ax = plt.subplot(111)

# Plot the mean ROC AUC scores for each dataset
ax.plot(range(1, len(mean_roc_auc_scores21) + 1), mean_roc_auc_scores21, marker='o', linestyle='-', color=colors[0])
ax.plot(range(1, len(mean_roc_auc_scores22) + 1), mean_roc_auc_scores22, marker='*', linestyle='-', color=colors[1])
ax.plot(range(1, len(mean_roc_auc_scores23) + 1), mean_roc_auc_scores23, marker='^', linestyle='-', color=colors[2])

# Add labels and legend
plt.title('Mean ROC AUC Scores',fontsize=15)
plt.xlabel('Subject ID',fontsize=15)
plt.ylabel('Mean ROC AUC',fontsize=15)
plt.legend(labels)
plt.xticks(fontsize=15)  # Increase the font size of x-axis ticks
plt.yticks(fontsize=15)  # 
# Show the plot
#plt.grid(True)
plt.show()


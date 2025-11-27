# Air Quality Forecasting using Hybrid Deep Learning Model (CNN + LSTM + Transformer)

This project implements a hybrid deep learning model for forecasting air quality based on historical data of pollutants such as PM2.5, O3, and NO2. The methodology combines Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and Transformer encoders, optimized using Bayesian Optimization for better accuracy in multi-pollutant air quality forecasting.

## Project Structure

The project consists of the following Python files:

- *data_preprocessing.py*: This file handles the data preprocessing steps, including imputation, outlier detection, normalization, and transformation.
- *model_architecture.py*: Contains the model architecture definitions, including the CNN (AMS-CNN), LSTM, and Transformer encoder components, as well as their integration.
- *train_model.py*: Responsible for training the model using the preprocessed data, and saving the trained model for future use.
- *bayesian_optimization.py*: Implements Bayesian optimization for tuning hyperparameters of the LSTM part of the model.
- *evaluate_model.py*: Evaluates the trained model using performance metrics such as RMSE and R², and visualizes the results.

## Requirements

To run this project, you need Python 3.10 or higher and the following libraries:

- TensorFlow 2.13
- PyTorch 2.3
- Hyperopt
- Scikit-learn 1.5
- Pandas
- Numpy
- Matplotlib

You can install all required dependencies using the following command:

bash
pip install -r requirements.txt


## Setup and Running the Project

### Step 1: *Prepare the Dataset*

The dataset should contain time-series data for pollutants (PM2.5, O3, and NO2). You can use the air quality dataset from the U.S. EPA or any similar dataset.

Make sure the data is in CSV format.

### Step 2: *Preprocessing the Data*

Run the data_preprocessing.py file to clean the dataset. This script performs the following tasks:
1. *Imputation of Missing Values* using a moving average technique.
2. *Outlier Removal* based on the Interquartile Range (IQR) method.
3. *Log Transformation* to handle skewed data.
4. *Z-Score Normalization* to standardize the features.

Run the preprocessing script:

bash
python data_preprocessing.py


### Step 3: *Model Building*

Run the model_architecture.py file to define the model architecture. This script defines the *Adaptive Multi-Scale CNN (AMS-CNN)* for spatial feature extraction, *LSTM* for capturing temporal dependencies, and *Transformer encoder* for long-range contextual relationships. It combines these components into a hybrid model for forecasting air quality.

Run the model architecture script:

bash
python model_architecture.py


### Step 4: *Training the Model*

Run the train_model.py file to train the model using the preprocessed data. This script:
- Trains the model using the preprocessed and normalized data.
- Saves the trained model as air_quality_forecasting_model.h5.

Run the training script:

bash
python train_model.py


### Step 5: *Hyperparameter Tuning (Optional)*

To optimize the *LSTM hyperparameters, run the bayesian_optimization.py file. This script uses **Bayesian Optimization* to fine-tune hyperparameters such as the learning rate, number of layers, and dropout rate of the LSTM model for better forecasting accuracy.

Run the Bayesian optimization script:

bash
python bayesian_optimization.py


### Step 6: *Evaluating the Model*

After training the model, run the evaluate_model.py file to evaluate its performance:
- It calculates performance metrics such as *RMSE, **MAE, **R², and **MAPE*.
- It visualizes the comparison of *Actual vs Predicted* pollutant concentrations and overlays of time-series predictions.

Run the evaluation script:

bash
python evaluate_model.py


### Performance Metrics:
- *RMSE* (Root Mean Squared Error)
- *R²* (Coefficient of Determination)
- *MAE* (Mean Absolute Error)
- *MAPE* (Mean Absolute Percentage Error)

### Example Results:
bash
RMSE: 3.78
R²: 0.97


### Visualizations:
The evaluate_model.py script generates the following plots:
- *Actual vs Predicted*: Visual comparison of the predicted and actual pollutant concentrations.
- *Time-Series Overlay*: Visual comparison of observed and predicted pollutant levels over time.

## Hyperparameter Tuning

The hyperparameters for the *LSTM* model are optimized using *Bayesian Optimization* via the bayesian_optimization.py script. The key parameters tuned include:

- *Learning Rate*
- *Number of LSTM Layers*
- *Dropout Rate*

### Example Output for Hyperparameter Optimization:
bash
Best Hyperparameters: {'num_layers': 3, 'units': 64}


## Results Summary

The model significantly improves prediction accuracy, with the proposed architecture (CNN + LSTM + Transformer) outperforming traditional methods and standalone models. 

- The *R²* value for the proposed model exceeds *0.95* for all pollutants.
- *RMSE* and *MAE* are reduced by *12.6% to 24.8%* compared to baseline models.

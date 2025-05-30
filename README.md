# ML-Based Racing Car Controller

This project implements a machine learning-based controller for the TORCS (The Open Racing Car Simulator) racing game using the SCRC (Simple Client for Race Cars) protocol. The controller uses XGBoost models trained on collected racing data to make real-time driving decisions.

## Features

- Machine learning-based control of racing car
- Real-time sensor data processing
- Predictive control for:
  - Acceleration
  - Braking
  - Steering
  - Gear shifting
- Data collection and logging capabilities
- Model training pipeline

## Prerequisites

- Python 3.x
- TORCS racing simulator
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - joblib

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install numpy pandas scikit-learn xgboost joblib
```

## Project Structure

- `driver.py`: Main driver implementation with ML-based control logic
- `train_model_supervised.py`: Script for training the ML models
- `dataset.csv`: Collected racing data for training
- Model files (generated after training):
  - `xgb_model_accel.pkl`
  - `xgb_model_brake.pkl`
  - `xgb_model_steer.pkl`
  - `xgb_model_gear.pkl`
  - `xgb_scaler.pkl`

## Usage

### Training the Models

1. First, collect racing data by running the simulator with the driver
2. Run the training script:
```bash
python train_model_supervised.py
```

This will:
- Load and preprocess the collected data
- Train separate XGBoost models for acceleration, braking, steering, and gear control
- Save the trained models and scaler

### Running the Driver

1. Start the TORCS simulator
2. Run the pyclient:
```bash
python pyclient.py
```

The driver will:
- Load the trained models
- Process real-time sensor data
- Make control decisions using the ML models
- Log racing data for future training

## Model Details

The system uses four separate XGBoost regressors for different control aspects:

1. **Acceleration Model**: Controls throttle input (0-1)
2. **Brake Model**: Controls brake input (0-1)
3. **Steering Model**: Controls steering angle (-1 to 1)
4. **Gear Model**: Controls gear selection (-1 to 6)

Input features include:
- Car speed (X, Y, Z components)
- Track position and angle
- Track sensor readings (19 sensors)
- Opponent sensor readings (36 sensors)

## Data Collection

The system automatically collects racing data including:
- Car state (speed, position, angle)
- Track information
- Opponent positions
- Control inputs
- Performance metrics

Data is saved to `dataset.csv` for future model training.

## Performance

The models are evaluated using R² scores on a test set. Typical performance metrics are printed during training.



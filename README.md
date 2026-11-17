# Industrial Predictive Maintenance | Machine Learning

A data science project focused on predictive maintenance, operational data analysis and machine learning applied to industrial equipment.

## Project Overview

This solution uses machine learning techniques to analyze industrial operating parameters and estimate equipment failure risk.

## Technologies

- Python
- Pandas & NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Matplotlib

## Key Features

- Synthetic industrial dataset generation
- Data analysis and preprocessing
- Predictive modeling with Random Forest
- Model evaluation with separate train, validation and test sets
- Failure-focused threshold selection (target recall) and false-alarm reporting
- Interactive monitoring dashboard
- Feature importance analysis
- Failure risk estimation

## Dataset

The project uses 8,000 synthetically generated records containing:

- Temperature
- Vibration
- RPM
- Pressure
- Operating hours
- Failure indicator

**Note:** The dataset is synthetic and intended exclusively for educational and demonstration purposes.

## Running the Project

```bash
pip install -r requirements.txt
python3 src/generate_data.py
python3 src/train.py
python3 -m streamlit run src/dashboard.py
```

The alert threshold is selected on the validation set to target 80% recall, choosing the highest threshold that reaches that target. The test set remains separate and reports detected failures, missed failures and false alarms. This target is a demonstration setting, not an operational recommendation. Results from the synthetic data do not establish real-world performance.

## Project Structure

```text
industrial-predictive-maintenance/
├── data/
├── models/
├── reports/
├── src/
│   ├── generate_data.py
│   ├── train.py
│   └── dashboard.py
├── README.md
└── requirements.txt
```

## Author

Data Science | Python | Machine Learning | Industrial Analytics

## Dashboard Preview

### 1. Main Dashboard

![Main Dashboard](images/dashboard1.png)

### 2. Industrial Monitoring

![Industrial Monitoring](images/dashboard2.png)

### 3. Predictive Analysis

![Predictive Analysis](images/dashboard3.png)

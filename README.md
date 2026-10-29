# INSURANCE AI — Medical Insurance Cost & Risk Predictor

## Business Problem
An insurance company wants to estimate customers' medical insurance charges and identify customers who may have higher insurance costs.

## Machine Learning Models
- **Linear Regression:** Predicts medical insurance charges.
- **Logistic Regression:** Classifies customers into normal-cost and high-cost groups.
- **R² Score:** Evaluates the regression model.
- **Accuracy:** Evaluates the classification model.

The high-cost label is based on the median insurance charges in the training data. A value of 0 means below the median, and 1 means at or above the median. This is a simple project definition, not an industry standard.

## How to Run the Project
1. Keep `insurance.csv` in the project folder.
2. Open a terminal in the project folder.
3. Install the required libraries:
   `pip install -r requirements.txt`
4. Train the models:
   `python train_models.py`
5. Start the website:
   `uvicorn main:app --reload`
6. Open `http://127.0.0.1:8000` in your browser.

The training script creates the model files and `threshold.txt` 
The training script creates the trained model files and their corresponding feature-column files, which are used by the FastAPI backend to generate predictions.
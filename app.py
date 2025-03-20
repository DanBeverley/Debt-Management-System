"""
Flask Web Application for AI Debt Management System

Module manages the user interface and ties the components together.
Includes routes for rendering homepage, processing user input, and displaying recommendations
"""

import os
import logging
from flask import Flask, request, render_template, redirect, url_for, flash

from model.model_prediction import ModelPredictor

app = Flask(__name__)
app.secret_key = 'supersecretkey'

logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler('app.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

CONFIG = {"model_path": os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl'),
          "preprocessor_path": os.path.join(os.path.dirname(__file__), 'models', 'preprocessor.pkl'),
          "model_prediction_config": {"model_type":"classifier"}}

model_predictor = ModelPredictor(model_path=CONFIG["model_path"],
                                 preprocessor_path = CONFIG["preprocessor_path"],
                                 config = CONFIG["model_prediction_config"])

@app.route('/')
def index():
    """
    Render homepage where user can input their debt information
    Returns : Rendered HTML template
    """
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    """
    Process user input and display results
    Returns : Rendered HTML template with predictions and recommendations
    """
    try:
        user_input = {
            'income': float(request.form['income']),
            'expenses': float(request.form['expenses']),
            'credit_score': int(request.form['credit_score']),
            'debts': [
                {
                    'amount': float(request.form['debt1_amount']),
                    'interest_rate': float(request.form['debt1_interest_rate']),
                    'minimum_payment': float(request.form['debt1_minimum_payment'])
                },
                {
                    'amount': float(request.form['debt2_amount']),
                    'interest_rate': float(request.form['debt2_interest_rate']),
                    'minimum_payment': float(request.form['debt2_minimum_payment'])
                }
            ]
        }
        logger.info(f"Received user input: {user_input}")

        preprocessed_data = model_predictor.preprocess_user_data(user_input)

        recommendations = model_predictor.predict_debt_strategy(preprocessed_data)

        return render_template('results.html', recommendations = recommendations)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        flash(f"An error occured: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug = True)

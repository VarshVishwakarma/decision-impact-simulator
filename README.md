# decision-impact-simulator
The Decision Impact Simulator is an interactive Streamlit application designed for data scientists and business stakeholders. It demonstrates a critical concept in applied machine learning: optimizing for business value often requires different decisions than optimizing for pure model accuracy.

Using a synthetic credit card fraud detection scenario, this tool allows users to visualize how adjusting the classification probability threshold impacts financial outcomes, balancing the cost of missing fraud (False Negatives) against the cost of interrupting legitimate users (False Positives).

Key Features

Interactive Simulation: dynamically adjust the decision threshold via a slider to see immediate impacts on model performance.

Business-Centric Metrics: Focuses on financial cost rather than abstract metrics like AUC or Accuracy.

Real-Time Visualization: Displays a Cost Sensitivity Curve to visually identify the optimal operating point for the model.

Customizable Parameters: Users can input specific costs for False Positives and False Negatives to match different business contexts.

Educational Focus: clearly illustrates the trade-off between "aggressive" and "conservative" model behaviors.

Technical Details

Data Source: Generates synthetic imbalanced data (using sklearn.datasets.make_classification) to mimic real-world fraud scenarios (approx. 10% fraud rate).

Model: Uses a standard Logistic Regression model from Scikit-Learn.

Logic: Calculates the confusion matrix (TP, TN, FP, FN) at 101 different threshold points to derive the total estimated cost curve.

Installation & Usage

Prerequisites

Ensure you have Python 3.8+ installed.

1. Clone the repository

git clone [https://github.com/yourusername/decision-impact-simulator.git](https://github.com/yourusername/decision-impact-simulator.git)
cd decision-impact-simulator


2. Install dependencies

pip install streamlit scikit-learn matplotlib numpy


3. Run the application

streamlit run decision_simulator.py


The application will launch in your default web browser (usually at http://localhost:8501).

Project Structure

decision_simulator.py: The main application script containing data generation, model training, and the Streamlit UI logic.

Why this matters

In many real-world applications (fraud detection, medical diagnosis, manufacturing QA), the cost of errors is asymmetric. A False Negative might cost $100, while a False Positive costs only $5. This tool helps visualize why the default decision threshold of 0.5 is rarely the most profitable choice in these scenarios.

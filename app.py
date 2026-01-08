import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Set page configuration for a wider layout
st.set_page_config(page_title="Decision Impact Simulator", layout="wide")

@st.cache_data
def get_model_predictions():
    """
    Generates synthetic data, trains a model, and returns test set predictions.
    Cached to prevent re-training on every UI interaction.
    """
    # Generate synthetic data: 2000 samples, 10% fraud (class 1)
    X, y = make_classification(
        n_samples=2000, 
        n_features=10, 
        n_classes=2, 
        weights=[0.9, 0.1], 
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Get probabilities for Class 1 (Fraud)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    return y_test, y_probs

def calculate_costs(y_true, y_probs, threshold, cost_fp, cost_fn):
    """
    Calculates confusion matrix and total business cost for a specific threshold.
    """
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    return total_cost, fp, fn, tp

def main():
    st.title("Decision Impact Simulator")
    st.markdown("""
    **Goal:** Optimize business value by adjusting the decision threshold.
    Move the slider to see how shifting the 'decision line' impacts False Positives, False Negatives, and overall Cost.
    """)

    # 1. Load Data
    y_test, y_probs = get_model_predictions()

    # 2. Sidebar Controls
    st.sidebar.header("Configuration")
    
    threshold = st.sidebar.slider(
        "Decision Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.01,
        help="Probability cut-off. If prob > threshold, predict Fraud."
    )

    st.sidebar.subheader("Business Costs")
    cost_fp = st.sidebar.number_input("Cost of False Positive ($)", value=5.0, step=1.0)
    cost_fn = st.sidebar.number_input("Cost of False Negative ($)", value=100.0, step=10.0)

    # 3. Calculate Metrics for Selected Threshold
    current_cost, fp, fn, tp = calculate_costs(y_test, y_probs, threshold, cost_fp, cost_fn)

    # 4. Display Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("False Positives", f"{fp}", help="Legitimate users interrupted")
    col2.metric("False Negatives", f"{fn}", help="Fraud missed")
    col3.metric("True Positives", f"{tp}", help="Fraud caught")
    col4.metric("Total Cost", f"${current_cost:,.2f}", delta_color="inverse")

    # 5. Contextual Explanation
    st.info(f"""
    **Impact Analysis:** At a threshold of **{threshold:.2f}**, you are accepting **${fn * cost_fn:,.0f}** in fraud losses to keep user interruption costs at **${fp * cost_fp:,.0f}**.
    {'The model is acting aggressively to catch fraud.' if threshold < 0.2 else 'The model is acting conservatively to avoid bothering users.' if threshold > 0.8 else 'The model is balanced.'}
    """)

    # 6. Cost Curve Visualization
    st.subheader("Cost Sensitivity Curve")
    
    # Calculate cost for ALL thresholds to render the curve
    thresholds_range = np.linspace(0.0, 1.0, 101)
    costs = []
    
    for t in thresholds_range:
        c, _, _, _ = calculate_costs(y_test, y_probs, t, cost_fp, cost_fn)
        costs.append(c)
        
    # Find optimal point
    min_cost = min(costs)
    opt_threshold = thresholds_range[np.argmin(costs)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds_range, costs, label='Total Business Cost', color='#4b7bec', linewidth=2)
    
    # Marker for current user selection
    ax.scatter([threshold], [current_cost], color='red', s=100, label=f'Current Setting ({threshold})', zorder=5)
    
    # Marker for optimal point
    ax.axvline(x=opt_threshold, color='green', linestyle='--', alpha=0.6, label=f'Optimal ({opt_threshold:.2f})')
    
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Estimated Cost ($)')
    ax.set_title('Business Cost vs. Decision Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# Placeholder for your six months of data
# Assuming X1, X2, ..., X6 are the feature matrices for six months, and y1, y2, ..., y6 are the targets
data = {"month1": (X1, y1), "month2": (X2, y2), "month3": (X3, y3),
        "month4": (X4, y4), "month5": (X5, y5), "month6": (X6, y6)}

# Step 1: Perform RFE with LightGBM and regularization for each month's data
feature_rankings = {}
num_features_total = X1.shape[1]  # Total number of features

# Set L1 and L2 regularization parameters for LightGBM
lambda_l1 = 0.1  # L1 regularization (Lasso)
lambda_l2 = 0.1  # L2 regularization (Ridge)

for month, (X, y) in data.items():
    model = LGBMClassifier(lambda_l1=lambda_l1, lambda_l2=lambda_l2, random_state=42)
    rfe = RFE(model, n_features_to_select=1, step=1)  # Rank all features
    rfe.fit(X, y)
    feature_rankings[month] = rfe.ranking_  # Store rankings (1=best, higher=worse)

# Step 2: Convert rankings into a DataFrame for easier manipulation
rankings_df = pd.DataFrame(feature_rankings)
rankings_df.index = [f"Feature_{i}" for i in range(1, num_features_total + 1)]

# Step 3: Perform feature selection for each number of features to select
results = {}

for num_features in range(2, num_features_total + 1):
    # For each month, get the top `num_features` based on rankings
    selected_features_per_month = [
        set(rankings_df[month].nsmallest(num_features).index) for month in rankings_df.columns
    ]
    
    # Aggregate rankings across months (average ranking for each feature)
    avg_rankings = rankings_df.mean(axis=1)
    
    # Select top `num_features` features based on aggregated scores
    final_features = avg_rankings.nsmallest(num_features).index.tolist()
    results[num_features] = final_features

# Step 4: Display results
for num_features, selected_features in results.items():
    print(f"Number of Features: {num_features}, Selected Features: {selected_features}")
  

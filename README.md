# ✈ Holiday Booking Prediction

This project aims to build a machine learning model that predicts whether a customer will complete a holiday booking based on their behavior and flight-related preferences. The analysis was done as part of a business case to help airlines **proactively identify high-value customers before they travel**.

##  Business Context

Airlines face increasing pressure to act *before* customers arrive at the airport. With predictive modeling, they can target those most likely to book. The goal of this project was to:

- **Analyze** customer and booking behavior
- **Engineer features** that boost prediction power
- **Train and evaluate** a model using real customer data
- **Interpret** feature contributions and business implications

##  Dataset Summary

The dataset included flight details (day, hour, route), customer preferences (meals, seat, baggage), and behavioral attributes (booking lead time, length of stay, number of passengers).

The target variable was `booking_complete` (1 = yes, 0 = no).

## 🛠 Approach

**Exploratory Steps:**
- Class imbalance detected (1 = ~15% of data)
- Feature distributions visualized
- New features created (e.g. weekend flag, passenger group size)

**Preprocessing:**
- One-hot encoding for categorical features  
- Custom feature engineering  
- Oversampling using `resample()` to balance classes  
- Feature selection via `RandomForest.feature_importances_`

**Modeling:**
- Algorithm: `RandomForestClassifier`
- Evaluation: Cross-validation with `F1`, `ROC AUC`, and classification report
- Threshold tuning for improving recall

##  Key Metrics

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | ~0.74     |
| F1 (class 1) | ~0.34     |
| ROC AUC      | ~0.68     |

##  Top 5 Predictive Features

- `purchase_lead` – how far in advance the flight is booked  
- `flight_hour` – hour of departure  
- `length_of_stay` – trip duration  
- `flight_duration` – total time in air  
- `num_passengers` – likely group size

> These features can help **target specific customer segments** before they complete bookings.

##  Visuals Included

- ROC curve  
- Precision-Recall curve  
- PCA-based 2D model output map  
- Bar chart of top feature importances  


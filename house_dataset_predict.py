from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

data = pd.read_csv("synthetic_large_dataset.csv")

data_encoded = pd.get_dummies(data, columns=["Neighborhood"], drop_first=True)

# print(data_encoded)

X = data_encoded.drop("Price", axis=1)
y = data_encoded["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

feature_names = X.columns

# Get feature values from user input
user_input = []
for feature_name in feature_names:
    value = input(f"Enter value for {feature_name}: ")
    user_input.append(value)

# Convert user input to a DataFrame
user_input_df = pd.DataFrame([user_input], columns=feature_names)

# Make a prediction using the trained model
predicted_price = model.predict(user_input_df)[0]
formatted_price = locale.currency(predicted_price, grouping=True)
print("Predicted Price:", formatted_price)

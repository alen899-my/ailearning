# ===============================
# Import basic libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# ===============================
# Load dataset
# ===============================
# Make sure openpyxl is installed for Excel files
# pip install openpyxl
data = pd.read_excel("flight_price.xlsx")

# Preview dataset
print(data.head())
print(data.tail())

# ===============================
# Basic info
# ===============================
print(data.info())
print(data.describe())

# ===============================
# Handle Date_of_Journey
# ===============================
data["Date"] = data["Date_of_Journey"].str.split("/").str[0].astype(int)
data["Month"] = data["Date_of_Journey"].str.split("/").str[1].astype(int)
data["Year"] = data["Date_of_Journey"].str.split("/").str[2].astype(int)

# Drop original Date_of_Journey
data.drop("Date_of_Journey", axis=1, inplace=True)

# ===============================
# Handle Arrival_Time
# ===============================
data["Arrival_Time"] = data["Arrival_Time"].apply(lambda x: x.split(" ")[0])
data["Arrival_hour"] = data["Arrival_Time"].str.split(":").str[0].astype(int)
data["Arrival_minute"] = data["Arrival_Time"].str.split(":").str[1].astype(int)

# Drop original Arrival_Time
data.drop("Arrival_Time", axis=1, inplace=True)

# ===============================
# Handle Dep_Time
# ===============================
data["Dept_hour"] = data["Dep_Time"].str.split(":").str[0].astype(int)
data["Dept_minute"] = data["Dep_Time"].str.split(":").str[1].astype(int)

# Drop original Dep_Time
data.drop("Dep_Time", axis=1, inplace=True)

# ===============================
# Handle Total_Stops
# ===============================
# Map textual stops into numbers
data["Total_Stops"] = data["Total_Stops"].map(
    {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4, np.nan: 1}
)

# Drop Route column
data.drop("Route", axis=1, inplace=True)

# ===============================
# Handle Duration
# ===============================
data["Minute_duration"] = data["Duration"].str.extract(r"(\d+)m").fillna(0).astype(int)
data["Hour_duration"] = data["Duration"].str.extract(r"(\d+)h").fillna(0).astype(int)

# Optional: create total duration in minutes
data["Total_duration_mins"] = data["Hour_duration"] * 60 + data["Minute_duration"]

# ===============================
# Encode Categorical Features
# ===============================
encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[["Airline", "Source", "Destination"]]).toarray()

# Convert encoded values into DataFrame
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# Concatenate with original dataset
final_data = pd.concat([data, encoded_df], axis=1)

# Drop original categorical columns
final_data.drop(
    columns=["Airline", "Source", "Destination", "Duration", "Additional_Info"],
    inplace=True,
)

# ===============================
# Final Dataset Overview
# ===============================
print(final_data.info())
print(final_data.describe())
print(final_data.head())

import pandas as pd

# Read your CSV file
data = pd.read_csv("data.csv")

# Show the first few rows (optional)
print("Before cleaning:")
print(data.head())

# Replace '-' with '/' in the Date column
# (handles both text and mixed types)
data['date'] = data['date'].astype(str).str.replace('-', '/', regex=False)

# Convert to proper datetime format (optional but recommended)
data['date'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=False)

# Show cleaned data
print("\nAfter cleaning:")
print(data.head())

# Save cleaned data back to CSV
data.to_csv("data_cleaned.csv", index=False)

print("\n✅ Cleaned data saved as 'data_cleaned.csv'")

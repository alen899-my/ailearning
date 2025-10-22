# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------- STEP 1: Load dataset ----------------
df = pd.read_csv("data.csv")

# ---------------- STEP 2: Encode categorical variables ----------------
label_encoders = {}
for col in ["Person1_Gender", "Person2_Gender", "Same_Last_Name", 
            "Marital_Status", "Household_Role_P1", "Household_Role_P2"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target label
le_target = LabelEncoder()
df["Relationship_Label"] = le_target.fit_transform(df["Relationship_Label"])

# ---------------- STEP 3: Features (X) and Target (y) ----------------
X = df.drop("Relationship_Label", axis=1)
y = df["Relationship_Label"]

# ---------------- STEP 4: Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- STEP 5: Train Model ----------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------- STEP 6: Evaluate ----------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

# ---------------- STEP 7: User Input ----------------
print("\n--- Enter New Data for Prediction ---")

p1_age = int(input("Enter Person1 Age: "))
p2_age = int(input("Enter Person2 Age: "))
p1_gender = input("Enter Person1 Gender (Male/Female): ")
p2_gender = input("Enter Person2 Gender (Male/Female): ")
same_last_name = input("Do they have the same last name? (Yes/No): ")
marital_status = input("Enter Marital Status (Single/Married/Divorced/Widowed): ")
role_p1 = input("Enter Household Role of Person1 (Parent/Child/Spouse/Sibling/Grandparent/Other): ")
role_p2 = input("Enter Household Role of Person2 (Parent/Child/Spouse/Sibling/Grandparent/Other): ")

# ---------------- STEP 8: Role Mapping (Safe) ----------------
def map_role(role):
    role = role.strip().capitalize()
    valid_roles = ["Child", "Parent", "Spouse", "Sibling", "Grandparent", "Other"]
    return role if role in valid_roles else "Other"

role_p1 = map_role(role_p1)
role_p2 = map_role(role_p2)

# ---------------- STEP 9: Create dataframe for new input ----------------
user_data = pd.DataFrame([{
    "Person1_Age": p1_age,
    "Person2_Age": p2_age,
    "Person1_Gender": p1_gender,
    "Person2_Gender": p2_gender,
    "Same_Last_Name": same_last_name,
    "Marital_Status": marital_status,
    "Household_Role_P1": role_p1,
    "Household_Role_P2": role_p2
}])

# ---------------- STEP 10: Encode categorical inputs ----------------
for col in ["Person1_Gender", "Person2_Gender", "Same_Last_Name", 
            "Marital_Status", "Household_Role_P1", "Household_Role_P2"]:
    user_data[col] = label_encoders[col].transform(user_data[col])

# ---------------- STEP 11: Predict ----------------
prediction = model.predict(user_data)
predicted_label = le_target.inverse_transform(prediction)[0]

# ---------------- STEP 12: Refine Output with Gender ----------------
def refine_label(label, g1, g2):
    g1, g2 = g1.lower(), g2.lower()
    
    if label == "Parent-Child":
        return "Father-Child" if g1 == "male" else "Mother-Child"
    elif label == "Child-Parent":
        return "Son-Parent" if g1 == "male" else "Daughter-Parent"
    elif label == "Siblings":
        if g1 == "male" and g2 == "male":
            return "Brothers"
        elif g1 == "female" and g2 == "female":
            return "Sisters"
        else:
            return "Brother-Sister"
    elif label == "Spouse":
        return "Husband-Wife"
    elif label == "Grandparent-Grandchild":
        return "Grandfather-Grandchild" if g1 == "male" else "Grandmother-Grandchild"
    return label

final_output = refine_label(predicted_label, p1_gender, p2_gender)

print("\nPredicted Relationship:", final_output)

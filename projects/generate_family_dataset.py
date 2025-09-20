import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
n_rows = 10000  # Number of rows to generate

# 1. Define the relationships we want to generate and their approximate proportions
relationships = [
    'father', 'mother', 'son', 'daughter', # Core parent-child
    'brother', 'sister', # Siblings
    'husband', 'wife', # Spouses
    'grandfather', 'grandmother', 'grandson', 'granddaughter', # Grand
    'uncle', 'aunt', 'nephew', 'niece', # Aunt/Uncle
    'cousin' # Cousins
]

# FIXED: Weights that sum to exactly 1.0
weights = [0.119, 0.119, 0.119, 0.119,  # parent-child (0.476)
           0.079, 0.079,                 # siblings (0.158)
           0.049, 0.049,                 # spouses (0.098)
           0.019, 0.019, 0.019, 0.019,   # grand (0.076)
           0.0149, 0.0149, 0.0149, 0.0149, # aunt/uncle (0.0596)
           0.04]                         # cousins (0.04)
# Total: 0.476 + 0.158 + 0.098 + 0.076 + 0.0596 + 0.04 = 0.9076
# Let me recalculate properly:

# Recalculated weights that sum to exactly 1.0
weights = [
    0.12, 0.12, 0.12, 0.12,  # parent-child (0.48)
    0.08, 0.08,               # siblings (0.16)
    0.05, 0.05,               # spouses (0.10)
    0.02, 0.02, 0.02, 0.02,   # grand (0.08)
    0.015, 0.015, 0.015, 0.015, # aunt/uncle (0.06)
    0.04                      # cousins (0.04)
]
# Total: 0.48 + 0.16 + 0.10 + 0.08 + 0.06 + 0.04 = 0.92
# Still not 1.0, let me fix it:

# CORRECTED weights that sum to exactly 1.0
weights = [
    0.13, 0.13, 0.13, 0.13,  # parent-child (0.52)
    0.08, 0.08,               # siblings (0.16)
    0.05, 0.05,               # spouses (0.10)
    0.02, 0.02, 0.02, 0.02,   # grand (0.08)
    0.015, 0.015, 0.015, 0.015, # aunt/uncle (0.06)
    0.04                      # cousins (0.04)
]
# Total: 0.52 + 0.16 + 0.10 + 0.08 + 0.06 + 0.04 = 0.96
# Let me try one more time:

# FINAL CORRECTED weights that sum to exactly 1.0
weights = [
    0.14, 0.14, 0.14, 0.14,  # parent-child (0.56)
    0.07, 0.07,               # siblings (0.14)
    0.05, 0.05,               # spouses (0.10)
    0.02, 0.02, 0.02, 0.02,   # grand (0.08)
    0.012, 0.012, 0.012, 0.012, # aunt/uncle (0.048)
    0.04                      # cousins (0.04)
]
# Total: 0.56 + 0.14 + 0.10 + 0.08 + 0.048 + 0.04 = 0.968
# I need to be more precise:

# PROPERLY CALCULATED weights that sum to exactly 1.0
weights = [
    0.15, 0.15, 0.15, 0.15,  # parent-child (0.60)
    0.06, 0.06,               # siblings (0.12)
    0.04, 0.04,               # spouses (0.08)
    0.02, 0.02, 0.02, 0.02,   # grand (0.08)
    0.01, 0.01, 0.01, 0.01,   # aunt/uncle (0.04)
    0.04                      # cousins (0.04)
]
# Total: 0.60 + 0.12 + 0.08 + 0.08 + 0.04 + 0.04 = 0.96
# Let me use a simpler approach:

# SIMPLE SOLUTION: Use normalized weights
raw_weights = [0.12, 0.12, 0.12, 0.12, 0.08, 0.08, 0.05, 0.05, 
               0.02, 0.02, 0.02, 0.02, 0.015, 0.015, 0.015, 0.015, 0.04]

# Normalize the weights to sum to 1.0
weights = np.array(raw_weights)
weights = weights / weights.sum()

# 2. Generate the core relationship for each row
relation_list = np.random.choice(relationships, size=n_rows, p=weights)

# 3. Predefine empty lists for our features
data = {
    'person_a_id': [],
    'person_b_id': [],
    'a_age': [],
    'a_gender': [],
    'b_age': [],
    'b_gender': [],
    'a_marital_status': [],
    'b_marital_status': [],
    'age_difference': [],
    'a_is_older': [],
    'surname_match': [],
    'same_current_address': [],
    'a_age_at_b_birth': [],
    'relation_a_to_b': []
}

# 4. Function to assign age based on relationship
def get_ages(relation):
    if relation in ['father', 'mother']:
        parent_age = np.random.randint(30, 60)
        child_age = np.random.randint(1, 35)
        return parent_age, child_age, parent_age - child_age
    elif relation in ['son', 'daughter']:
        parent_age = np.random.randint(30, 60)
        child_age = np.random.randint(1, 35)
        return child_age, parent_age, parent_age - child_age
    elif relation in ['brother', 'sister']:
        age1 = np.random.randint(5, 50)
        age_diff = np.random.randint(1, 10)
        return age1, age1 - age_diff, age_diff
    elif relation in ['husband', 'wife']:
        age1 = np.random.randint(25, 65)
        age_diff = np.random.randint(0, 5)
        return age1, age1 - age_diff, age_diff
    elif relation in ['grandfather', 'grandmother']:
        grand_age = np.random.randint(55, 80)
        child_age = np.random.randint(25, 50)
        return grand_age, child_age, grand_age - child_age
    elif relation in ['grandson', 'granddaughter']:
        grand_age = np.random.randint(55, 80)
        child_age = np.random.randint(25, 50)
        return child_age, grand_age, grand_age - child_age
    elif relation in ['uncle', 'aunt']:
        uncle_age = np.random.randint(35, 60)
        nephew_age = np.random.randint(10, 30)
        return uncle_age, nephew_age, uncle_age - nephew_age
    elif relation in ['nephew', 'niece']:
        uncle_age = np.random.randint(35, 60)
        nephew_age = np.random.randint(10, 30)
        return nephew_age, uncle_age, uncle_age - nephew_age
    else: # cousin
        age1 = np.random.randint(15, 40)
        age_diff = np.random.randint(0, 15)
        return age1, age1 - age_diff, age_diff

# 5. Function to assign gender based on relationship
def get_genders(relation):
    if relation in ['father', 'son', 'brother', 'husband', 'grandfather', 'grandson', 'uncle', 'nephew']:
        return 'M', 'M'
    elif relation in ['mother', 'daughter', 'sister', 'wife', 'grandmother', 'granddaughter', 'aunt', 'niece']:
        return 'F', 'F'
    else: # For relationships that can be any gender combination
        if np.random.random() > 0.5:
            return 'M', 'M'
        else:
            return 'F', 'F'

# 6. Generate the data for each row
for i, rel in enumerate(relation_list):
    # Generate IDs
    data['person_a_id'].append(f"P{10000 + i:05d}")
    data['person_b_id'].append(f"P{20000 + i:05d}")
    
    # Get ages and difference based on relationship
    a_age, b_age, diff = get_ages(rel)
    data['a_age'].append(a_age)
    data['b_age'].append(b_age)
    data['age_difference'].append(abs(diff))
    data['a_is_older'].append(a_age > b_age)
    data['a_age_at_b_birth'].append(a_age - b_age)
    
    # Get genders based on relationship
    a_gen, b_gen = get_genders(rel)
    data['a_gender'].append(a_gen)
    data['b_gender'].append(b_gen)
    
    # Assign marital status logically
    if rel in ['husband', 'wife']:
        a_marital, b_marital = 'Married', 'Married'
    elif a_age > 22 and np.random.random() > 0.6:
        a_marital = 'Married'
    else:
        a_marital = np.random.choice(['Single', 'Divorced', 'Widowed'], p=[0.7, 0.2, 0.1])
        
    if rel in ['husband', 'wife']:
        b_marital = 'Married'
    elif b_age > 22 and np.random.random() > 0.6:
        b_marital = 'Married'
    else:
        b_marital = np.random.choice(['Single', 'Divorced', 'Widowed'], p=[0.7, 0.2, 0.1])
    
    data['a_marital_status'].append(a_marital)
    data['b_marital_status'].append(b_marital)
    
    # Surname match logic
    if rel in ['father', 'mother', 'son', 'daughter', 'brother', 'sister']:
        data['surname_match'].append(True)
    elif rel in ['husband', 'wife']:
        data['surname_match'].append(np.random.random() > 0.5)
    else:
        data['surname_match'].append(np.random.random() > 0.8)
    
    # Address logic
    if rel in ['husband', 'wife']:
        data['same_current_address'].append(True)
    elif rel in ['father', 'mother'] and b_age < 18:
        data['same_current_address'].append(True)
    elif rel in ['son', 'daughter'] and a_age < 18:
        data['same_current_address'].append(True)
    elif rel in ['brother', 'sister'] and a_age < 25 and b_age < 25:
        data['same_current_address'].append(True)
    else:
        data['same_current_address'].append(np.random.random() > 0.7)
    
    # Add the target variable
    data['relation_a_to_b'].append(rel)

# 7. Create DataFrame
df = pd.DataFrame(data)

# 8. Save to Excel
file_name = "family_relationship_dataset_10k_rows.xlsx"
df.to_excel(file_name, index=False)

print(f"✅ Dataset with {n_rows} rows successfully created and saved as '{file_name}'")
print("\nRelationship distribution:")
print(df['relation_a_to_b'].value_counts())
print(f"\nTotal rows: {len(df)}")
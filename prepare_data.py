import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("employee_performance.csv")

# Target column
target = "KPIs_met_more_than_80"

# Train-test split (80% train, 20% test)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[target]
)

# Save files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✅ train.csv and test.csv created successfully")

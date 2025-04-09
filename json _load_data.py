import pandas as pd

# Read the CSV file
df = pd.read_csv('Coding_dataset_03.csv')

# Extract the 2999th row
row = df.iloc[2998]  # Note that the index is 0-based, so the 2999th row is at index 2998

# Convert the row to a JSON string
json_row = row.to_json()

# Send the JSON string to the /predict endpoint
curl -X POST -H "Content-Type: application/json" -d '{"df": "' + json_row + '"}' http://localhost:5000/predict
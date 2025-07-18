# Instead of reading the xls file we are creating a function here that dumps file to csv_data folder 
# From there on our projects starts

import pandas as pd

# Load the .xls file
xls_path = "raw_data/ai-dev-ticket.xls"
df = pd.read_excel(xls_path)

# Save as .csv in your data folder
csv_path = "csv_data/ticket.csv"
df.to_csv(csv_path, index=False)

print(f"File converted and saved to {csv_path}")

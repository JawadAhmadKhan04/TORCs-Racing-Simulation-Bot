import csv
import re

# Input string
data = "(angle 1.74846e-007)(curLapTime -0.982)(damage 0)(distFromStart 3160.83)(distRaced 0)(fuel 94)(gear 0)(lastLapTime 0)(opponents 200 200 11.1804 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200)(racePos 1)(rpm 942.478)(speedX 0)(speedY 0)(speedZ -0.000605693)(track 10 10.3528 11.547 14.1421 20 29.2381 38.6371 57.5878 114.737 200 57.3684 28.7938 19.3185 14.619 9.99999 7.07106 5.7735 5.17638 5)(trackPos -0.333334)(wheelSpinVel 0 0 0 0)(z 0.345263)(focus -1 -1 -1 -1 -1)"

# Regex to extract key-value pairs
matches = re.findall(r"\((\w+)((?:\s-?\d+\.?\d*e?-?\d*)+)\)", data)

# Process extracted data
parsed_data = {key: value.strip().split() for key, value in matches}

# Expand single values from lists
for key in parsed_data:
    if len(parsed_data[key]) == 1:
        parsed_data[key] = parsed_data[key][0]  # Convert to single value

# CSV file path
csv_file = "race_data.csv"

# Writing to CSV
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Writing header
    writer.writerow(parsed_data.keys())

    # Writing values
    writer.writerow(parsed_data.values())

print(f"Data saved to {csv_file}")

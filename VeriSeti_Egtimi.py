import pandas as pd
import random
from datetime import datetime, timedelta


# Function to generate random data for the dataset
def generate_data(num_rows):
    data = []
    start_date = datetime(2018, 6, 1)

    for _ in range(num_rows):
        # Generate random values
        day = start_date.strftime('%A')
        date = start_date.strftime('%d-%m-%y')
        coded_day = 3  # Static value based on your example
        zone = random.randint(1, 10)
        weather = random.randint(10, 50)
        temperature = random.randint(5, 45)
        traffic = random.randint(1, 5)

        # Add row to data list
        data.append([day, date, coded_day, zone, weather, temperature, traffic])

        # Move to the next day
        start_date += timedelta(days=1)

    return data


# Generate a dataset with 10,000 rows
num_rows = 10000  # Change to 10,000 rows or as many as you need
data = generate_data(num_rows)

# Create a DataFrame
columns = ['Day', 'Date', 'CodedDay', 'Zone', 'Weather', 'Temperature', 'Traffic']
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv('generated_data_10000.csv', index=False)

# Optionally, display the first few rows of the dataset to the user
print(df.head())  # Displays the first 5 rows

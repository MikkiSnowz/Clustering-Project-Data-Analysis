import pandas as pd
import re
import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


household = pd.read_csv("household_vista_2023_2024.csv")

def income_mean(entry):
    """
    Inputs an income group entry as string and outputs the middle yearly
    value as an integer.
    """
    # Handle missing values separately to avoid an error
    if type(entry) == float:
        return 0
    # For the highest (and unbounded) income group, return a special value
    if entry.startswith('$8,000 or more'):
        return 450000
    # Obtain the middle yearly value from the string by splitting it
    left, right = entry.split()[1].split('-')
    left = int(left[2:].replace(',',''))
    right = int(right[1:-1].replace(',',''))
    return int((left + right) // 2)

# Obtain a list of the index of the rows with missing values
missing_mask = household['hhinc_group'].isna()
missing_index = list(household['hhinc_group'][missing_mask].index)

# Put values in 'hhinc_group' with average yearly figure
household['hhinc_group_number'] = household['hhinc_group'].apply(income_mean)

# Imputes missing values in 'hhinc_group_number' with median figure
# Median is more robust to outliers compared to mean
group_median = household['hhinc_group_number'][missing_mask == False].median()
household['hhinc_group_number'].iloc[missing_index] = int(group_median)

# Imputs missing values in 'hhinc_group' with most frequent income category
group_mode = household['hhinc_group'][missing_mask == False].mode()[0]
household['hhinc_group'].iloc[missing_index] = group_mode





# Read the dataset
df_journey = pd.read_csv('journey_to_work_vista_2023_2024.csv')

# Categorize the travel mode for each journey
def modeGroupCategories(mode):
    m = mode.strip().lower()
    if m in {"train", "tram", "school bus", "public bus"}:
        return "Public"
    elif m in {"vehicle driver", "vehicle passenger", "taxi / rideshare", "motorcycle"}:
        return "Private"
    elif m in {"bicycle", "walking", "other"}:
        return "Active"
    else:
        return None
# Create a new column called "travel categories"
df_journey["travelCategory"] = df_journey["main_journey_mode"].apply(modeGroupCategories)

# Calculate Wasted Time for each journey in a separate column called "timeWasted"
df_journey["timeWasted"] = df_journey["journey_elapsed_time"] - df_journey["journey_travel_time"].clip(lower = 0)
def timeWastedCategory(time):
    if time == 0:
        return "0"
    elif 0 < time <= 5:
        return "1-5"
    elif 5 < time <= 10 :
        return "5-10"
    elif 10 < time <= 30:
        return "10-30"
    else:
        return "30+"
# Based on the "timeWasted" column, create a new column to categorize wasted time, and call this column "timeWastedCategory"
df_journey["timeWastedCategory"] = df_journey["timeWasted"].apply(timeWastedCategory)

# Calculate The Number of Stops for each journey in a separate column (starting to count from the column called "destpurp1_desc_1" until "destpurp1_desc_15")
numOfStops = [c for c in df_journey.columns if c.startswith("destpurp1_desc_")]
df_journey["numOfStops"] = df_journey[numOfStops].notna().sum(axis=1)
def stopsCategory(num):
    if num == 0:
        return "0"
    elif num == 1:
        return "0"
    elif num == 2:
        return "1"
    elif num == 3:
        return "2"
    else:
        return "3+"
# Based on the number of stops create another column to categorize stops called "stopsCategory"
df_journey["stopsCategory"] = df_journey["numOfStops"].apply(stopsCategory)

# Calculate the Distance categories for each journey and store them in a new column called "distanceCategory"
def distanceCategory(d):
    if d <= 10:
        return "0-10"
    elif 10 < d <= 20:
        return "10-20"
    elif 20 < d <= 40:
        return "20-40"
    else:
        return "40+"
df_journey["distanceCategory"] = df_journey["journey_distance"].apply(distanceCategory)


#----------------------------------------------------------------------------------------------------------------



# Separate the dataset based off travel types
# add relevant stuff
clean_df_journey = df_journey[["hhid", "travelCategory", "journey_travel_time", "journey_distance", 'timeWasted', 'numOfStops']]
clean_household = household[["hhid", "hhinc_group_number", "hhinc_group"]]

# clean_merge = pd.merge(clean_df_journey, clean_household, left_index=True, right_index=True)
clean_merge = pd.merge(clean_df_journey, clean_household, on='hhid', how='inner')

print(clean_merge)

def filter_journey(database, journeytype):
    # Filter only the entries that contain the relevant journey type
    household_mask = database['travelCategory'].apply(lambda x: journeytype == x)
    hhinc_group_filtered = database[household_mask]

    return hhinc_group_filtered

df_journey_public = filter_journey(clean_merge, "Public")
df_journey_private = filter_journey(clean_merge, "Private")
df_journey_active = filter_journey(clean_merge, "Active")




# group the data together to perform elbow method and find the optimal k value
df_journey_public_numeric = df_journey_public[["journey_travel_time", "journey_distance", 'timeWasted', 'numOfStops',"hhinc_group_number"]]
df_journey_private_numeric = df_journey_private[["journey_travel_time", "journey_distance", 'timeWasted', 'numOfStops',"hhinc_group_number"]]
df_journey_active_numeric = df_journey_active[["journey_travel_time", "journey_distance", 'timeWasted', 'numOfStops',"hhinc_group_number"]]


# group travel categories
normalized_data_public = MinMaxScaler().fit_transform(df_journey_public_numeric)
normalized_data_private = MinMaxScaler().fit_transform(df_journey_private_numeric)
normalized_data_active = MinMaxScaler().fit_transform(df_journey_active_numeric)


def elbow_method(data, title, filename):
    plt.figure(figsize=(8, 6))
    distortions = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.plot(k_range, distortions, 'bx-')
    plt.title(f'The Elbow Method for {title} - Optimal k')
    plt.xlabel('k')
    plt.ylabel('Distortion')       
    plt.savefig(filename)     
    plt.show()

# Public Transport Elbow Method
elbow_method(normalized_data_public, 'Public Transport', 'public_transport_elbow.png')


# Private Transport Elbow Method
elbow_method(normalized_data_private, 'Private Transport', 'private_transport_elbow.png')


# Active Transport Elbow Method
elbow_method(normalized_data_active, 'Active Transport', 'active_transport_elbow.png')


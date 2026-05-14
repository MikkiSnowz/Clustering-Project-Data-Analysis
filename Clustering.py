import pandas as pd
import re
import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.cm as cm



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


def filter_journey(database, journeytype):
    # Filter only the entries that contain the relevant journey type
    household_mask = database['travelCategory'].apply(lambda x: journeytype == x)
    hhinc_group_filtered = database[household_mask]

    return hhinc_group_filtered

df_journey_public = filter_journey(clean_merge, "Public")
df_journey_private = filter_journey(clean_merge, "Private")
df_journey_active = filter_journey(clean_merge, "Active")





# group the data together to perform elbow method and find the optimal k value
df_journey_public_numeric = df_journey_public[["journey_travel_time", "journey_distance","hhinc_group_number"]]
df_journey_private_numeric = df_journey_private[["journey_travel_time", "journey_distance","hhinc_group_number"]]
df_journey_active_numeric = df_journey_active[["journey_travel_time", "journey_distance","hhinc_group_number"]]


# group travel categories
normalized_data_public = MinMaxScaler().fit_transform(df_journey_public_numeric)
normalized_data_private = MinMaxScaler().fit_transform(df_journey_private_numeric)
normalized_data_active = MinMaxScaler().fit_transform(df_journey_active_numeric)




def elbow_method(data, title, filename):
    plt.figure(figsize=(8, 6))
    distortions = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
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


#----------------------------------------------------------------
# clustering part

df_normalized_public = pd.DataFrame(normalized_data_public, columns=df_journey_public_numeric.columns)
df_normalized_private = pd.DataFrame(normalized_data_private, columns=df_journey_private_numeric.columns)
df_normalized_active = pd.DataFrame(normalized_data_active, columns=df_journey_active_numeric.columns)


public_clusters = KMeans(n_clusters=3)
public_clusters.fit(df_normalized_public)

private_clusters = KMeans(n_clusters=4)
private_clusters.fit(df_normalized_private)

active_clusters = KMeans(n_clusters=3)
active_clusters.fit(df_normalized_active)



def plot_kmeans(df, clusters, filename):
    """
    generate a 3d plot given sklearn's kmeans implementation
    """

    fig = plt.figure(figsize=(7, 10))
    num_clusters = len(set(clusters.labels_))
    colors = plt.get_cmap('tab10', num_clusters)
    ax = plt.axes(projection="3d")
    ax.scatter(df['hhinc_group_number'],
               df['journey_travel_time'],
               df['journey_distance'],
               c=[colors(x) for x in clusters.labels_])
           

    ax.set_ylabel('hhinc_group_number')
    ax.set_xlabel('journey_travel_time')
    ax.set_zlabel('journey_distance')
    ax.set_title(f"k = {len(set(clusters.labels_))}")

    plt.savefig(f'{filename}.png')
    plt.show()
    

plot_kmeans(df_normalized_public, public_clusters, 'public_transport_kmeans')
plot_kmeans(df_normalized_private, private_clusters, 'private_transport_kmeans')
plot_kmeans(df_normalized_active, active_clusters, 'active_transport_kmeans')



#hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage

sample_public = df_normalized_public.sample(n=10, random_state=42)
sample_private = df_normalized_private.sample(n=10, random_state=42)
sample_active = df_normalized_active.sample(n=10, random_state=42)

Z_public = linkage(sample_public, method='ward')
Z_private = linkage(sample_private, method='ward')
Z_active = linkage(sample_active, method='ward')

def plot_dendrogram(Z, title, filename):
    
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title(f'Dendrogram for {title}')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig(filename)
    plt.show()

plot_dendrogram(Z_public, 'Public Transport', 'public_transport_dendrogram.png')
plot_dendrogram(Z_private, 'Private Transport', 'private_transport_dendrogram.png')
plot_dendrogram(Z_active, 'Active Transport', 'active_transport_dendrogram.png')

def combined_summary_plot(
    normalized_data, Z, clusters, df, section_name, filename
):
    """
    Plots elbow, dendrogram, and k-means 3D for a section in one PNG.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(18, 5))

    # Elbow plot
    ax1 = fig.add_subplot(1, 3, 1)
    distortions = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(normalized_data)
        distortions.append(kmeans.inertia_)
    ax1.plot(k_range, distortions, 'bx-')
    ax1.set_title(f'Elbow for {section_name}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Distortion')

    # Dendrogram
    ax2 = fig.add_subplot(1, 3, 2)
    dendrogram(Z, ax=ax2)
    ax2.set_title(f'Dendrogram for {section_name}')
    ax2.set_xlabel('Sample index')
    ax2.set_ylabel('Distance')

    # K-means 3D plot
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    num_clusters = len(set(clusters.labels_))
    colors = plt.get_cmap('tab10', num_clusters)
    ax3.scatter(
        df['hhinc_group_number'],
        df['journey_travel_time'],
        df['journey_distance'],
        c=[colors(x) for x in clusters.labels_],
        s=50,
        alpha=0.6
    )
    ax3.set_xlabel('hhinc_group_number')
    ax3.set_ylabel('journey_travel_time')
    ax3.set_zlabel('journey_distance')
    ax3.set_title(f'K-means 3D (k={num_clusters})')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Example usage for Public Transport:
combined_summary_plot(
    normalized_data_public,
    Z_public,
    public_clusters,
    df_normalized_public,
    'Public Transport',
    'public_transport_summary.png'
)
combined_summary_plot(
    normalized_data_private,
    Z_private,
    private_clusters,
    df_normalized_private,
    'Private Transport',
    'private_transport_summary.png'
)
combined_summary_plot(
    normalized_data_active,
    Z_active,
    active_clusters,
    df_normalized_active,
    'Active Transport',
    'active_transport_summary.png'
)



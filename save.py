# group the relevant numeric data together to perform elbow method and find the optimal k value
df_journey_public_numeric = df_journey_public[["journey_travel_time", "journey_distance","hhinc_group_number"]]
df_journey_private_numeric = df_journey_private[["journey_travel_time", "journey_distance","hhinc_group_number"]]
df_journey_active_numeric = df_journey_active[["journey_travel_time", "journey_distance","hhinc_group_number"]]


# normalize the data based on each travel categories
normalized_data_public = MinMaxScaler().fit_transform(df_journey_public_numeric)
normalized_data_private = MinMaxScaler().fit_transform(df_journey_private_numeric)
normalized_data_active = MinMaxScaler().fit_transform(df_journey_active_numeric)


def elbow_method(data, title, filename):
    '''
    Function: This function performs the elbow method by calculating average distance from every centroids
    to every points and then it plots these points showing an elbow shape graph
    '''
    plt.figure(figsize=(8, 6))
    distortions = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k,random_state=42, n_init=10)
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

private_clusters = KMeans(n_clusters=3)
private_clusters.fit(df_normalized_private)

active_clusters = KMeans(n_clusters=4)
active_clusters.fit(df_normalized_active)



def plot_kmeans(df, clusters, filename):
    """
    generate a 3d plot given sklearn's kmeans implementation
    """



    fig = plt.figure(figsize=(15, 15))
    num_clusters = len(set(clusters.labels_))
    colors = plt.get_cmap('tab10', num_clusters)
    ax = plt.axes(projection="3d")
    ax.scatter(df['hhinc_group_number'],
               df['journey_travel_time'],
               df['journey_distance'],
               c=[colors(x) for x in clusters.labels_],
               s=50,
               alpha=0.6)

    ax.set_ylabel('Normalised Household Income', fontsize = 15)
    ax.set_xlabel('Normalised Travel Time', fontsize = 15)
    ax.set_zlabel('Normalised Journey Distance', fontsize = 15)
    ax.set_title(f"K-means 3D (k = {len(set(clusters.labels_))})", fontsize = 20)

    plt.savefig(f'{filename}.png')
    plt.show()


plot_kmeans(df_normalized_public, public_clusters, 'public_transport_kmeans')
plot_kmeans(df_normalized_private, private_clusters, 'private_transport_kmeans')
plot_kmeans(df_normalized_active, active_clusters, 'active_transport_kmeans')



#---------------------------------------------------------------------------------------
#hierarchical clustering

# sampling the data
sample_public = df_normalized_public.sample(n=10, random_state=42)
sample_private = df_normalized_private.sample(n=10, random_state=42)
sample_active = df_normalized_active.sample(n=10, random_state=42)

# creating the linkage
Z_public = linkage(sample_public, method='ward')
Z_private = linkage(sample_private, method='ward')
Z_active = linkage(sample_active, method='ward')

# function for plotting dendrogram
def plot_dendrogram(Z, title, filename):

    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title(f'Dendrogram for {title}')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.savefig(filename)
    plt.show()

# plot dendrogram
plot_dendrogram(Z_public, 'Public Transport', 'public_transport_dendrogram.png')
plot_dendrogram(Z_private, 'Private Transport', 'private_transport_dendrogram.png')
plot_dendrogram(Z_active, 'Active Transport', 'active_transport_dendrogram.png')


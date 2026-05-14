import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('clustering.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Get the shape of the DataFrame
print(data.shape)


clusters = KMeans(n_clusters=2)
clusters.fit(data)

def plot_kmeans(df, clusters):
    """
    generate a 3d plot given sklearn's kmeans implementation 
    """
    colormap = {0: 'red', 1: 'green',  2: 'blue'}
    
    fig = plt.figure(figsize=(7, 10))
    ax = plt.axes(projection="3d")
    ax.scatter(df['time_on_site'], 
               df['quantity'], 
               df['movie_rating'],
               c=[colormap.get(x) for x in clusters.labels_])
    
    ax.set_ylabel('quantity')
    ax.set_xlabel('time_on_site')
    ax.set_zlabel('movie_rating')
    ax.set_title(f"k = {len(set(clusters.labels_))}")
    
    plt.show()

plot_kmeans(data, clusters)
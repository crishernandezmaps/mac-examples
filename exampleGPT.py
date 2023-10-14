def dash_example():
    """
    # Animated visualization to demonstrate RFM
    RFM stands for **Recency, Frequency, and Monetary value**. It's a segmentation technique used by marketers and retailers to categorize customers based on their purchase behavior. Here's a breakdown of each component:
        1. **Recency (R)**: Refers to how recently a customer made a purchase. A customer who has purchased recently is more likely to make another purchase compared to someone who hasn't purchased in a long time. It helps in identifying customers who are currently engaged with the brand or product.
        2. **Frequency (F)**: Refers to how often a customer makes a purchase. Customers who purchase frequently are more engaged and are more likely to respond positively to promotions compared to those who purchase less often.
        3. **Monetary Value (M)**: Refers to how much money a customer has spent over time. Customers who have spent more (either on high-value purchases or through frequent lower-value purchases) are typically seen as higher-value customers.
    The RFM model is effective because it leverages the Pareto Principle (or the 80/20 rule), which suggests that 80% of a company's revenue often comes from 20% of its customers. By segmenting customers based on these three characteristics, companies can prioritize their marketing and sales efforts on segments that are more likely to generate the most revenue.
    In practice, customers are often scored on each of the RFM factors, and then they can be grouped or segmented based on their combined RFM score. This helps businesses tailor specific marketing strategies for different segments. For example, customers with high recency, frequency, and monetary values might be treated as VIPs, while those with low recency might be targeted with re-engagement campaigns.    
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import plotly.express as px
    import pandas as pd

    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Sample data
    data = {
        'Forest_ID': ['A', 'B', 'C', 'D', 'E'],
        'R': [5, 12, 3, 9, 15],
        'F': [120, 150, 80, 200, 90],
        'M': [3.5, 4.2, 2.8, 5.0, 3.0]
    }
    df = pd.DataFrame(data)

    # Define the app layout
    app.layout = html.Div([
        html.H1("Dynamic RFM Visualization"),
        
        html.Div([
            html.Label("Forest ID:"),
            dcc.Input(id='input-forest-id', type='text', value=''),

            html.Label("R (Age in years):"),
            dcc.Input(id='input-r', type='number', value=0),

            html.Label("F (Number of trees):"),
            dcc.Input(id='input-f', type='number', value=0),

            html.Label("M (Tons per hectare):"),
            dcc.Input(id='input-m', type='number', value=0),

            html.Button("Add Data", id="add-button")
        ]),

        dcc.Graph(id='3d-scatter-plot')
    ])

    @app.callback(
        Output('3d-scatter-plot', 'figure'),
        [Input('add-button', 'n_clicks')],
        [dash.dependencies.State('input-forest-id', 'value'),
        dash.dependencies.State('input-r', 'value'),
        dash.dependencies.State('input-f', 'value'),
        dash.dependencies.State('input-m', 'value')]
    )
    def update_plot(n, forest_id, r, f, m):
        if n:
            new_data = {'Forest_ID': forest_id, 'R': r, 'F': f, 'M': m}
            global df
            df = df.append(new_data, ignore_index=True)
        
        fig = px.scatter_3d(df, x='R', y='F', z='M', 
                            text='Forest_ID',
                            color='Forest_ID',
                            size_max=18,
                            opacity=0.7)
        
        fig.update_layout(margin=dict(t=0, b=0, r=0, l=0))
        return fig

    if __name__ == '__main__':
        app.run_server(debug=True)


def nltk_example():
    """
    `nltk` stands for the **Natural Language Toolkit**. It's a leading Python library for working with human language data (text) and is used in the field of natural language processing (NLP). Here's a comprehensive description:

    1. **History & Popularity**:
    - `nltk` was created by Steven Bird and Edward Loper in the Department of Computer and Information Science at the University of Pennsylvania.
    - It has become one of the standard libraries in NLP and text processing in Python, especially for academic and prototyping purposes.

    2. **Main Features**:
    - **Tokenization**: Splitting sentences and words from the body of the text.
    - **POS Tagging**: Assigning a category tag to the tokenized parts of speech.
    - **Name Entity Recognition**: Identifying common entities (like people, organization names) in the text.
    - **Stemming and Lemmatization**: Reducing words to their base or root form.
    - **Stopword Removal**: Filtering out commonly used words.
    - **Frequency Analysis**: Counting words, phrases, etc.
    - **Concordance Views**: Locating specific words and viewing surrounding context.
    
    3. **Datasets & Corpora**:
    - `nltk` includes a wide range of corpora, lexicons, and trained models.
    - These are useful for training and testing algorithms, studying linguistic structures, and more.
    - Example datasets include word lists, treebanks, and even entire books.

    4. **Additional Tools**:
    - **Parsers**: For analyzing the grammatical structure of sentences.
    - **n-gram and collocations**: For identifying commonly co-occurring words.
    - **Chunkers**: For extracting phrases.
    - **Sentiwordnet**: A tool for sentiment analysis tasks.
    
    5. **Extensibility**:
    - You can extend `nltk` with your own corpora, trained models, etc.
    - While `nltk` is comprehensive, it's not always the best for production-level tasks due to speed and scalability. For such tasks, other libraries like SpaCy might be more suitable.
    
    6. **Educational & Research Value**:
    - It's accompanied by a book, "Natural Language Processing with Python" which serves as an excellent introduction to the field.
    - The library is widely used in academia for teaching and research.

    In summary, `nltk` is a comprehensive library with a variety of tools for text processing and analysis, making it a popular choice for those getting started with NLP in Python. However, for high-performance applications or large-scale data processing, other libraries such as SpaCy or transformers might be more suitable.    
    """
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Downloading necessary datasets from NLTK
    nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    # Sample documents
    docs = [
        "I love programming with Python.",
        "Python is a versatile programming language.",
        "NLTK is a leading platform for building Python programs to work with human language data."
    ]

    # Creating the TF-IDF vectorizer and removing stopwords
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    # Transforming documents to TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Displaying the terms and their scores
    terms = vectorizer.get_feature_names_out()
    for i, doc in enumerate(docs):
        tfidf_scores = [(terms[col], tfidf_matrix[i, col]) for col in tfidf_matrix[i].nonzero()[1]]
        print(f"\nDocument {i+1}")
        for word, score in tfidf_scores:
            print(f"Word: {word}, TF-IDF Score: {score}")


def fuzzyExample():
    """
    Fuzzy segmentation is a method that classifies or groups data points into overlapping clusters rather than distinct, non-overlapping clusters. This is especially useful in scenarios where boundaries between clusters aren't clear-cut and a data point can belong to multiple clusters to varying degrees. The term "fuzzy" here indicates that the classification is not hard; instead, it's soft or probabilistic. 
    In the context of customer segmentation or any other form of data segmentation, the fuzzy method allows each data point (e.g., a customer) to belong to multiple segments with varying degrees of membership.
    **Fuzzy c-means (FCM)** is one of the most popular algorithms used for fuzzy segmentation:
    1. **Initialization**: The algorithm starts by initializing cluster centroids randomly.
    2. **Membership Assignment**: Each data point is assigned a membership value for each cluster, based on its distance from the cluster centroids. The closer a data point is to a centroid, the higher its degree of membership to that cluster.
    3. **Centroid Recalculation**: Cluster centroids are recalculated based on the membership values of the data points.
    4. **Iteration**: Steps 2 and 3 are repeated until the algorithm converges, i.e., until the centroids don't change significantly between iterations or a certain number of iterations is reached.
    The key difference between FCM and traditional clustering methods, like k-means, is the membership assignment step. In k-means, each data point belongs to one and only one cluster. In FCM, each data point has a membership value for each cluster, indicating the degree to which it belongs to that cluster.
    **Applications of Fuzzy Segmentation**:
    1. **Customer Segmentation**: Rather than placing a customer in a single segment, fuzzy segmentation might determine that a customer is 70% in Segment A, 20% in Segment B, and 10% in Segment C. This can give businesses a more nuanced view of their customers.
    2. **Image Processing**: Fuzzy segmentation can be used to segment parts of an image where boundaries between regions are not clear.
    3. **Medical Imaging**: Fuzzy methods can be particularly useful in medical imaging, where the boundaries between different tissues or lesions might be ambiguous.
    4. **Market Research**: It can help in understanding products or services that have overlapping features or audiences.
    In summary, fuzzy segmentation offers a more nuanced and flexible approach to clustering, especially in situations where boundaries between clusters are not distinct.    
    """
    import numpy as np
    import skfuzzy as fuzz
    import matplotlib.pyplot as plt

    # Generate some sample data
    np.random.seed(42)  # for reproducibility
    xpts = np.zeros(1 * 100)
    ypts = np.zeros(1 * 100)
    xpts[:100] = np.random.uniform(0, 10, 100)
    ypts[:100] = np.random.uniform(0, 10, 100)

    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    # Set up the data array
    alldata = np.vstack((xpts, ypts))
    fpcs = []

    # Let's say we want 3 clusters
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, 3, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later plotting
    fpcs.append(fpc)

    # Plot the clustered data points
    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))

    for ax in axes1.reshape(-1):
        ax.axis('off')

    ax = axes1.reshape(-1)[0]
    ax.plot(xpts, ypts, 'yo')
    ax.set_title('Test data')

    # Plot the real cluster assignments
    for j in range(3):
        ax = axes1.reshape(-1)[j + 1]
        for pt in zip(xpts, ypts):
            ax.plot(pt[0], pt[1], colors[j])

        # Adjust the x and y axis limits
        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 12)
        ax.set_title(f'True cluster {j}')

    # Plot the results of the fuzzy c-means clustering
    ax = axes1.reshape(-1)[-2]
    for j in range(3):
        ax.plot(xpts[u.argmax(axis=0) == j],
                ypts[u.argmax(axis=0) == j], '.', color=colors[j])

    # Highlight the centers of the clusters
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    print(fpcs)

    ax.set_title('Fuzzy C-Means Results')
    plt.tight_layout()
    plt.show()


def kmeans3():
    """
    It seems you're referring to "k-means," a popular clustering algorithm. Let's dive into its definition:
    **k-means clustering** is a partitioning method that divides a dataset into \(k\) distinct, non-overlapping subsets (or clusters). The goal is to partition the data into clusters such that the total within-cluster variation, or equivalently the total distance between the data points and their cluster centroids, is minimized.
    **Working of k-means**:
    1. **Initialization**: Randomly select \(k\) data points (from the dataset) to be the initial centroids.
    2. **Assignment Step**: Assign each data point to the nearest centroid. This forms \(k\) clusters.
    3. **Update Step**: Calculate the new centroid (mean) of each cluster.
    4. **Iteration**: Repeat the assignment and update steps until the centroids do not change significantly between successive iterations, or a set number of iterations is reached.
    **Characteristics and Use Cases**:
    1. **Sensitivity to Initial Centroids**: The choice of initial centroids can affect the final clusters. There are various methods to initialize centroids more effectively, such as k-means++.
    2. **Number of Clusters**: The user must specify \(k\), the number of clusters. This can be a limitation as the optimal number of clusters might not be known beforehand. Methods like the Elbow Method can be used to estimate the best \(k\).
    3. **Globally Optimal Solution**: k-means might converge to a local minimum. Running the algorithm multiple times with different initializations can help achieve a more global solution.
    4. **Linear Boundaries**: k-means tends to find clusters with linear boundaries. It might not work well for complex-shaped clusters.
    5. **Use Cases**: k-means is widely used in market segmentation, image compression, document clustering, and many other domains.
    6. **Variations**: There are variations of the k-means algorithm to handle different types of data and requirements, such as k-medoids or k-medians.
    In essence, k-means is a simple, yet powerful, clustering algorithm that partitions a dataset into \(k\) clusters by iteratively updating cluster assignments and centroids until convergence.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # Generate some sample data
    np.random.seed(42)  # for reproducibility
    xpts = np.zeros(1 * 100)
    ypts = np.zeros(1 * 100)
    xpts[:100] = np.random.uniform(0, 10, 100)
    ypts[:100] = np.random.uniform(0, 10, 100)

    # Combine x and y points
    data = np.array(list(zip(xpts, ypts)))

    # Use KMeans to cluster the data into 3 clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.cluster_centers_

    # Plot the original data points and the centroids
    colors = ['b', 'g', 'r']

    fig, ax = plt.subplots()

    for i in range(3):
        # Plot each cluster
        points = np.array([data[j] for j in range(len(data)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=50, c=colors[i], label=f'Cluster {i + 1}')

    # Plot the centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='black', zorder=10)

    ax.set_title('KMeans Clustering Results with K=3')
    ax.legend()
    plt.show()


def kmeansElbow():
    """
    The **Elbow Method** is a heuristic used in determining the optimal number of clusters for a dataset in k-means clustering. The basic idea behind the method is to run k-means clustering on the dataset for a range of values of \(k\) (e.g., \(k\) from 1 to 10), and then for each value of \(k\) compute the sum of squared distances from each point to its assigned center.
    **Steps to Implement the Elbow Method**:
    1. **Compute k-means clustering**: For each value of \(k\), compute the k-means clustering algorithm and record the sum of squared distances (within-cluster sum of squares).
    2. **Plot the curve**: Plot the curve of the sum of squared distances as a function of the number of clusters \(k\).
    3. **Identify the Elbow Point**: When the reduction in the sum of squared distances begins to slow, an "elbow" is formed in the graph. The \(k\) at which this change becomes noticeable is considered a good estimate for the actual number of clusters.
    **Interpretation**:
    - For smaller values of \(k\) (1, 2, ..), the sum of squared distances tends to be high; this is because there are fewer clusters and points are farther from the centroids of their respective clusters.
    - As \(k\) increases, the sum of squared distances decreases because the clusters are smaller and tighter.
    - At some point, however, the benefit of increasing \(k\) will start to plateau, leading to smaller reductions in the sum of squared distances. The value of \(k\) at which this change in the rate of decrease becomes noticeable is called the elbow, and this is considered a reasonable estimate of the true number of clusters.
    **Limitations**:
    - The elbow method is more of a rule of thumb than a rigorous statistical method. In some datasets, the elbow might not be clear or well-defined.
    - Other techniques, such as the silhouette method or gap statistic, can also be used in combination with the elbow method to validate the number of clusters.
    In summary, the Elbow Method is a visual tool in k-means clustering to estimate the optimal number of clusters by spotting the location where the decrease of the within-cluster sum of squares begins to slow down.    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # Generate some sample data
    np.random.seed(42)
    xpts = np.concatenate([np.random.normal(loc, 1, 100) for loc in [0, 4, 8]])
    ypts = np.concatenate([np.random.normal(loc, 1, 100) for loc in [0, 4, 8]])
    data = np.array(list(zip(xpts, ypts)))

    # Calculate the sum of squared distances for a range of cluster counts
    wcss = []
    cluster_range = range(1, 11)  # Checking for up to 10 clusters
    for n in cluster_range:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # Inertia: Sum of squared distances to closest cluster center

    # Plot the elbow graph
    plt.figure(figsize=(10,5))
    plt.plot(cluster_range, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.show()


def elbowForFuzzy():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import skfuzzy as fuzz

    # Generate sample data
    np.random.seed(42)
    xpts = np.concatenate([np.random.normal(loc, 1, 100) for loc in [0, 4, 8]])
    ypts = np.concatenate([np.random.normal(loc, 1, 100) for loc in [0, 4, 8]])
    data = np.vstack((xpts, ypts)).T

    # Elbow Method to find the optimal number of clusters
    wcss = []
    cluster_range = range(1, 11)
    for n in cluster_range:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    optimal_clusters = np.argmax(np.diff(np.diff(wcss))) + 2  # Find the elbow

    # Apply Fuzzy C-Means using the optimal number of clusters
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data.T, optimal_clusters, 2, error=0.005, maxiter=1000)

    # Plot the data points with the fuzzy memberships using opacity for percentage membership
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    fig, ax = plt.subplots()

    # Plot each point based on its membership degree for each cluster
    for j in range(optimal_clusters):
        for i, (x, y) in enumerate(data):
            ax.scatter(x, y, color=colors[j], s=10, alpha=u[j, i])
        ax.scatter(cntr[j, 0], cntr[j, 1], marker='o', s=200, edgecolors='k', facecolors=colors[j], linewidths=2)

    ax.set_title('Fuzzy C-Means Clustering with Membership Visualization')
    plt.show()


def davisBoulding():
    """
    Davis Bouldin Index (Davies and Bouldin, 1979)
    """
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import davies_bouldin_score

    # Create a sample dataset
    data, _ = make_blobs(n_samples=500, centers=5, random_state=42)

    # Cluster data using KMeans
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Compute Davis-Bouldin Index
    dbi = davies_bouldin_score(data, labels)

    print(f"Davis-Bouldin Index: {dbi:.2f}")


def dbScan():
    """
    The DBSCAN algorithm uses two parameters:
    •minPts: The minimum number of points (a threshold) clustered together 
    for a region to be considered dense.
    •eps (ε): A distance measure that will be used to locate the points in the 
    neighborhood of any point. 
    These parameters can be understood if we explore two concepts called 
    Density Reachability and Density Connectivity.
    Reachability in terms of density establishes a point to be reachable from 
    another if it lies within a particular distance (eps) from it.
    Connectivity, on the other hand, involves a transitivity-based chainingapproach to determine whether points are in a particular cluster. For 
    example, p and q points could be connected if p->r->s->t->q, where a->b 
    means b is in the neighborhood of a.    
    """
    import dash
    from dash import dcc, html
    import numpy as np
    from sklearn.datasets import make_moons
    from sklearn.cluster import DBSCAN
    import pandas as pd
    import plotly.express as px

    # Generate sample data
    data, _ = make_moons(n_samples=500, noise=0.05, random_state=42)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(data)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["x", "y"])
    df["label"] = labels

    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='live-graph'),
        dcc.Interval(
            id='interval-component',
            interval=100,  # in milliseconds
            n_intervals=0
        )
    ])

    @app.callback(
        dash.dependencies.Output('live-graph', 'figure'),
        [dash.dependencies.Input('interval-component', 'n_intervals')]
    )
    def update_graph(n_intervals):
        # Select the first n_intervals of the data and labels
        limited_data = df.iloc[:n_intervals]
        fig = px.scatter(limited_data, x="x", y="y", color="label", title="DBSCAN Clustering Animation")

        return fig

    if __name__ == '__main__':
        app.run_server(debug=True)


dbScan()

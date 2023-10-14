def dash_example():
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


davisBoulding()
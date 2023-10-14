import os
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Load data
data = load_iris()
X = data.data
y = data.target

# Prepare PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Random Forest Classifier with Different Criteria"),
    dcc.Dropdown(
        id='method-dropdown',
        options=[
            {'label': 'Gini', 'value': 'gini'},
            {'label': 'Entropy', 'value': 'entropy'}
        ],
        value='gini',
        style={'width': '50%'}
    ),
    dcc.Graph(id='rf-graph')
])

@app.callback(
    Output('rf-graph', 'figure'),
    [Input('method-dropdown', 'value')]
)
def update_figure(method):
    clf = RandomForestClassifier(criterion=method, random_state=42)
    clf.fit(X, y)
    predictions = clf.predict(X)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=predictions, title=f'2D PCA using {method}',
                     color_continuous_scale='Rainbow', labels={'x':'Principal Component 1', 'y':'Principal Component 2'})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
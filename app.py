from flask import Flask, render_template, request, flash
import numpy as np
import matplotlib.pyplot as plt
import mpld3

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Global variables to hold data and states
data = None
centroids = None
clusters = None
num_clusters = 2  # Default number of clusters
current_step = 0
steps = []

def generate_random_data(num_points=100):
    """Generate random data points."""
    return np.random.rand(num_points, 2)

def initialize_centroids(method):
    """Initialize centroids based on the chosen method."""
    global centroids, num_clusters, data  # Declare globals to modify them
    if data is None:
        flash('Please generate data first.')
        return  # Early exit if there's no data
    
    if method == 'random':
        centroids = data[np.random.choice(data.shape[0], num_clusters, replace=False)]
    elif method == 'farthest_first':
        centroids = [data[np.random.randint(0, data.shape[0])]]
        for _ in range(1, num_clusters):
            distances = np.linalg.norm(data[:, np.newaxis] - np.array(centroids), axis=2)
            centroids.append(data[np.argmax(np.min(distances, axis=1))])
        centroids = np.array(centroids)
    elif method == 'kmeans++':
        centroids = [data[np.random.choice(data.shape[0])]]
        for _ in range(1, num_clusters):
            distances = np.min(np.linalg.norm(data[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
            probabilities = distances / np.sum(distances)
            centroids.append(data[np.random.choice(data.shape[0], p=probabilities)])
        centroids = np.array(centroids)

def kmeans():
    """Run the KMeans algorithm."""
    global clusters, steps, centroids
    steps = []  # Reset steps for the new run
    if centroids is None:
        flash('Centroids must be initialized before running KMeans.')
        return  # Early exit if centroids are not set

    for _ in range(10):  # Iterate a fixed number of times (or until convergence)
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)

        # Save the current step's state for visualization
        steps.append((clusters.copy(), centroids.copy()))

        # Update centroids
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(num_clusters) if np.any(clusters == i)])
        
        if len(new_centroids) < num_clusters:  # Check if centroids can be updated
            break
        if np.all(centroids == new_centroids):  # Check for convergence
            break
        centroids = new_centroids

def plot_data(show_centroids=False, reset_colors=False):
    """Plot the data and centroids."""
    plt.figure(figsize=(10, 6))

    if reset_colors:
        plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data Points')  # All points in blue
    else:
        if clusters is None:  # No clustering has been done yet
            plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data Points')  # All points in blue
        else:  # Clustering has been done
            colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))  # Create colormap
            # Plot each cluster in a different color
            for i in range(num_clusters):
                plt.scatter(data[clusters == i, 0], data[clusters == i, 1],
                            color=colors[i], label=f'Cluster {i + 1}')
    
    if show_centroids and centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

    plt.title('KMeans Clustering')
    plt.axis('equal')
    plt.legend()  # Add a legend to identify clusters

    # Use mpld3 to make the plot interactive
    html_graph = mpld3.fig_to_html(plt.gcf())
    plt.close()  # Close the figure to avoid saving it as an image
    return html_graph

@app.route('/', methods=['GET', 'POST'])
def index():
    global data, centroids, num_clusters, current_step, steps, clusters
    if request.method == 'POST':
        if 'generate' in request.form:
            data = generate_random_data()
            clusters = None  # Reset clusters to None for new data
            flash('Data generated successfully!')
            # Immediately plot the generated data without centroids
            html_graph = plot_data(show_centroids=False)
            return render_template('index.html', graph=html_graph)
        elif 'run_to_convergence' in request.form:
            if data is None:
                flash('Please generate data first.')
            else:
                method = request.form.get('init-method')
                num_clusters = int(request.form.get('num-clusters', 2))  # Get the number of clusters from the form
                initialize_centroids(method)
                if centroids is not None:  # Ensure centroids were initialized
                    kmeans()  # Run the KMeans algorithm to convergence
                    html_graph = plot_data(show_centroids=True)  # Show final state with centroids
                    return render_template('index.html', graph=html_graph)
        elif 'step' in request.form:
            if current_step < len(steps) - 1:
                current_step += 1
                clusters, centroids = steps[current_step]
                html_graph = plot_data(show_centroids=True)  # Show the current state with centroids
                return render_template('index.html', graph=html_graph)
            else:
                flash('You have reached the end of the steps.')
        elif 'reset' in request.form:
            centroids = None
            clusters = None
            steps = []
            current_step = 0
            flash('Algorithm reset successfully!')
            # Reset the data points color to blue without regenerating data
            html_graph = plot_data(show_centroids=False, reset_colors=True)
            return render_template('index.html', graph=html_graph)

    return render_template('index.html', graph=None)

if __name__ == '__main__':
    app.run(debug=True)

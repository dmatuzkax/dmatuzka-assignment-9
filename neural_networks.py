import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
import networkx as nx

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim)
        self.b2 = np.zeros((1, self.output_dim))       

    def forward(self, X):
        # TODO: forward pass, apply layers to input 
        self.Z1 = np.dot(X, self.W1) + self.b1

        # TODO: store activations for visualization
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))

        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        
        out = 1 / (1 + np.exp(-self.Z2)) 
        self.out = out
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        out = self.out

        dZ2 = out - y
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)

        if self.activation_fn == 'tanh':
            A1 = np.tanh(self.Z1)
            dZ1 = dA1 * (1 - A1 ** 2)
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            A1 = 1 / (1 + np.exp(-self.Z1))
            dZ1 = dA1 * A1 * (1 - A1)

        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

        # TODO: update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # TODO: store gradients for visualization
        self.gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        }
  
        return

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f"Hidden Space at Step {frame*10}")

    # TODO: Hyperplane visualization in the hidden space
    x_min, x_max = hidden_features[:, 0].min() - 1, hidden_features[:, 0].max() + 1
    y_min, y_max = hidden_features[:, 1].min() - 1, hidden_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z_hidden = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z_hidden = Z_hidden.reshape(xx.shape)

    # ax_hidden.contourf(xx, yy, Z_hidden, levels=[-1, 0, 1], cmap='bwr', alpha=0.3)
    ax_hidden.plot_surface(xx, yy, Z_hidden, color='orange', alpha=0.3)

    # TODO: Distorted input space transformed by the hidden layer
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z_input = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z_input = Z_input.reshape(xx.shape)

    # TODO: Plot input layer decision boundary
    ax_input.contourf(xx, yy, Z_input, levels=[0, 0.5, 1], cmap='bwr', alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title(f"Input Space at Step {frame*10}")

    # TODO: Visualize features and gradients as circles and edges 

    pos = {
        'x1': (0, 0),  
        'x2': (0, 1),
        'h1': (0.5, 1),
        'h2': (0.5, 0.5),
        'h3': (0.5, 0),
        'y': (1, 0)
    }

    edges = [
        ('x1', 'h1'), ('x2', 'h1'), ('x1', 'h2'), ('x2', 'h2'),
        ('x1', 'h3'), ('x2', 'h3'), ('h1', 'y'), ('h2', 'y'), ('h3', 'y')
    ]

    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    G.add_edges_from(edges)

    node_sizes = [500 for _ in pos] 
    edge_widths = []
    edge_colors = []

    for grad_key, grad_value in mlp.gradients.items():
        gradient_magnitude = np.linalg.norm(grad_value, axis=0)

        if np.max(gradient_magnitude) != np.min(gradient_magnitude):
            norm_gradients = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
        else:
            norm_gradients = np.ones_like(gradient_magnitude)
        
        edge_widths.extend([norm_grad * 4 for norm_grad in norm_gradients])
        edge_colors.extend([(0.5, 0, 0.5, norm_grad) for norm_grad in norm_gradients])

    nx.draw_networkx_nodes(G, pos, ax=ax_gradient, node_size=node_sizes, node_color='blue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=ax_gradient, width=edge_widths, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, ax=ax_gradient, font_size=12, font_color='white')

    ax_gradient.set_title(f"Gradients at Step {frame*10}")
    ax_gradient.axis('on')
    ax_gradient.set_xlim(-0.1, 1.1) 
    ax_gradient.set_ylim(-0.1, 1.1)
    
    plt.draw()

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000 
    visualize(activation, lr, step_num)
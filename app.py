import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import plotly.graph_objects as go
from google.colab import files
import matplotlib.pyplot as plt

# --- Vector Geometry Utilities ---
def normalize(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def pseudo_cross(v1, v2):
    """Compute the 2D pseudo-cross product (scalar)."""
    return v1[0] * v2[1] - v1[1] * v2[0]

def compute_curvature(points):
    """Compute local curvature for each vertex based on adjacent edges."""
    curvatures = []
    for i in range(len(points)):
        prev = points[i - 1] if i > 0 else points[-1]
        curr = points[i]
        next = points[(i + 1) % len(points)]
        v1 = curr - prev
        v2 = next - curr
        angle = np.arccos(np.clip(np.dot(normalize(v1), normalize(v2)), -1.0, 1.0))
        curvatures.append(angle)
    return np.array(curvatures)

# --- Adaptive Vector Flow Deformation (AVFD) ---
def adaptive_vector_flow_deform(points, anchor_1, anchor_2, alpha=0.5, warp_depth=1.0, sigma=0.5):
    """Deform 2D shape vertices using adaptive vector flow deformation with curvature-based weighting."""
    direction = anchor_2 - anchor_1
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return np.hstack((points, np.zeros((points.shape[0], 1))))

    direction_unit = normalize(direction)
    normal_vector = np.array([-direction_unit[1], direction_unit[0]])

    curvatures = compute_curvature(points)
    max_curvature = np.max(curvatures)
    weights = 1 + curvatures / max_curvature if max_curvature > 0 else np.ones_like(curvatures)

    deformed_points = []
    flow_field = []

    for p, w in zip(points, weights):
        v_i = p - anchor_1
        s_i = np.dot(v_i, direction) / (direction_norm ** 2)
        c_i = pseudo_cross(v_i, direction) / direction_norm
        flow = s_i * direction + alpha * c_i * w * normal_vector
        flow_field.append(flow)

    flow_field = np.array(flow_field)
    flow_field = gaussian_filter(flow_field, sigma=sigma)

    for i, p in enumerate(points):
        new_point_2d = p + flow_field[i]
        z = warp_depth * pseudo_cross(p - anchor_1, direction) / direction_norm
        deformed_points.append([new_point_2d[0], new_point_2d[1], z])

    return np.array(deformed_points)

# --- Image Processing to Extract Vertices ---
def preprocess_image(img):
    """Convert to grayscale, resize, and detect edges for vertex extraction."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 100, 200)
    return img, edges

def extract_vertices_from_edges(edge_img, max_vertices=500):
    """Extract 2D vertices from edge image using contour detection."""
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for contour in contours:
        # Simplify contour to reduce noise while retaining shape
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points.extend(approx.reshape(-1, 2))
    points = np.array(points)

    edge_pixels = np.sum(edge_img > 0)
    if len(points) == 0:
        raise ValueError("No edges detected in the image.")
    if len(points) > max_vertices:
        curvatures = compute_curvature(points)
        weights = 1 + curvatures / np.max(curvatures) if np.max(curvatures) > 0 else np.ones_like(curvatures)
        probs = weights / np.sum(weights)
        indices = np.random.choice(len(points), max_vertices, replace=False, p=probs)
        points = points[indices]
    return points, edge_pixels

# --- 2D Visualization with OpenCV ---
def visualize_2d_edges(image, edges, vertices, anchor_1, anchor_2):
    """Visualize edges and vertices with control points using OpenCV."""
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    vis_img[edges > 0] = [0, 255, 0]  # Green edges
    for vertex in vertices:
        cv2.circle(vis_img, (int(vertex[0]), int(vertex[1])), 2, (255, 0, 0), -1)  # Red vertices
    cv2.circle(vis_img, (int(anchor_1[0]), int(anchor_1[1])), 5, (0, 255, 255), -1)  # Yellow anchor 1
    cv2.circle(vis_img, (int(anchor_2[0]), int(anchor_2[1])), 5, (255, 0, 255), -1)  # Purple anchor 2
    cv2.line(vis_img, (int(anchor_1[0]), int(anchor_1[1])), (int(anchor_2[0]), int(anchor_2[1])), (0, 0, 255), 1)  # Red line
    plt.figure(figsize=(6, 6))
    plt.imshow(vis_img)
    plt.title("Edges, Vertices, and Control Points")
    plt.axis('off')
    plt.savefig('2d_visualization.png')
    plt.close()

# --- Interactive 3D Visualization (Vertex-Based Deformation) ---
def plot_interactive_3d_deformation(original_points, deformed_points, anchor_1, anchor_2, title="Optimized AVFD 3D Deformation (Vertices)"):
    """Visualize original and deformed shapes with control points in 3D."""
    fig = go.Figure()

    x, y = original_points[:, 0], original_points[:, 1]
    z = np.zeros_like(x)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='lines+markers', name='Original Shape',
        line=dict(color='blue'), marker=dict(size=3)
    ))

    x, y, z = deformed_points[:, 0], deformed_points[:, 1], deformed_points[:, 2]
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='lines+markers', name='Deformed Shape',
        line=dict(color='red'), marker=dict(size=3)
    ))

    fig.add_trace(go.Scatter3d(
        x=[anchor_1[0], anchor_2[0]], y=[anchor_1[1], anchor_2[1]], z=[0, 0],
        mode='markers+lines', name='Control Points',
        marker=dict(size=8, color=['green', 'purple']), line=dict(color='black', dash='dash')
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            zaxis=dict(range=[-max(abs(z)) * 1.5, max(abs(z)) * 1.5])
        ),
        width=700,
        height=500
    )
    fig.show()

# --- Interactive 3D Surface Visualization (Complete Deformation) ---
def plot_interactive_3d_surface(deformed_points, title="Optimized AVFD 3D Deformation (Surface)"):
    """Create an interactive 3D surface plot from deformed points."""
    x = deformed_points[:, 0]
    y = deformed_points[:, 1]
    z = deformed_points[:, 2]

    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')  # Use linear for speed

    zi = np.nan_to_num(zi, nan=0.0)

    fig = go.Figure(data=[
        go.Surface(
            z=zi, x=xi, y=yi,
            colorscale='Viridis',
            name='Deformed Surface'
        )
    ])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            zaxis=dict(range=[min(z) * 1.5, max(z) * 1.5])
        ),
        width=700,
        height=500
    )
    fig.show()

# --- Main Pipeline ---
try:
    print("Please upload an image (color or grayscale, e.g., PNG or JPG).")
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No image uploaded.")

    for fname in uploaded.keys():
        image = cv2.imread(fname)
        if image is None:
            raise ValueError(f"Failed to load image: {fname}")

    image, edges = preprocess_image(image)
    vertices, edge_pixels = extract_vertices_from_edges(edges, max_vertices=500)

    # Visualize 2D edges and vertices
    visualize_2d_edges(image, edges, vertices, np.array([50, 128]), np.array([200, 128]))

    A = np.array([50, 128])  # Anchor 1
    B = np.array([200, 128])  # Anchor 2
    alpha = 0.8  # Perpendicular stretch
    warp_depth = 1.2  # Pseudo-3D deformation scaling
    sigma = 0.5  # Reduced for faster smoothing

    deformed_vertices = adaptive_vector_flow_deform(vertices, A, B, alpha=alpha, warp_depth=warp_depth, sigma=sigma)

    # Plot vertex-based 3D deformation
    plot_interactive_3d_deformation(vertices, deformed_vertices, A, B, title="Optimized AVFD 3D Deformation (Vertices)")

    # Plot complete 3D surface deformation
    plot_interactive_3d_surface(deformed_vertices, title="Optimized AVFD 3D Deformation (Surface)")

    # Calculate and display deformation efficiency
    vertex_count = len(vertices)
    efficiency = (vertex_count / edge_pixels) * 100 if edge_pixels > 0 else 0
    print(f"\n📊 Deformation Efficiency: {efficiency:.2f}% (Vertices: {vertex_count}, Edge Pixels: {edge_pixels})")

except Exception as e:
    print(f"Error: {str(e)}")
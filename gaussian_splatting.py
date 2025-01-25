import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# # Points: [x, y, z], Colors: [r, g, b], Normals (optional)
# def generate_sample_points():
#     points = np.array([
#         [1.0, 0.5, 0.2, 255, 0, 0],
#         [0.5, 1.0, 0.8, 0, 255, 0],
#         [0.8, 0.3, 1.0, 0, 0, 255],
#         [1.2, 0.8, 0.4, 255, 255, 0],
#         [0.4, 1.2, 0.6, 0, 255, 255],
#     ])
#     return points[:, :3], points[:, 3:]/255.0

# points, colors = generate_sample_points()

point = [1.0, 0.5, 0.2]
color =  [255, 0, 0]
theta = np.radians(45)
R = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])
S = np.diag([4, 1, 0.25])
opacity = 0.4
covariance_matrix = R @ S @ R.T

def get_camera_and_sample_image():
    v, i = 0, 0
    return v, i

def gaussian(x, y, sigma):
    xy = np.array([x,y])
    exponent = -0.5 * (xy.T @ np.linalg.inv(sigma) @ xy)
    return np.exp(exponent)

def rasterize(w, h, M, S, C, A, V):
    """
    Rasterizes Gaussian points onto a 2D image.

    Parameters:
        w (int): Width of the image.
        h (int): Height of the image.
        M (np.ndarray): Array of Gaussian centers in 3D (shape: [N, 3]).
        S (np.ndarray): Array of covariance matrices (shape: [N, 3, 3]).
        C (np.ndarray): Array of colors (shape: [N, 3]).
        A (np.ndarray): Array of alpha values (shape: [N, 1]).
        V (np.ndarray): View transformation matrix (4x4).
    
    Returns:
        np.ndarray: Rasterized image (shape: [h, w, 3]).
    """
    # Step 1: Frustum culling (filter points within the view frustum)
    def cull_gaussian(M, V):
        M_h = np.concatenate([M, np.ones((M.shape[0], 1))], axis=1)
        M_view = M_h @ V.T
        return M_view[M_view[:, 2] > 0]

    # Step 2: Transform Gaussians to screen space
    def screen_space_gaussians(M, S, V):
        print("Shape von M (vorher):", M.shape)
        M_transformed = M @ V.T  # (N x 4)
        M_screen = M_transformed[:, :3] / M_transformed[:, 3][:, None]
        S_screen = []
        for S_n in S:
            transformed_S = V[:3, :3] @ S_n @ V[:3, :3].T
            S_screen.append(transformed_S)
        S_screen = np.array(S_screen)
        return M_screen, S_screen

    # Step 3: Create tiles (split the screen into small regions)
    def create_tiles(w, h, tile_size=16):
        tiles_x = w // tile_size
        tiles_y = h // tile_size
        return tiles_x, tiles_y, tile_size

    # Step 4: Duplicate points and assign them to tiles
    def duplicate_with_keys(M, tiles_x, tiles_y, tile_size):
        keys = (M[:, 0] // tile_size).astype(int) + (M[:, 1] // tile_size).astype(int) * tiles_x
        return np.arange(len(M)), keys

    # Step 5: Sort points by tile keys
    def sort_by_keys(keys, indices):
        sorted_indices = np.argsort(keys)
        return keys[sorted_indices], indices[sorted_indices]

    # Step 6: Identify ranges for each tile
    def identify_tile_ranges(tiles_x, tiles_y, keys):
        ranges = {}
        for tile_key in np.unique(keys):
            indices = np.where(keys == tile_key)[0]
            ranges[tile_key] = indices
        return ranges

    # Step 7: Blend Gaussians into the image
    def blend_in_order(pixel, indices, ranges, tile_key, M, S, C, A, image):
        if tile_key not in ranges:
            return image
        for idx in ranges[tile_key]:
            center = M[idx]
            color = C[idx]
            alpha = A[idx]
            dx, dy = pixel[0] - center[0], pixel[1] - center[1]
            distance_squared = dx**2 + dy**2
            weight = np.exp(-distance_squared / (2 * S[idx, 0, 0]))
            image[pixel[1], pixel[0], :] += alpha * weight * color
        return image

    # Initialize canvas
    image = np.zeros((h, w, 3), dtype=np.float32)
    M = cull_gaussian(M, V)
    M, S = screen_space_gaussians(M, S, V)
    tiles_x, tiles_y, tile_size = create_tiles(w, h)
    indices, keys = duplicate_with_keys(M, tiles_x, tiles_y, tile_size)
    keys, indices = sort_by_keys(keys, indices)
    ranges = identify_tile_ranges(tiles_x, tiles_y, keys)

    # Rasterize points into tiles
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile_key = tile_y * tiles_x + tile_x
            for y in range(tile_y * tile_size, min((tile_y + 1) * tile_size, h)):
                for x in range(tile_x * tile_size, min((tile_x + 1) * tile_size, w)):
                    image = blend_in_order((x, y), indices, ranges, tile_key, M, S, C, A, image)

    return image


w, h = 256, 256
num_points = 100
M = np.random.rand(num_points, 3) * 10 - 5
S = np.array([np.eye(3) for _ in range(num_points)])
C = np.random.rand(num_points, 3)
A = np.random.rand(num_points, 1)
V = np.eye(4)
image = rasterize(w, h, M, S, C, A, V)

# Visualize the output
import matplotlib.pyplot as plt
plt.imshow(image)
plt.title("Rasterized Gaussian Splatting")
plt.show()


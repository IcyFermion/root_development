import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List, Tuple, Set
import time
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree

class Node:
    def __init__(self, pos, t):
        self.pos = np.array([pos[0], pos[1]])
        self.t = t
        self.children = []

class TimeAwareTreeCoverage:
    def __init__(self, root: Node, radius: float):
        """
        Initialize the time-aware tree coverage calculator.
        
        Args:
            root: Root node of the tree
            radius: Coverage radius r
        """
        self.root = root
        self.radius = radius
        self.node_distances = {}  # Cache for node distances to root
        self.node_colors = {}     # Cache for node color mappings
        self.color_to_node = {}   # Reverse mapping from color to node
    
    def get_active_edges(self, node: Node, time_threshold: float, parent_pos=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Recursively get all edges that exist at the given time.
        
        Args:
            node: Current node
            time_threshold: Only include nodes with t <= time_threshold
            parent_pos: Position of parent node (if any)
        
        Returns:
            List of edges as (start_pos, end_pos) tuples
        """
        edges = []
        
        # Skip this node if it doesn't exist yet at the given time
        if node.t > time_threshold:
            return edges
        
        # Add edge from parent to this node (if parent exists)
        if parent_pos is not None:
            edges.append((parent_pos, node.pos))
        
        # Recursively process children
        for child in node.children:
            edges.extend(self.get_active_edges(child, time_threshold, node.pos))
        
        return edges
    
    def get_active_nodes(self, node: Node, time_threshold: float) -> List[np.ndarray]:
        """
        Recursively get all node positions that exist at the given time.
        
        Args:
            node: Current node
            time_threshold: Only include nodes with t <= time_threshold
        
        Returns:
            List of node positions
        """
        nodes = []
        
        # Skip this node if it doesn't exist yet
        if node.t > time_threshold:
            return nodes
        
        nodes.append(node.pos)
        
        # Recursively process children
        for child in node.children:
            nodes.extend(self.get_active_nodes(child, time_threshold))
        
        return nodes
    
    def calculate_node_distances(self, node: Node, time_threshold: float, 
                               parent_node=None, distance_to_parent=0) -> None:
        """
        Recursively calculate L2 path distance from each node to root.
        
        Args:
            node: Current node
            time_threshold: Only include nodes with t <= time_threshold
            parent_node: Parent node (if any)
            distance_to_parent: Accumulated distance from root to parent
        """
        if node.t > time_threshold:
            return
        
        # Calculate distance from parent to this node
        if parent_node is None:
            # Root node has distance 0
            self.node_distances[id(node)] = 0
        else:
            # Add edge length to parent's distance
            edge_length = np.linalg.norm(node.pos - parent_node.pos)
            self.node_distances[id(node)] = distance_to_parent + edge_length
        
        # Recursively process children
        for child in node.children:
            self.calculate_node_distances(child, time_threshold, node, 
                                        self.node_distances[id(node)])
    
    def assign_node_colors(self, node: Node, time_threshold: float, color_index: List[int]) -> None:
        """
        Assign unique colors to each node.
        
        Args:
            node: Current node
            time_threshold: Only include nodes with t <= time_threshold
            color_index: Mutable list containing current color index
        """
        if node.t > time_threshold:
            return
        
        # Assign color to node
        # Use RGB values to create unique colors (skip pure white)
        r = (color_index[0] % 256)
        g = ((color_index[0] // 256) % 256)
        b = ((color_index[0] // (256 * 256)) % 256)
        
        # Skip white (255, 255, 255) as it's our background
        if r == 255 and g == 255 and b == 255:
            color_index[0] += 1
            r = (color_index[0] % 256)
            g = ((color_index[0] // 256) % 256)
            b = ((color_index[0] // (256 * 256)) % 256)
        
        color = (r, g, b)
        self.node_colors[id(node)] = color
        self.color_to_node[color] = node
        
        color_index[0] += 1
        
        # Recursively process children
        for child in node.children:
            self.assign_node_colors(child, time_threshold, color_index)
    
    def get_all_active_nodes_with_info(self, node: Node, time_threshold: float) -> List[Tuple[np.ndarray, float, Tuple[int, int, int]]]:
        """
        Get all active nodes with their positions, distances, and colors.
        
        Returns:
            List of (position, distance, color) tuples
        """
        nodes_info = []
        
        if node.t > time_threshold:
            return nodes_info
        
        node_id = id(node)
        if node_id in self.node_distances and node_id in self.node_colors:
            nodes_info.append((
                node.pos,
                self.node_distances[node_id],
                self.node_colors[node_id]
            ))
        
        for child in node.children:
            nodes_info.extend(self.get_all_active_nodes_with_info(child, time_threshold))
        
        return nodes_info
        """
        Recursively get all edges that exist at the given time.
        
        Args:
            node: Current node
            time_threshold: Only include nodes with t <= time_threshold
            parent_pos: Position of parent node (if any)
        
        Returns:
            List of edges as (start_pos, end_pos) tuples
        """
        edges = []
        
        # Skip this node if it doesn't exist yet at the given time
        if node.t > time_threshold:
            return edges
        
        # Add edge from parent to this node (if parent exists)
        if parent_pos is not None:
            edges.append((parent_pos, node.pos))
        
        # Recursively process children
        for child in node.children:
            edges.extend(self.get_active_edges(child, time_threshold, node.pos))
        
        return edges
    
    def get_active_nodes(self, node: Node, time_threshold: float) -> List[np.ndarray]:
        """
        Recursively get all node positions that exist at the given time.
        
        Args:
            node: Current node
            time_threshold: Only include nodes with t <= time_threshold
        
        Returns:
            List of node positions
        """
        nodes = []
        
        # Skip this node if it doesn't exist yet
        if node.t > time_threshold:
            return nodes
        
        nodes.append(node.pos)
        
        # Recursively process children
        for child in node.children:
            nodes.extend(self.get_active_nodes(child, time_threshold))
        
        return nodes
    
    def calculate_coverage_with_distance(self, time_threshold: float, precision: float = 0.1, 
                                       show_images: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Calculate coverage area and average distance at a given time using Voronoi diagrams.
        
        Args:
            time_threshold: Only include nodes with t <= time_threshold
            precision: Size of each pixel in original units (default 0.1)
            show_images: Whether to display the generated images
        
        Returns:
            Tuple of (coverage_area, average_distance, coverage_image, distance_image)
        """
        # Reset caches
        self.node_distances = {}
        self.node_colors = {}
        self.color_to_node = {}
        
        # Calculate distances and assign colors
        self.calculate_node_distances(self.root, time_threshold)
        self.assign_node_colors(self.root, time_threshold, [1])  # Start from 1 to avoid black
        
        # Get active edges and nodes
        edges = self.get_active_edges(self.root, time_threshold)
        nodes_info = self.get_all_active_nodes_with_info(self.root, time_threshold)
        
        if not nodes_info:
            print(f"No nodes exist at time {time_threshold}")
            return 0.0, 0.0, np.array([]), np.array([])
        
        # Calculate scale factor
        scale = 1.0 / precision
        
        # Find bounding box with margin
        all_positions = np.array([info[0] for info in nodes_info])
        margin = self.radius + 1
        min_x, min_y = all_positions.min(axis=0) - margin
        max_x, max_y = all_positions.max(axis=0) + margin
        
        # Calculate image dimensions
        width = int((max_x - min_x) * scale) + 1
        height = int((max_y - min_y) * scale) + 1
        
        # Create coverage image
        coverage_img = Image.new('RGB', (width, height), 'white')
        coverage_draw = ImageDraw.Draw(coverage_img)
        
        # Line thickness in pixels
        line_width = int(2 * self.radius * scale)
        
        # Draw edges on coverage image
        for start_pos, end_pos in edges:
            x1 = int((start_pos[0] - min_x) * scale)
            y1 = int((max_y - start_pos[1]) * scale)
            x2 = int((end_pos[0] - min_x) * scale)
            y2 = int((max_y - end_pos[1]) * scale)
            
            coverage_draw.line([(x1, y1), (x2, y2)], fill='black', width=line_width)
        
        # Draw circular caps at nodes on coverage image
        for pos, _, _ in nodes_info:
            px = int((pos[0] - min_x) * scale)
            py = int((max_y - pos[1]) * scale)
            radius_pixels = int(self.radius * scale)
            
            coverage_draw.ellipse([(px - radius_pixels, py - radius_pixels),
                                 (px + radius_pixels, py + radius_pixels)], 
                                fill='black')
        
        # Create distance image using Voronoi/KDTree approach
        print(f"Creating distance image using KDTree for {len(nodes_info)} nodes...")
        
        # Prepare node data
        node_positions = np.array([info[0] for info in nodes_info])
        node_colors = [info[2] for info in nodes_info]
        node_distances_list = [info[1] for info in nodes_info]
        
        # Build KDTree for efficient nearest neighbor queries
        kdtree = cKDTree(node_positions)
        
        # Create distance image
        distance_img = Image.new('RGB', (width, height), 'white')
        distance_array = np.array(distance_img)
        
        # Create coordinate grids for all pixels
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Convert pixel coordinates to world coordinates
        world_coords = np.column_stack([
            xx.ravel() * precision + min_x,
            max_y - yy.ravel() * precision  # Fixed: parenthesis was in wrong place
        ])
        
        # Find nearest node for all pixels at once
        _, nearest_indices = kdtree.query(world_coords)
        
        # Reshape back to image dimensions
        nearest_indices = nearest_indices.reshape(height, width)
        
        # Color the distance image based on nearest nodes
        for y in range(height):
            for x in range(width):
                nearest_idx = nearest_indices[y, x]
                distance_array[y, x] = node_colors[nearest_idx]
        
        # Convert coverage image to array
        coverage_array = np.array(coverage_img)
        
        # Calculate average distance for covered pixels
        total_distance = 0.0
        covered_pixels = 0
        
        for y in range(height):
            for x in range(width):
                # Check if pixel is covered (not white in coverage image)
                if not np.all(coverage_array[y, x] == 255):
                    covered_pixels += 1
                    # Get the nearest node index directly
                    nearest_idx = nearest_indices[y, x]
                    node_distance = node_distances_list[nearest_idx]
                    total_distance += node_distance
        
        # Calculate results
        pixel_area = precision ** 2
        coverage_area = covered_pixels * pixel_area
        average_distance = total_distance / covered_pixels if covered_pixels > 0 else 0
        
        print(f"Time {time_threshold}: {len(nodes_info)} nodes, {covered_pixels} pixels covered")
        print(f"Coverage area: {coverage_area:.2f}, Average distance: {average_distance:.2f}")
        
        if show_images:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            ax1.imshow(coverage_array)
            ax1.set_title(f'Coverage at t={time_threshold}\nArea={coverage_area:.2f}')
            ax1.axis('off')
            
            ax2.imshow(distance_array)
            ax2.set_title(f'Nearest Node Map (KDTree)\nAvg Distance={average_distance:.2f}')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return coverage_area, average_distance, coverage_array, distance_array
        """
        Calculate coverage area at a given time using graphics rasterization.
        
        Args:
            time_threshold: Only include nodes with t <= time_threshold
            precision: Size of each pixel in original units (default 0.1)
            show_image: Whether to display the generated image
        
        Returns:
            Tuple of (coverage_area, image_array)
        """
        # Get active edges and nodes at the given time
        edges = self.get_active_edges(self.root, time_threshold)
        nodes = self.get_active_nodes(self.root, time_threshold)
        
        if not nodes:
            print(f"No nodes exist at time {time_threshold}")
            return 0.0, np.array([])
        
        # Calculate scale factor
        scale = 1.0 / precision
        
        # Find bounding box with margin
        all_points = np.array(nodes)
        margin = self.radius + 1
        min_x, min_y = all_points.min(axis=0) - margin
        max_x, max_y = all_points.max(axis=0) + margin
        
        # Calculate image dimensions
        width = int((max_x - min_x) * scale) + 1
        height = int((max_y - min_y) * scale) + 1
        
        # Create image with white background
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Line thickness in pixels (diameter = 2 * radius)
        line_width = int(2 * self.radius * scale)
        
        # Draw edges with thick lines
        for start_pos, end_pos in edges:
            # Convert to pixel coordinates
            x1 = int((start_pos[0] - min_x) * scale)
            y1 = int((max_y - start_pos[1]) * scale)  # Flip y-axis
            x2 = int((end_pos[0] - min_x) * scale)
            y2 = int((max_y - end_pos[1]) * scale)
            
            # Draw thick line
            draw.line([(x1, y1), (x2, y2)], fill='black', width=line_width)
        
        # Draw circular caps at nodes
        for pos in nodes:
            px = int((pos[0] - min_x) * scale)
            py = int((max_y - pos[1]) * scale)
            radius_pixels = int(self.radius * scale)
            
            # Draw filled circle
            draw.ellipse([(px - radius_pixels, py - radius_pixels),
                         (px + radius_pixels, py + radius_pixels)], 
                        fill='black')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Count non-white pixels
        covered_pixels = np.sum(np.any(img_array != 255, axis=2))
        
        # Convert pixel count to area
        pixel_area = precision ** 2
        coverage_area = covered_pixels * pixel_area
        
        print(f"Time {time_threshold}: {len(nodes)} nodes, {len(edges)} edges, "
              f"{covered_pixels} pixels covered, area = {coverage_area:.2f}")
        
        if show_image:
            plt.figure(figsize=(10, 10))
            plt.imshow(img_array)
            plt.title(f'Coverage at t={time_threshold} (area={coverage_area:.2f})')
            plt.axis('off')
            plt.show()
        
        return coverage_area, img_array
    
    def animate_growth(self, time_points: List[float], precision: float = 0.1):
        """
        Visualize how the tree coverage grows over time.
        
        Args:
            time_points: List of time points to visualize
            precision: Pixel precision
        """
        n_times = len(time_points)
        fig, axes = plt.subplots(1, n_times, figsize=(5*n_times, 5))
        
        if n_times == 1:
            axes = [axes]
        
        areas = []
        for i, t in enumerate(time_points):
            area, img = self.calculate_coverage(t, precision, show_image=False)
            areas.append(area)
            
            axes[i].imshow(img)
            axes[i].set_title(f't={t}\nArea: {area:.2f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return areas
    
    def visualize_distance_distribution(self, time_threshold: float, precision: float = 0.1):
        """
        Visualize the distribution of distances for covered pixels.
        
        Args:
            time_threshold: Time threshold for active nodes
            precision: Pixel precision
        """
        # Calculate coverage and distances
        area, avg_dist, coverage_array, distance_array = self.calculate_coverage_with_distance(
            time_threshold, precision, show_images=False)
        
        # Get node information
        nodes_info = self.get_all_active_nodes_with_info(self.root, time_threshold)
        node_positions = np.array([info[0] for info in nodes_info])
        node_distances_list = [info[1] for info in nodes_info]
        
        # Build KDTree for distance lookup
        kdtree = cKDTree(node_positions)
        
        # Collect distances of all covered pixels
        distances = []
        height, width = coverage_array.shape[:2]
        
        # Get bounds
        all_positions = np.array([info[0] for info in nodes_info])
        margin = self.radius + 1
        min_x, min_y = all_positions.min(axis=0) - margin
        max_x, max_y = all_positions.max(axis=0) + margin
        
        for y in range(height):
            for x in range(width):
                if not np.all(coverage_array[y, x] == 255):  # Covered pixel
                    # Convert pixel to world coordinates
                    world_x = x * precision + min_x
                    world_y = max_y - y * precision  # Fixed: parenthesis was in wrong place
                    
                    # Find nearest node
                    _, nearest_idx = kdtree.query([world_x, world_y])
                    distances.append(node_distances_list[nearest_idx])
        
        if not distances:
            print("No covered pixels found")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Coverage image
        axes[0, 0].imshow(coverage_array)
        axes[0, 0].set_title(f'Coverage (Area={area:.2f})')
        axes[0, 0].axis('off')
        
        # 2. Distance-colored coverage
        # Create a colored version where distance maps to color intensity
        distance_colored = np.ones_like(coverage_array) * 255
        max_dist = max(node_distances_list) if node_distances_list else 1
        
        for y in range(height):
            for x in range(width):
                if not np.all(coverage_array[y, x] == 255):
                    # Convert pixel to world coordinates
                    world_x = x * precision + min_x
                    world_y = max_y - y * precision  # Fixed: parenthesis was in wrong place
                    
                    # Find nearest node
                    _, nearest_idx = kdtree.query([world_x, world_y])
                    dist = node_distances_list[nearest_idx]
                    
                    # Map distance to color intensity (closer = darker blue)
                    intensity = int(255 * (1 - dist / max_dist))
                    distance_colored[y, x] = [intensity, intensity, 255]
        
        axes[0, 1].imshow(distance_colored.astype(np.uint8))
        axes[0, 1].set_title(f'Distance Heatmap (Avg={avg_dist:.2f})')
        axes[0, 1].axis('off')
        
        # 3. Distance histogram
        axes[1, 0].hist(distances, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(avg_dist, color='red', linestyle='--', linewidth=2, 
                          label=f'Average: {avg_dist:.2f}')
        axes[1, 0].set_xlabel('Distance from Root')
        axes[1, 0].set_ylabel('Number of Pixels')
        axes[1, 0].set_title('Distance Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Tree structure with distances
        axes[1, 1].set_aspect('equal')
        
        # Draw edges
        edges = self.get_active_edges(self.root, time_threshold)
        for start_pos, end_pos in edges:
            axes[1, 1].plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                           'k-', linewidth=1, alpha=0.5)
        
        # Draw nodes colored by distance
        positions = np.array([info[0] for info in nodes_info])
        distances_list = [info[1] for info in nodes_info]
        
        scatter = axes[1, 1].scatter(positions[:, 0], positions[:, 1], 
                                   c=distances_list, cmap='viridis', 
                                   s=100, edgecolors='black', linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Distance from Root')
        
        axes[1, 1].set_title('Tree Structure (Nodes colored by distance)')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nDistance Statistics:")
        print(f"  Minimum: {min(distances):.2f}")
        print(f"  Maximum: {max(distances):.2f}")
        print(f"  Average: {avg_dist:.2f}")
        print(f"  Median: {np.median(distances):.2f}")
        print(f"  Std Dev: {np.std(distances):.2f}")
    
    def plot_coverage_over_time(self, max_time: float, n_points: int = 50, precision: float = 0.1):
        """
        Plot how coverage area changes over time.
        
        Args:
            max_time: Maximum time to plot
            n_points: Number of time points to sample
            precision: Pixel precision for calculations
        """
        time_points = np.linspace(0, max_time, n_points)
        areas = []
        
        for t in time_points:
            area, _ = self.calculate_coverage(t, precision, show_image=False)
            areas.append(area)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, areas, 'b-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Coverage Area')
        plt.title('Tree Coverage Area Over Time')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return time_points, areas
    
    def visualize_voronoi_diagram(self, time_threshold: float):
        """
        Visualize the Voronoi diagram of the tree nodes.
        
        Args:
            time_threshold: Time threshold for active nodes
        """
        # Get node information
        nodes_info = self.get_all_active_nodes_with_info(self.root, time_threshold)
        if len(nodes_info) < 3:
            print("Need at least 3 nodes to create Voronoi diagram")
            return
        
        node_positions = np.array([info[0] for info in nodes_info])
        node_distances = [info[1] for info in nodes_info]
        
        # Create Voronoi diagram
        vor = Voronoi(node_positions)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot Voronoi diagram
        voronoi_plot_2d(vor, ax=ax1, show_vertices=False, line_colors='blue', line_width=2)
        
        # Plot nodes
        scatter1 = ax1.scatter(node_positions[:, 0], node_positions[:, 1], 
                              c=node_distances, cmap='viridis', 
                              s=100, edgecolors='black', linewidth=2, zorder=5)
        
        # Draw tree edges
        edges = self.get_active_edges(self.root, time_threshold)
        for start_pos, end_pos in edges:
            ax1.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                    'k-', linewidth=2, alpha=0.7, zorder=4)
        
        ax1.set_title('Voronoi Diagram of Tree Nodes')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Distance from Root')
        
        # Plot coverage with Voronoi regions
        ax2.set_aspect('equal')
        
        # Draw Voronoi regions with colors based on distance
        from matplotlib.patches import Polygon
        from matplotlib.colors import Normalize
        from matplotlib.cm import viridis
        
        norm = Normalize(vmin=min(node_distances), vmax=max(node_distances))
        
        for i, (point_idx, region_idx) in enumerate(zip(range(len(node_positions)), vor.point_region)):
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                polygon_vertices = [vor.vertices[idx] for idx in region]
                if len(polygon_vertices) > 2:  # Valid polygon
                    color = viridis(norm(node_distances[i]))
                    poly = Polygon(polygon_vertices, facecolor=color, 
                                 edgecolor='black', alpha=0.6)
                    ax2.add_patch(poly)
        
        # Draw tree structure on top
        for start_pos, end_pos in edges:
            ax2.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                    'k-', linewidth=3, alpha=0.8)
        
        # Plot nodes
        scatter2 = ax2.scatter(node_positions[:, 0], node_positions[:, 1], 
                              c='red', s=150, edgecolors='black', 
                              linewidth=2, zorder=5)
        
        # Draw coverage circles around edges
        for start_pos, end_pos in edges:
            # Sample points along edge
            t_values = np.linspace(0, 1, 20)
            for t in t_values:
                point = start_pos + t * (end_pos - start_pos)
                circle = plt.Circle(point, self.radius, fill=False, 
                                  edgecolor='red', linewidth=1, alpha=0.3)
                ax2.add_patch(circle)
        
        ax2.set_title(f'Voronoi Regions with Tree Coverage (r={self.radius})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        
        # Set axis limits
        margin = self.radius + 2
        all_points = np.vstack([node_positions, vor.vertices])
        ax2.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax2.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax1.set_xlim(ax2.get_xlim())
        ax1.set_ylim(ax2.get_ylim())
        
        plt.tight_layout()
        plt.show()


# Add backward compatibility method
    def calculate_coverage(self, time_threshold: float, precision: float = 0.1, 
                         show_image: bool = False) -> Tuple[float, np.ndarray]:
        """
        Calculate coverage area at a given time (backward compatibility).
        """
        area, _, coverage_array, _ = self.calculate_coverage_with_distance(
            time_threshold, precision, show_image)
        return area, coverage_array
        """
        Plot how coverage area changes over time.
        
        Args:
            max_time: Maximum time to plot
            n_points: Number of time points to sample
            precision: Pixel precision for calculations
        """
        time_points = np.linspace(0, max_time, n_points)
        areas = []
        
        for t in time_points:
            area, _ = self.calculate_coverage(t, precision, show_image=False)
            areas.append(area)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, areas, 'b-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Coverage Area')
        plt.title('Tree Coverage Area Over Time')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return time_points, areas


# Helper function to build a tree from the Node structure
def print_tree(node: Node, prefix: str = "", time_threshold: float = float('inf')):
    """Print tree structure for debugging."""
    if node.t <= time_threshold:
        print(f"{prefix}Node at {node.pos} (t={node.t})")
        for i, child in enumerate(node.children):
            is_last = i == len(node.children) - 1
            extension = "└── " if is_last else "├── "
            print_tree(child, prefix + extension, time_threshold)


# Example usage
def create_growing_tree_example():
    """Create an example of a tree that grows over time."""
    # Root node at time 0
    root = Node([0, 0], 0)
    
    # First generation children at time 1
    child1 = Node([5, 0], 1)
    child2 = Node([-3, 3], 1)
    root.children = [child1, child2]
    
    # Second generation at time 2
    child1_1 = Node([8, 2], 2)
    child1_2 = Node([7, -3], 2)
    child1.children = [child1_1, child1_2]
    
    child2_1 = Node([-6, 5], 2)
    child2.children = [child2_1]
    
    # Third generation at time 3
    child1_1_1 = Node([12, 4], 3)
    child1_1.children = [child1_1_1]
    
    child2_1_1 = Node([-8, 8], 3)
    child2_1_2 = Node([-9, 3], 3)
    child2_1.children = [child2_1_1, child2_1_2]
    
    return root


def test_time_aware_coverage():
    """Test the time-aware coverage calculator."""
    # Create a growing tree
    root = create_growing_tree_example()
    radius = 1.5
    
    # Initialize calculator
    calc = TimeAwareTreeCoverage(root, radius)
    
    print("Tree structure:")
    print_tree(root)
    print("\n" + "="*60 + "\n")
    
    # Calculate coverage at different times
    time_points = [0, 1, 2, 3, 4]
    
    for t in time_points:
        print(f"\nCoverage at time t={t}:")
        area, img = calc.calculate_coverage(t, precision=0.1, show_image=True)
    
    # Animate growth
    print("\n" + "="*60)
    print("Animating tree growth:")
    calc.animate_growth([0, 1, 2, 3], precision=0.1)
    
    # Plot coverage over time
    print("\n" + "="*60)
    print("Coverage area over time:")
    calc.plot_coverage_over_time(max_time=4, n_points=41, precision=0.1)


def create_radial_growth_example():
    """Create a tree that grows radially outward over time."""
    # Center root
    root = Node([0, 0], 0)
    
    # Create radial growth pattern
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    
    # First ring at time 1
    for angle in angles[:3]:  # Only 3 branches initially
        x = 5 * np.cos(angle)
        y = 5 * np.sin(angle)
        child = Node([x, y], 1)
        root.children.append(child)
        
        # Second ring at time 2
        for offset in [-0.3, 0.3]:
            x2 = 10 * np.cos(angle + offset)
            y2 = 10 * np.sin(angle + offset)
            grandchild = Node([x2, y2], 2)
            child.children.append(grandchild)
    
    # Add more branches at time 3
    for angle in angles[3:]:
        x = 5 * np.cos(angle)
        y = 5 * np.sin(angle)
        late_child = Node([x, y], 3)
        root.children.append(late_child)
    
    return root


def test_distance_aware_coverage():
    """Test the distance-aware coverage calculation."""
    print("DISTANCE-AWARE COVERAGE TEST")
    print("="*60)
    
    # Create a tree with clear distance progression
    root = Node([0, 0], 0)
    
    # First level - distance 5 from root
    branch1 = Node([5, 0], 1)
    branch2 = Node([0, 5], 1)
    branch3 = Node([-5, 0], 1)
    root.children = [branch1, branch2, branch3]
    
    # Second level - distance 10 from root (5 + 5)
    branch1_1 = Node([10, 0], 2)
    branch1_2 = Node([5, 5], 2)
    branch1.children = [branch1_1, branch1_2]
    
    branch2_1 = Node([0, 10], 2)
    branch2_2 = Node([5, 8], 2)
    branch2.children = [branch2_1, branch2_2]
    
    # Third level - even further
    branch1_1_1 = Node([15, 0], 3)
    branch1_1_2 = Node([12, 5], 3)
    branch1_1.children = [branch1_1_1, branch1_1_2]
    
    # Initialize calculator with radius 2
    calc = TimeAwareTreeCoverage(root, radius=2.0)
    
    # Calculate and visualize at different times
    for t in [1, 2, 3]:
        print(f"\n\nAnalyzing at time t={t}:")
        print("-"*40)
        
        area, avg_dist, coverage_img, distance_img = calc.calculate_coverage_with_distance(
            t, precision=0.1, show_images=True)
        
        # Show detailed visualization
        calc.visualize_distance_distribution(t, precision=0.1)
    
    # Visualize Voronoi diagram
    print("\n\nVoronoi Diagram Visualization:")
    calc.visualize_voronoi_diagram(time_threshold=2)
    
    # Plot how average distance changes over time
    print("\n\nAverage Distance Over Time:")
    print("-"*40)
    
    time_points = np.linspace(0, 3.5, 36)
    areas = []
    avg_distances = []
    
    for t in time_points:
        area, avg_dist, _, _ = calc.calculate_coverage_with_distance(
            t, precision=0.1, show_images=False)
        areas.append(area)
        avg_distances.append(avg_dist)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(time_points, areas, 'b-', linewidth=2)
    ax1.set_ylabel('Coverage Area')
    ax1.set_title('Tree Growth Metrics Over Time')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_points, avg_distances, 'r-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Average Distance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def benchmark_distance_calculation():
    """Benchmark the performance of distance calculation with KDTree optimization."""
    print("\n\nPERFORMANCE BENCHMARK FOR DISTANCE CALCULATION")
    print("="*60)
    
    # Create a larger tree
    np.random.seed(42)
    root = Node([0, 0], 0)
    
    def create_balanced_tree(parent, level, max_level, branch_factor=3):
        if level >= max_level:
            return
        
        angle_step = 2 * np.pi / branch_factor
        for i in range(branch_factor):
            angle = i * angle_step + np.random.uniform(-0.2, 0.2)
            distance = 5 + np.random.uniform(-1, 1)
            x = parent.pos[0] + distance * np.cos(angle)
            y = parent.pos[1] + distance * np.sin(angle)
            
            child = Node([x, y], level)
            parent.children.append(child)
            create_balanced_tree(child, level + 1, max_level, branch_factor)
    
    create_balanced_tree(root, 1, 4, 3)
    
    calc = TimeAwareTreeCoverage(root, radius=1.5)
    
    # Count nodes
    def count_nodes(node, time_threshold):
        if node.t > time_threshold:
            return 0
        return 1 + sum(count_nodes(child, time_threshold) for child in node.children)
    
    n_nodes = count_nodes(root, 10)
    print(f"Tree has {n_nodes} total nodes")
    
    # Benchmark different precisions
    print("\nBenchmarking KDTree-based approach:")
    print("-" * 40)
    for precision in [0.2, 0.1, 0.05]:
        print(f"\nPrecision = {precision}:")
        
        start_time = time.time()
        area, avg_dist, _, _ = calc.calculate_coverage_with_distance(
            10, precision=precision, show_images=False)
        elapsed = time.time() - start_time
        
        # Calculate image size for reference
        nodes_info = calc.get_all_active_nodes_with_info(calc.root, 10)
        all_positions = np.array([info[0] for info in nodes_info])
        margin = calc.radius + 1
        min_x, min_y = all_positions.min(axis=0) - margin
        max_x, max_y = all_positions.max(axis=0) + margin
        width = int((max_x - min_x) / precision) + 1
        height = int((max_y - min_y) / precision) + 1
        
        print(f"  Image size: {width}x{height} ({width*height:,} pixels)")
        print(f"  Area: {area:.2f}")
        print(f"  Avg Distance: {avg_dist:.2f}")
        print(f"  Time: {elapsed:.3f} seconds")
    
    # Visualize the Voronoi diagram
    print("\n\nVisualizing Voronoi diagram:")
    calc.visualize_voronoi_diagram(time_threshold=3)


def test_voronoi_correctness():
    """Test to verify Voronoi diagram is correctly calculated in 2D."""
    print("\n\nVORONOI CORRECTNESS TEST")
    print("="*60)
    
    # Create a simple symmetric tree to clearly show 2D Voronoi
    root = Node([0, 0], 0)
    
    # Four nodes in a square pattern
    node1 = Node([5, 5], 1)    # Top-right
    node2 = Node([-5, 5], 1)   # Top-left
    node3 = Node([-5, -5], 1)  # Bottom-left
    node4 = Node([5, -5], 1)   # Bottom-right
    
    root.children = [node1, node2, node3, node4]
    
    # Add some children to make it interesting
    node1.children = [Node([10, 10], 2)]
    node3.children = [Node([-10, -10], 2)]
    
    calc = TimeAwareTreeCoverage(root, radius=1.0)
    
    # Test at different times
    for t in [0, 1, 2]:
        print(f"\n\nTime t={t}:")
        print("-"*40)
        
        # Calculate coverage with distance
        area, avg_dist, coverage_img, distance_img = calc.calculate_coverage_with_distance(
            t, precision=0.1, show_images=False)
        
        # Create a figure to show the distance image clearly
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show the distance image
        ax1.imshow(distance_img)
        ax1.set_title(f'Nearest Node Regions at t={t}')
        ax1.axis('off')
        
        # Show the Voronoi diagram if there are enough nodes
        nodes_info = calc.get_all_active_nodes_with_info(calc.root, t)
        if len(nodes_info) >= 3:
            node_positions = np.array([info[0] for info in nodes_info])
            vor = Voronoi(node_positions)
            
            voronoi_plot_2d(vor, ax=ax2, show_vertices=False, 
                          line_colors='blue', line_width=2)
            ax2.scatter(node_positions[:, 0], node_positions[:, 1], 
                       c='red', s=100, zorder=5)
            ax2.set_title(f'Voronoi Diagram at t={t}')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')
        else:
            ax2.text(0.5, 0.5, f'Need at least 3 nodes\n(currently {len(nodes_info)})', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Active nodes: {len(nodes_info)}")
        print(f"Coverage area: {area:.2f}")
        print(f"Average distance: {avg_dist:.2f}")


# Update main execution
if __name__ == "__main__":
    # Test basic growing tree
    print("EXAMPLE 1: Basic Growing Tree")
    print("="*60)
    test_time_aware_coverage()
    
    # Test Voronoi correctness
    test_voronoi_correctness()
    
    # Test distance-aware coverage
    print("\n\n")
    test_distance_aware_coverage()
    
    # Benchmark performance
    benchmark_distance_calculation()
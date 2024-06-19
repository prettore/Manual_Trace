# Mobility Pattern Simulation Tool

This script provides a graphical interface for drawing and simulating mobility patterns for network nodes. It allows users to place nodes on a grid, generate trace files, and create visual representations of node movement and network states over time.

## Features

- **Grid Drawing Interface**: Draw nodes on a grid to simulate their movement over time.
- **Trace File Generation**: Save node coordinates and timestamps into a CSV file.
- **Equidistant Point Generation**: Generate equidistant points between nodes for detailed trace analysis.
- **State Mapping**: Match node locations to predefined states based on distances.
- **Distance Computation**: Calculate distances between nodes at different timestamps.
- **Visualization**: Create animations and static plots to visualize node movements and network states.

## Requirements

- Python 3.x
- Tkinter
- Pandas
- Numpy
- Matplotlib
- Imageio
- Scipy

## Usage

### 1. Set Up the Environment

Ensure you have the necessary libraries installed. You can install them using:

```bash
pip install pandas numpy matplotlib imageio scipy
```

### 2. Running the Script

Execute the script by running:

```bash
python mobility_pattern_simulation.py
```

### 3. Using the Interface

- **Drawing Nodes**: Click on the grid to place nodes. The coordinates and timestamps will be recorded.
- **Node Selection**: Use the dropdown menu to select which node to place on the grid.
- **Setting Time**: Enter the current timestamp in the provided entry box before placing nodes.

### 4. Generating Trace Files

The script saves the node coordinates and timestamps in a CSV file located in the `data/drawing_trace/` directory.

### 5. Creating Visualizations

#### Animation

To create an animation of the node movements:

```python
create_animation(data, img_folder, file_description)
```

This function generates a GIF file in the `data/img/` directory.

#### Static Plot

To create a static plot:

```python
create_static_plot(data, img_folder, file_description)
```

This function generates a PDF file in the `data/img/` directory.

## Functions

### SquareGrid Class

- **__init__(self, master, size, scale_factor)**: Initializes the grid with the specified size and scale factor.
- **on_click(self, event)**: Handles the click event to place nodes on the grid.
- **set_node_selector(self, options)**: Sets the dropdown menu for node selection.
- **set_node_colors(self, colors)**: Sets colors for different nodes.
- **set_node_ranges(self, ranges)**: Sets the range for nodes.
- **set_marker_radius(self, radius)**: Sets the marker radius for nodes.
- **set_time_entry(self, entry)**: Sets the entry widget for time input.

### Utility Functions

- **lerp(v0, v1, i)**: Linear interpolation between two values.
- **get_equidistant_points(trace, n)**: Generates equidistant points between nodes.
- **creatingFolders(dataFolder)**: Creates a directory if it doesn't exist.
- **get_trace(trace_file, csv)**: Reads trace files and returns a DataFrame.
- **datarate_match(trace, rings, trace2)**: Matches node locations to states based on distances.
- **compute_distance(node1, node2)**: Computes the distance between two nodes.
- **add_datarate(trace, communication_dic)**: Adds datarate information to the trace.
- **create_animation(data, img_folder, file_description)**: Creates an animation of the node movements.
- **create_static_plot(data, img_folder, file_description)**: Creates a static plot of the node movements.

## Example

1. Place nodes on the grid using the graphical interface.
2. Save the trace file with node coordinates and timestamps.
3. Use the `create_animation` function to generate an animation.
4. Use the `create_static_plot` function to generate a static plot.

## Directory Structure

- **data/drawing_trace/**: Directory to save trace files.
- **data/img/**: Directory to save generated images (GIFs and PDFs).

## License

This project is licensed under the MIT License.

## Author

Paulo H. L. Rettore

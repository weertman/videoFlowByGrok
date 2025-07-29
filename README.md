# videoFlowByGrok
optical flow ui made by grok heavy

# Optical Flow Visualizer

## Overview

This is a desktop application built with PySide6 (Qt for Python) and OpenCV for visualizing optical flow in videos. It supports dense optical flow algorithms like Farneback and DIS (Dense Inverse Search), as well as a custom "FlowTrace" method for tracing motion over multiple frames. Users can load videos, select frame ranges, adjust parameters, process visualizations, and export results as videos, image sequences, or NumPy arrays.

The application provides a user-friendly GUI for previewing videos, configuring visualization settings, and viewing real-time processing results.

## Features

- **Video Loading and Preview**: Browse or drag-and-drop videos (MP4, AVI, MOV, MKV). Display metadata (resolution, FPS, duration, codec) and preview playback with seeking.
- **Frame Selection**: Choose start/end frames, process the entire video, or set a frame step for subsampling.
- **Algorithms**:
  - Farneback: Dense optical flow using Gunnar Farneback's polynomial expansion.
  - Dense Optical Flow: Uses OpenCV's DIS implementation for robust dense flow.
  - FlowTrace: Custom method that stacks frames, optionally inverts or subtracts background, and computes max projection for motion traces.
- **Visualization Parameters**:
  - Window size/trace length, pyramid levels, iterations (for flow algorithms).
  - Magnitude threshold, arrow density/size (for flow visualizations).
  - Color maps (HSV, Jet, Viridis, Grayscale), smoothing, invert frames, background subtraction (for FlowTrace).
- **Processing**: Multi-threaded for non-blocking UI. Progress bar and error handling.
- **Output**: Side-by-side preview of original and visualized frames. Export as MP4 video, PNG image sequence, or .npy NumPy array.
- **Logging**: Errors and events logged to `logs/app.log`.

## Requirements

- Python 3.8+
- PySide6 (for GUI)
- OpenCV (cv2)
- NumPy

Optional: Matplotlib (used in utils.py for colormaps, but can be replaced if needed).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/optical-flow-visualizer.git
   cd optical-flow-visualizer
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install PySide6 opencv-python numpy
   ```

4. Ensure the project structure is set up with `src/` containing the modules (`ui.py`, `worker.py`, `processors.py`, `utils.py`).

## Usage

1. Run the application:
   ```
   python -m src.main
   ```

2. **Load a Video**:
   - Click "Browse Video" or drag-drop a video file.
   - Metadata and preview will appear.

3. **Select Frames**:
   - Set start/end frames or check "Process Entire Video".
   - Adjust frame step for processing every Nth frame.

4. **Configure Parameters**:
   - Choose an algorithm from the dropdown.
   - Adjust sliders/spinboxes for visualization settings (parameters adapt based on algorithm).

5. **Process**:
   - Click "Process" to start computing and visualizing flow.
   - Watch progress and previews update.

6. **Export**:
   - Click "Export" and choose format (MP4, Image Sequence, NumPy Array).
   - Save to desired location.

### Example

For a video of moving objects:
- Select "FlowTrace" with trace length 10, background subtraction enabled, and "jet" colormap.
- Process a short range to visualize motion trails.

## Project Structure

- `main.py`: Application entry point.
- `src/ui.py`: GUI definition using PySide6.
- `src/worker.py`: Threaded workers for video loading and flow processing.
- `src/processors.py`: Core computation functions for flow algorithms.
- `src/utils.py`: Helper functions for metadata and visualizations.
- `logs/`: Directory for app logs (created automatically).

## Known Limitations

- FlowTrace assumes grayscale frames and may clip values during processing.
- Arrow visualization in flow is simplified (grid-based).
- No real-time processing for very long videos; use frame stepping.
- Stylesheet loading is commented out (optional, requires `assets/styles.qss`).

## Contributing

Contributions are welcome! Please open an issue or pull request for bug fixes, new features, or improvements.

## License

MIT License. See [LICENSE](LICENSE) for details.

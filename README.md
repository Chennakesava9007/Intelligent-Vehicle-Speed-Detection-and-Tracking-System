# Intelligent-Vehicle-Speed-Detection-and-Tracking-System

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)

This project provides a robust solution for detecting vehicles in a video stream, tracking their movement across frames, and estimating their speed in real-time. It uses the state-of-the-art YOLOv10 model for object detection and the DeepSORT algorithm for persistent tracking. Vehicles exceeding a user-defined speed limit are highlighted for easy identification.



---

## Key Features

* **High-Performance Object Detection**: Utilizes the efficient and accurate **YOLOv10** model.
* **Multi-Object Tracking**: Implements **DeepSORT** to maintain a unique ID for each vehicle as it moves through the frame.
* **Speed Estimation**: Calculates the real-world speed (in km/h) by applying a perspective transformation to map pixel coordinates to a bird's-eye view.
* **Overspeeding Alerts**: Bounding boxes of vehicles exceeding the specified speed limit are highlighted in red.
* **User-Friendly Interface**: Includes a simple Tkinter GUI for easy operation.
* **Flexible CLI**: Can also be run via the command line for automation and advanced use cases.

---

## Technology Stack

* **Python**
* **PyTorch**: For running the deep learning model.
* **Ultralytics YOLO**: For the YOLOv10 object detection implementation.
* **OpenCV**: For video processing and image manipulation.
* **DeepSORT**: For real-time object tracking.
* **NumPy**: For numerical calculations and perspective transformations.
* **Tkinter**: For the graphical user interface.

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
This project uses a `requirements.txt` file to manage all necessary libraries.
```bash
pip install -r requirements.txt
```
> **Note**: If you don't have a `requirements.txt` file yet, you can create one after installing the libraries manually with this command:
> `pip freeze > requirements.txt`

### 4. Project Structure
Ensure your project directory is organized as follows. The `configs/coco.names` file and `yolov10n.pt` model are required.
```
/your-repository-name
|-- /configs
|   |-- coco.names
|-- /output
|   |-- (Generated videos are saved here)
|-- gui_app.py
|-- object_tracking.py
|-- yolov10n.pt
|-- requirements.txt
|-- README.md
```

---

## How to Use

You can run the project using either the GUI or the command line.

### Method 1: Using the GUI (Recommended)
The graphical interface is the simplest way to get started.

1.  Make sure your virtual environment is activated.
2.  Run the application:
    ```bash
    python gui_app.py
    ```
3.  In the application window:
    * Click **"Select Video"** to choose your input file.
    * Set the **"Speed Limit (km/h)"**.
    * Click **"Start Speed Detection"**.

### Method 2: Using the Command Line
For more control and automation, use the `object_tracking.py` script directly.

```bash
python object_tracking.py --video "path/to/your/video.mp4" --output "output/result.mp4" --speed_limit 60
```

**Available Arguments:**
* `--video`: Path to the input video file (required).
* `--output`: Path to save the processed output video (required).
* `--speed_limit`: Integer value for the speed limit in km/h (default: 60).
* `--conf`: Confidence threshold for object detection (default: 0.50).

---

## Roadmap: From Desktop GUI to Web Application

The current Tkinter GUI is functional, but a web-based application would be more modern, accessible, and scalable. Here's a high-level roadmap to achieve that.

### 1. **Backend Development (API Server)**
The backend will handle the core logic and video processing.
* **Framework**: Choose a lightweight Python web framework like **Flask** or **FastAPI**. FastAPI is recommended for its speed and automatic API documentation.
* **Functionality**:
    * Create an API endpoint (e.g., `/process_video`) that accepts video file uploads.
    * Adapt the existing `object_tracking.py` logic into a function that can be called by this endpoint.
    * The function will take the uploaded video, process it, and save the result.
    * The API should return a response, such as the path to the processed video or a job ID for tracking progress.
    * Consider using WebSockets to send real-time processing updates (e.g., "Frame 100/2500 processed") back to the user.

### 2. **Frontend Development (User Interface)**
The frontend is the website that users will interact with in their browser.
* **Framework**: You can use a modern JavaScript framework like **React** or **Vue.js** for a dynamic user experience, or start with simple **HTML, CSS, and JavaScript**.
* **Functionality**:
    * Create a clean and intuitive page with a file upload component.
    * Add an input field for the user to set the speed limit.
    * When the user submits the form, the frontend will send the video file and settings to the backend API.
    * Display a progress bar or status updates received from the backend.
    * Once processing is complete, display the resulting video for the user to view and download.

### 3. **Connecting Frontend & Backend**
* The frontend will use the `fetch` API or a library like `axios` in JavaScript to make HTTP requests to the backend.
* The backend will be configured to handle these requests, including Cross-Origin Resource Sharing (CORS) if the frontend and backend are served from different domains.

This approach separates the user interface from the heavy processing logic, creating a more robust and professional application.

---

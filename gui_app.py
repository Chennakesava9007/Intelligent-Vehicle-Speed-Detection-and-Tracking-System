import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import sys

print("GUI script is starting...")

class SpeedEstimationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Speed Estimation")
        self.root.geometry("500x400")

        self.video_path = None

        # --- GUI Components ---
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Vehicle Speed Estimation", font=("Arial", 16)).pack(pady=10)

        # Video Selection
        tk.Button(self.root, text="Select Video", command=self.select_video).pack(pady=5)
        self.video_label = tk.Label(self.root, text="No file selected")
        self.video_label.pack()

        # Speed Threshold
        tk.Label(self.root, text="Speed Limit (km/h):").pack(pady=5)
        self.speed_entry = tk.Entry(self.root)
        self.speed_entry.insert(0, "60")
        self.speed_entry.pack()

        # Process Button
        tk.Button(self.root, text="Start Speed Detection", command=self.run_detection).pack(pady=20)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=os.path.basename(file_path))

    def run_detection(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        speed_limit = self.speed_entry.get()

        if not speed_limit.isdigit():
            messagebox.showerror("Error", "Speed must be a number.")
            return

        # Ensure output directory exists
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "result.mp4")

        # Use the exact path to the current Python executable to ensure venv is used
        python_path = sys.executable

        cmd = [
            python_path,
            "object_tracking.py",
            "--video", self.video_path,
            "--output", output_path,
            "--speed_limit", speed_limit
        ]

        subprocess.run(cmd)

        # Show success and open video (optional)
        messagebox.showinfo("Success", f"Processing complete!\nSaved to:\n{output_path}")
        try:
            os.startfile(output_path)  # This opens the video file
        except Exception as e:
            print("Could not open video:", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedEstimationGUI(root)
    root.mainloop()
import csv
import glob
import os
import tkinter as tk
from tkinter import Label, Button, Entry, simpledialog, filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pygrabber.dshow_graph import FilterGraph
import time
import processing
import matplotlib.pyplot as plt

# Initialize the camera
cap = cv2.VideoCapture("noFeed.jpeg")
# Global variables to hold schedule intervals (in milliseconds) and current index
intervals = []
current_interval_index = 0
count=0
size_cm = []
folder=''
capture_count=0
phase_name = ["Vials","foams","slightly turbids","transparent","very turbids"]
description_entries = []
timer_job = None

def choose_input():
    global cap

    choice = messagebox.askquestion("Input Source", "Run from live camera?", icon="question")
    if choice == "yes":
        graph = FilterGraph()
        cams = graph.get_input_devices()
        if not cams:
            return messagebox.showerror("No Cameras","No webcams found.")

        # Ask user to pick by name
        cam_name = simpledialog.askstring(
            "Select Camera",
            "Available cameras:\n" + "\n".join(f"{i}: {n}" for i,n in enumerate(cams)) +
            "\n\nEnter the index of your choice:"
        )
        if cam_name is None:
            return

        try:
            idx = int(cam_name)
            if idx < 0 or idx >= len(cams):
                raise ValueError
        except ValueError:
            return messagebox.showerror("Invalid selection","Please enter a valid index.")

        cap.release()
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    else:
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video","*.mp4 *.avi *.mov")])
        if not path:
            return
        cap.release()
        cap = cv2.VideoCapture(path)

def update_feed():
    # Read frame from the camera and update the live feed panel
    ret, frame = cap.read()
    if ret:
        # Convert from BGR to RGB for PIL display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_panel.imgtk = imgtk
        video_panel.configure(image=imgtk)
    # Schedule next update after 10 milliseconds
    video_panel.after(10, update_feed)


def capture_and_detect(counts=999):
    # Capture one frame for object detection
    start = time.time()
    global capture_count
    global size_cm
    global count
    global folder

    ret, frame = cap.read()
    if ret:
        detected_frame = frame.copy()
        processed,new_count,finalVials = processing.ProcessImage(detected_frame,size_cm)
        # Convert processed frame to RGB and update the detection panel
        detected_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img_detected = Image.fromarray(detected_rgb)
        imgtk_detected = ImageTk.PhotoImage(image=img_detected)
        detected_panel.imgtk = imgtk_detected
        detected_panel.configure(image=imgtk_detected)
        if counts != new_count:
            capture_count=0
            size_cm.clear()
            for i in range(new_count):
                    size_cm.append(simpledialog.askfloat(
                        title=f"Enter size for vial {i}",
                        prompt=f"Please type the total size (cm) of vial {i}:",
                        minvalue=0.5
                    ))
                    # Store into a variable or dict
            clear_textboxes()
            create_textboxes(new_count)
            count = new_count
            folder = filedialog.askdirectory(title="Choose folder to save CSVs")
            stop_timer()
            if not folder:
                return
        capture_count +=1
        save_vial_phase_heights(finalVials,capture_count,folder)
        capture_button.configure(state="normal")

def clear_textboxes():
    for widget in entry_container.winfo_children():
        widget.destroy()
def auto_capture():
    global current_interval_index
    capture_and_detect()
    # If there are more intervals scheduled, set up the next capture
    if current_interval_index < len(intervals):
        root.after(intervals[current_interval_index], auto_capture)
        current_interval_index += 1
    else:
        print("Capture schedule complete.")




def create_textboxes(count):
    global description_entries
    # Clear old widgets & list
    for widget in entry_container.winfo_children():
        widget.destroy()
    description_entries = []

    for i in range(count):
        tk.Label(entry_container, text=f"Vial {i+1}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
        entry = tk.Entry(entry_container, width=40)
        entry.grid(row=i, column=1, padx=5, pady=2)
        description_entries.append(entry)

def get_descriptions():
    """Call this anytime to get a Python list of all current textbox values."""
    return [entry.get() for entry in description_entries]

def create_plot_window_for_vial(vd):
    """
    Create a Toplevel window for one vial folder.
    vd: the path to the vial folder (e.g. "vial_1 ...")
    """
    # Get the CSV files for this vial folder
    files = sorted(glob.glob(os.path.join(vd, "phase_*.csv")))
    if not files:
        return

    # Each window gets its own independent state.
    idx = {"i": 0}

    # Create a new Toplevel window.
    top = tk.Toplevel(root)
    top.title(os.path.basename(vd))

    # Create a matplotlib figure and canvas for this window.
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    def plot():
        """Plot the CSV data corresponding to the current index."""
        ax.clear()
        with open(files[idx["i"]]) as f:
            reader = csv.DictReader(f)
            # Read data as list of (capture_count, height)
            data = sorted((int(r["capture_count"]), float(r["height"])) for r in reader)
        x, y = zip(*data)
        ax.plot(x, y, marker="o")
        ax.set_title(os.path.basename(files[idx["i"]]).replace(".csv", ""))
        ax.set_xlabel("Capture Count")
        ax.set_ylabel("Height (cm)")
        canvas.draw()

    def next_plot():
        """Move to the next CSV file."""
        idx["i"] = (idx["i"] + 1) % len(files)
        plot()

    def prev_plot():
        """Move to the previous CSV file."""
        idx["i"] = (idx["i"] - 1) % len(files)
        plot()

    # Create navigation buttons inside a frame.
    nav_frame = ttk.Frame(top)
    ttk.Button(nav_frame, text="Prev", command=prev_plot).pack(side="left", padx=5)
    ttk.Button(nav_frame, text="Next", command=next_plot).pack(side="right", padx=5)
    nav_frame.pack(fill="x", pady=5)

    # Plot the first CSV file.
    plot()

def open_plot_window():
    """
    Ask the user for a root folder containing vial data. For each vial folder found,
    create a separate Toplevel window with its own navigation controls.
    """
    root_folder = filedialog.askdirectory(title="Select Root Folder")
    if not root_folder:
        return

    vial_dirs = sorted(glob.glob(os.path.join(root_folder, "vial_*")))
    if not vial_dirs:
        return messagebox.showinfo("No Data", "No vial data found")

    for vd in vial_dirs:
        create_plot_window_for_vial(vd)


def save_vial_phase_heights(vial_groups, capture_count: int, output_folder="vial_data"):

    os.makedirs(output_folder, exist_ok=True)

    descs = get_descriptions()

    for vial_index, group in enumerate(vial_groups, start=1):
        desc = descs[vial_index - 1].strip()
        if not desc:
            messagebox.showwarning(
                title="Missing Description",
                message=f"Description for Vial {vial_index} is empty — please fill it in before saving."
            )
            return  # abort the entire save until the user corrects it

        vial_folder = os.path.join(output_folder, f"vial_{vial_index} {desc}")
        os.makedirs(vial_folder, exist_ok=True)

        for phase in group:
            phase_id, _, _, height = phase
            phase_filename = os.path.join(vial_folder, f"phase_{phase_name[int(phase_id)]}.csv")

            # Determine if we need to write header
            write_header = not os.path.exists(phase_filename)

            with open(phase_filename, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["capture_count", "height"])
                writer.writerow([capture_count, round(height,2)])

def start_timer():
    global timer_job
    try:
        value = float(interval_entry.get())
    except ValueError:
        tk.messagebox.showerror("Invalid interval", "Enter a number.")
        return

    unit = unit_var.get()
    multiplier = {"seconds":1000, "minutes":60000, "hours":3600000}[unit]
    interval_ms = int(value * multiplier)

    # Disable start button, enable stop
    start_timer_btn.config(state="disabled")
    stop_timer_btn.config(state="normal")

    # Schedule first capture immediately, then every interval
    capture_and_detect(count)
    timer_job = root.after(interval_ms, lambda: recurring_capture(interval_ms))

def recurring_capture(interval_ms):
    global timer_job
    capture_and_detect(count)
    timer_job = root.after(interval_ms, lambda: recurring_capture(interval_ms))

def stop_timer():
    global timer_job
    if timer_job:
        root.after_cancel(timer_job)
        timer_job = None
    start_timer_btn.config(state="normal")
    stop_timer_btn.config(state="disabled")


# Set up the main window
root = tk.Tk()
root.title("Camera Feed with Object Detection")

source_button = ttk.Button(root, text="Choose Input Source", command=choose_input)
source_button.pack(side="top", fill="x", padx=10, pady=10)



# Create a panel for the live video feed
video_panel = Label(root)
video_panel.pack(side="left", padx=10, pady=10)

# Create a panel for the detection result
detected_panel = Label(root)
detected_panel.pack(side="right", padx=10, pady=10)

set_up = ttk.Button(root, text="Set Up Experiment", command=lambda: (size_cm.clear(),capture_and_detect()))
set_up.pack(side="bottom", fill="both", padx=10, pady=10)

# Button to manually capture an image and run object detection
capture_button = ttk.Button(root, text="Capture & Detect", command=lambda: capture_and_detect(count),state="disabled")
capture_button.pack(side="bottom", fill="both", padx=10, pady=10)


# Timer setup widgets
tk.Label(root, text="Interval:").pack(side="top", pady=(10,0))
interval_entry = tk.Entry(root)
interval_entry.insert(0, "30")    # default = 30
interval_entry.pack(side="top", padx=10)

unit_var = tk.StringVar(value="seconds")
unit_menu = ttk.OptionMenu(root, unit_var, "seconds", "seconds", "minutes", "hours")
unit_menu.pack(side="top", pady=5)

start_timer_btn = ttk.Button(root, text="Start Timer", command=lambda: start_timer(),state="disabled")
start_timer_btn.pack(side="top", pady=5)

stop_timer_btn = ttk.Button(root, text="Stop Timer", command=lambda: stop_timer(), state="disabled")
stop_timer_btn.pack(side="top", pady=5)


plot_button = ttk.Button(root, text="Open Phase Plots", command=open_plot_window)
plot_button.pack(side="top", padx=10, pady=5)

entry_container = tk.Frame(root)
entry_container.pack(side="bottom", fill="x", padx=10, pady=10)

update_feed()
root.mainloop()
cap.release()

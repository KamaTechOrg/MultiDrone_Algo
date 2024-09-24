import grpc
import os
import sys
server_path = os.path.abspath('../server')
sys.path.append(server_path)
import stereo_vehicle_pb2 as pb2
import stereo_vehicle_pb2_grpc as pb2_grpc
import cv2
import time
from tkinter import Tk, Label, Button, Entry, StringVar, Text, Scrollbar, filedialog, messagebox, simpledialog, Toplevel
# פונקציה לעזור לקרוא תמונה ולהמיר אותה לבייטים
def read_image_as_bytes(image_path):
    with open(image_path, 'rb') as img_file:
        return img_file.read()
# פונקציה לעזור לקרוא סרטון ולהמיר אותו לבייטים
def read_video_as_bytes(video_path):
    with open(video_path, 'rb') as video_file:
        return video_file.read()
# פונקציה לבחירת תמונה
def select_image():
    return filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
# פונקציה לבחירת סרטון
def select_video():
    return filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
# פונקציה למציאת גובה (שדות הנדרשים)
def test_find_height(stub, output_text):
    dialog = Toplevel()
    dialog.title("Find Height")
    Label(dialog, text="Select Left Image:").pack(pady=5)
    left_image_var = StringVar()
    left_image_entry = Entry(dialog, textvariable=left_image_var, width=50)
    left_image_entry.pack(pady=5)
    Button(dialog, text="Browse", command=lambda: left_image_var.set(select_image())).pack(pady=5)
    Label(dialog, text="Select Right Image:").pack(pady=5)
    right_image_var = StringVar()
    right_image_entry = Entry(dialog, textvariable=right_image_var, width=50)
    right_image_entry.pack(pady=5)
    Button(dialog, text="Browse", command=lambda: right_image_var.set(select_image())).pack(pady=5)
    Label(dialog, text="Enter X coordinate:").pack(pady=5)
    x_var = StringVar()
    Entry(dialog, textvariable=x_var).pack(pady=5)
    Label(dialog, text="Enter Y coordinate:").pack(pady=5)
    y_var = StringVar()
    Entry(dialog, textvariable=y_var).pack(pady=5)
    def submit():
        left_image_path = left_image_var.get()
        right_image_path = right_image_var.get()
        if not left_image_path or not right_image_path:
            messagebox.showerror("Error", "Both images must be selected!")
            return
        left_image_bytes = read_image_as_bytes(left_image_path)
        right_image_bytes = read_image_as_bytes(right_image_path)
        height_request = pb2.HeightRequest(
            left_image=left_image_bytes,
            right_image=right_image_bytes,
            x=int(x_var.get()),
            y=int(y_var.get()),
            focal_length=1.2,
            baseline=0.5,
            camera_height=1.5
        )
        response = stub.FindHeight(height_request)
        output_text.insert('end', f"FindHeight response: {response.number}\n")
        dialog.destroy()
    Button(dialog, text="Submit", command=submit).pack(pady=10)
# פונקציה לזיהוי רכב (שדות הנדרשים)
def test_identify_vehicle(stub, output_text):
    dialog = Toplevel()
    dialog.title("Identify Vehicle")
    Label(dialog, text="Select Reference Image:").pack(pady=5)
    reference_image_var = StringVar()
    reference_image_entry = Entry(dialog, textvariable=reference_image_var, width=50)
    reference_image_entry.pack(pady=5)
    Button(dialog, text="Browse", command=lambda: reference_image_var.set(select_image())).pack(pady=5)
    Label(dialog, text="Select Video:").pack(pady=5)
    video_var = StringVar()
    video_entry = Entry(dialog, textvariable=video_var, width=50)
    video_entry.pack(pady=5)
    Button(dialog, text="Browse", command=lambda: video_var.set(select_video())).pack(pady=5)
    def submit():
        reference_image_path = reference_image_var.get()
        video_path = video_var.get()
        if not reference_image_path or not video_path:
            messagebox.showerror("Error", "Both reference image and video must be selected!")
            return
        reference_image_bytes = read_image_as_bytes(reference_image_path)
        video_bytes = read_video_as_bytes(video_path)
        vehicle_request = pb2.IdentifyVehicleRequest(
            image=reference_image_bytes,
            video=video_bytes
        )
        response = stub.IdentifyVehicle(vehicle_request)
        output_text.insert('end', f"IdentifyVehicle response: {response.results}\n")
        dialog.destroy()
    Button(dialog, text="Submit", command=submit).pack(pady=10)
# פונקציה לשליטה על חוט (שדות הנדרשים)
def test_control_thread(stub, output_text):
    dialog = Toplevel()
    dialog.title("video stabilization")
    Label(dialog, text="Select Video:").pack(pady=5)
    video_var = StringVar()
    video_entry = Entry(dialog, textvariable=video_var, width=50)
    video_entry.pack(pady=5)
    Button(dialog, text="Browse", command=lambda: video_var.set(select_video())).pack(pady=5)
    def submit():
        video_path = video_var.get()
        if not video_path:
            messagebox.showerror("Error", "Video must be selected!")
            return
        video_bytes = read_video_as_bytes(video_path)
        response = stub.ControlThread(pb2.ControlRequest(activate=True, video=video_bytes))
        output_text.insert('end', f"ControlThread activate response: {response.message}\n")
        time.sleep(30)
        response = stub.ControlThread(pb2.ControlRequest(activate=False))
        output_text.insert('end', f"video stabilized response: {response.message}\n")
        dialog.destroy()
    Button(dialog, text="Submit", command=submit).pack(pady=10)
# פונקציית main
def main():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.ImageProcessingStub(channel)
    # יצירת ממשק משתמש
    root = Tk()
    root.title("Vehicle Detection and Height Measurement")
    # כותרת
    label = Label(root, text="Select an action to perform:", font=("Helvetica", 14))
    label.pack(pady=10)
    # כפתור למציאת גובה
    find_height_button = Button(root, text="Find Height", command=lambda: test_find_height(stub, output_text))
    find_height_button.pack(pady=5)
    # כפתור לזיהוי רכב
    identify_vehicle_button = Button(root, text="Identify Vehicle", command=lambda: test_identify_vehicle(stub, output_text))
    identify_vehicle_button.pack(pady=5)
    # כפתור לייצוכ וידיו
    control_thread_button = Button(root, text="video stabilzation", command=lambda: test_control_thread(stub, output_text))
    control_thread_button.pack(pady=5)
        # תיבת טקסט לתוצאות
    output_text = Text(root, height=15, width=50)
    output_text.pack(pady=10)
    # גלילה לתיבת התוצאות
    scroll_bar = Scrollbar(root, command=output_text.yview)
    scroll_bar.pack(side='right', fill='y')
    output_text.config(yscrollcommand=scroll_bar.set)
    root.mainloop()
if __name__ == '__main__':
    main()
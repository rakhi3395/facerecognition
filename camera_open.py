import tkinter as tk
import mysql.connector
from tkinter import messagebox
import cv2
from datetime import datetime
import uuid
from config import DB_PASSWORD
# from mtcnn import MTCNN
# import face_recognition
# import numpy as np
# import json
# from scipy.spatial.distance import cosine
from deepface import DeepFace
import distance as dst
# import warnings
# warnings.filterwarnings("ignore")
import os
import time
import pandas as pd
import numpy as np
class CameraApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Camera App")
        self.master.geometry("800x500+300+100")
        self.create_ui()
        # self.faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
        # Load VGGFace model for face recognition
        # self.model_path = "models/embeddings.npy"
        # self.img_size = (224,224)
        self.models = [
                            "VGG-Face", 
                            "Facenet", 
                            "Facenet512", 
                            "OpenFace", 
                            "DeepFace", 
                            "DeepID", 
                            "ArcFace", 
                            "Dlib", 
                            "SFace",
                            ]
        self.metrics = ["cosine", "euclidean", "euclidean_l2"]
        self.model_name = self.models[2]
        self.metrics_name = self.metrics[2]
        self.distance_metric = self.metrics[2]
        self.model = DeepFace.build_model(self.model_name)

        representations_file = 'data/representations_facenet512.pkl'
        self.data_dir = "data"

        # Check if the file exists and then delete it
        if os.path.exists(representations_file):
            os.remove(representations_file)
            print(f"Deleted {representations_file}")
        self.conn = mysql.connector.connect(host="localhost", username="root", password=DB_PASSWORD, database="face_recognition")
        self.my_cursor = self.conn.cursor()
        self.my_cursor.execute(f"SELECT emp_id, email_id, name, dep FROM employee")
        self.all_data = self.my_cursor.fetchall()
        print(self.all_data)
        self.detector_backend = "opencv"
        self.target_size = (160, 160)
        self.normalization = "base"
        self.representations = []
        self.load_representations()


    def create_ui(self):
        # Frame for the form
        form_frame = tk.Frame(self.master, bg="skyblue")
        form_frame.place(x=0, y=0, width=1530, height=790)

        register_label = tk.Label(form_frame, text="CAMERA FORM", font=("times new roman", 20, "bold"), fg="darkblue", bg="skyblue")
        register_label.place(x=30, y=20)

        # Labels and entry fields for camera name and IP with larger font size
        tk.Label(form_frame, text="Camera Name:", font=("Arial", 14), bg="skyblue").place(x=50, y=80)
        self.camera_name_entry = tk.Entry(form_frame, font=("Arial", 13))
        self.camera_name_entry.place(x=200, y=80)

        tk.Label(form_frame, text="Camera IP:", font=("Arial", 14), bg="skyblue").place(x=50, y=120)
        self.camera_ip_entry = tk.Entry(form_frame, font=("Arial", 13))
        self.camera_ip_entry.place(x=200, y=120)

        # Blue-colored Save button with larger font size
        save_button = tk.Button(form_frame, text="Save", command=self.save_camera, bg="blue", font=("Arial", 12))
        save_button.place(x=260, y=150)

        # Green-colored Preview button with larger font size
        open_camera_button = tk.Button(form_frame, text="Open Camera", command=self.open_camera, bg="green", font=("Arial", 12))
        open_camera_button.place(x=50, y=240)

        # Listbox to display saved cameras
        self.camera_listbox = tk.Listbox(form_frame, font=("Arial", 10))
        self.camera_listbox.place(x=50, y=274, width=330, height=150)


        preview_button = tk.Button(self.master, text="Preview", command=self.preview_camera, bg="yellow", font=("Arial", 12))
        preview_button.pack()
        preview_button.place(x=180, y=240)

        # Dictionary to store camera names as keys and IP addresses as values
        self.camera_dict = {}

        # Load initial camera data
        self.load_camera_data()

            # attendance
    def mark_attendance(self,emp_id,email_id,name,dep):
        current_date = datetime.now().date()
        # print("current_date:",current_date)
        # conn = mysql.connector.connect(host="localhost", username="root", password=DB_PASSWORD, database="face_recognition")
        # my_cursor = conn.cursor()
        self.my_cursor.execute(f"SELECT attendance_id FROM face_recognition.attendance WHERE emp_id = {emp_id} AND date='{current_date}'")
        fetched_data = self.my_cursor.fetchone()
        if fetched_data:
            # handle checkout case
            attendance_id = fetched_data[0]
            checkout_time = datetime.now()
            self.my_cursor.execute(f"UPDATE attendance SET checkout_time = '{checkout_time}' WHERE attendance_id = '{attendance_id}'")
            self.conn.commit()

        else:
            # handle checkin date
            attendance_id = str(uuid.uuid4())
            checkin_time = datetime.now()
            insert_query = "INSERT INTO attendance (attendance_id, emp_id, date, email_id, name, dep, checkin_time) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            data = (attendance_id, emp_id, current_date, email_id, name, dep, checkin_time)
            self.my_cursor.execute(insert_query, data)
            self.conn.commit()


    def save_camera(self):
        # Get camera name and IP from the entry fields
        camera_name = self.camera_name_entry.get().strip()
        camera_ip = self.camera_ip_entry.get().strip()
        if camera_ip=="localhost" or camera_ip == "127.0.0.1":
            camera_ip = 0
        if camera_name and (camera_ip or camera_ip==0):  # Check if fields are not empty
            # Attempt to open the camera stream to check if it's working
            cap = cv2.VideoCapture(camera_ip)
            if cap.isOpened():
                # If the camera is working, save its details in the database
                self.save_to_database(camera_name, camera_ip)
                cap.release()  # Release the camera

                # Clear entry fields and reload camera data
                self.camera_name_entry.delete(0, tk.END)
                self.camera_ip_entry.delete(0, tk.END)
                self.load_camera_data()
            else:
                cap.release()  # Release the camera
                messagebox.showerror("Error", "Unable to open the camera. Please check the camera IP.")
        else:
            messagebox.showerror("Error", "Camera Name and IP cannot be empty!")
    def create_representations(self, img):
        img = cv2.resize(img, self.target_size)
        img = np.expand_dims(img, axis=0)
        embedding = self.model(img, training=False).numpy()[0].tolist()
        return embedding
    def load_representations(self):
        print(os.listdir(self.data_dir))
        for employee_folder in os.listdir(self.data_dir):
            print("employee folder:",employee_folder)
            # Load employee photos from the provided folder
            for filename in os.listdir(os.path.join(self.data_dir,employee_folder)):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") :
                    img_path = os.path.join(os.path.join(self.data_dir,employee_folder), filename)

                    img_objs = DeepFace.extract_faces(
                            img_path=img_path,
                            target_size=self.target_size,
                            detector_backend=self.detector_backend,
                            grayscale=False,
                            enforce_detection=False,
                            align=True,
                        )
                    for img_obj in img_objs:
                        img = img_obj['face']
                        
                        img_representation = self.create_representations(img)

                        instance = []
                        instance.append(employee_folder)
                        instance.append(img_representation)
                        self.representations.append(instance)
        
        self.df = pd.DataFrame(self.representations, columns=["identity", f"{self.model_name}_representation"])
        print("loaded img representation:", self.df)


    def load_camera_data(self):
        # Retrieve camera names and IPs from the database and populate the listbox and dictionary
        conn = mysql.connector.connect(host="localhost", username="root", password=DB_PASSWORD, database="face_recognition")
        cursor = conn.cursor()
        select_query = "SELECT camera_name, camera_ip FROM cameras"
        cursor.execute(select_query)
        cameras = cursor.fetchall()
        self.camera_listbox.delete(0, tk.END)
        self.camera_dict.clear()
        for camera in cameras:
            camera_name, camera_ip = camera
            self.camera_listbox.insert(tk.END, camera_name)
            self.camera_dict[camera_name] = camera_ip
        conn.close()


    def get_matched_data(self, emp_id):
        for emp_data in self.all_data:
            if emp_data[0] == int(emp_id):
                return emp_data
        return None
    def filter_roi_faces(self, target_objs, roi_x, roi_y, roi_width, roi_height):
        filtered_faces = []
        for target_obj in target_objs:
            face_area = target_obj['facial_area']
            x,y,w,h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            if roi_x <= x and x + w <= roi_x + roi_width and roi_y <= y and y + h <= roi_y + roi_height:
                filtered_faces.append(target_obj)
        return filtered_faces


    def open_camera(self):
        # DeepFace.stream(db_path = "data")
        # Get the selected camera name from the listbox
        selected_camera_name = self.camera_listbox.get(self.camera_listbox.curselection())

        # Get the corresponding IP address from the dictionary
        camera_ip = self.camera_dict.get(selected_camera_name)
        print("face_recognition started")
        # Open the selected camera stream
        # check if local camera
        if camera_ip=="0":
            camera_ip=0
        camera_ip = "test_video.mp4"
        # camera_ip = 0
        cap = cv2.VideoCapture(camera_ip)
        # cap.set(cv2.CAP_PROP_FPS, 50)
        is_mark_attendance = False
        previous_id = -1
        # Define the ROI (Region of Interest) boundaries
        roi_x, roi_y, roi_width, roi_height = 500, 92, 600, 550
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print("Actual FPS of the video:", actual_fps)
        desired_fps = actual_fps
        frame_count = 0
        while True:
            start_time = time.time()
            ret, img = cap.read()
            roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

            # Resize the ROI
            # resized_roi = cv2.resize(roi, (new_width, new_height))
            rgb_frame = roi[:, :, ::-1]
            # features = self.faceCascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10)
            # print(rgb_frame)
            cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 1)  # Draw green rectangle on the face
            if frame_count%3==0:
                # detected_faces = DeepFace.find(
                #     img_path=rgb_frame,
                #     db_path="data",
                #     model_name=self.models[2],
                #     distance_metric=self.metrics[2],
                #     enforce_detection=False,
                #     silent=True
                # )
                
                target_objs = DeepFace.extract_faces(img_path=rgb_frame,
                        target_size = self.target_size, 
                        detector_backend = self.detector_backend,
                        enforce_detection=False,
                )
                # print(target_objs)
                # target_objs = self.filter_roi_faces(target_objs, roi_x, roi_y, roi_width, roi_height)
                for target_obj in target_objs:
                    # Check if the face coordinates are within the ROI
                    target_region = target_obj['facial_area']
                    target_confidence = target_obj['confidence']

                    # print("face_confidence:",target_confidence)

                    if target_confidence<10:
                        continue
                    target_representation = self.create_representations(target_obj['face'])

                    result_df = self.df.copy()  # df will be filtered in each img
                    result_df["source_x"] = target_region["x"]
                    result_df["source_y"] = target_region["y"]
                    result_df["source_w"] = target_region["w"]
                    result_df["source_h"] = target_region["h"]

                    distances = []
                    for index, instance in self.df.iterrows():
                        source_representation = instance[f"{self.model_name}_representation"]

                        if self.distance_metric == "cosine":
                            distance = dst.findCosineDistance(source_representation, target_representation)
                        elif self.distance_metric == "euclidean":
                            distance = dst.findEuclideanDistance(source_representation, target_representation)
                        elif self.distance_metric == "euclidean_l2":
                            distance = dst.findEuclideanDistance(
                                dst.l2_normalize(source_representation),
                                dst.l2_normalize(target_representation),
                            )
                        else:
                            raise ValueError(f"invalid distance metric passes - {distance_metric}")

                        distances.append(distance)

                        # ---------------------------

                    result_df[f"{self.model_name}_{self.distance_metric}"] = distances

                    threshold = dst.findThreshold(self.model_name, self.distance_metric)
                    result_df = result_df.drop(columns=[f"{self.model_name}_representation"])
                    result_df = result_df[result_df[f"{self.model_name}_{self.distance_metric}"] <= threshold]
                    result_df = result_df.sort_values(
                        by=[f"{self.model_name}_{self.distance_metric}"], ascending=True
                    ).reset_index(drop=True)

                    print("result_df:", result_df)
                    if len(result_df):
                        face = result_df
                        x, y, w, h = face['source_x'][0],face['source_y'][0],face['source_w'][0],face['source_h'][0]
                        confidence_score = face['Facenet512_euclidean_l2'][0]
                        predicted_class = face['identity'][0]
                        
                        # if confidence > 0.5:  # Adjust the threshold as needed
                        # if roi_x <= x and x + w <= roi_x + roi_width and roi_y <= y and y + h <= roi_y + roi_height:
                            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Draw green rectangle on the face

                            # print("predicted_class:",predicted_class)
                            # print("confidence_score:",confidence_score)
                            # Compare the detected face embeddings with known face embeddings
                        if confidence_score > 0.9 : 
                            emp_id = predicted_class                  
                            matched_data = self.get_matched_data(emp_id)
                            print("matched_data:", matched_data)
                            if matched_data:
                                if previous_id!=matched_data[0]:
                                    is_mark_attendance = False
                                    previous_id = matched_data[0]
                                    # print("matched_data:", matched_data[0])
                            
                            if matched_data is not None:
                                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3) 
                                # print("is_mark_attendance:",is_mark_attendance)
                                emp_id, email_id, name, dep = matched_data[0], matched_data[1], matched_data[2], matched_data[3]
                                cv2.putText(roi, f"emp_id:{emp_id}", (x, y-75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                                # cv2.putText(img, f"email_id:{email_id}", (x, y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                                cv2.putText(roi, f"name:{name}", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                                # cv2.putText(img, f"department:{dep}", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                                if not is_mark_attendance:
                                    self.mark_attendance(emp_id, email_id, name, dep)
                                    is_mark_attendance = True
                        # else:
                        #     # print("predicted_class:",predicted_class)
                        #     # print("confidence_score:",confidence_score)
                        #     cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        #     cv2.putText(roi, "Unknown Face", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        #     is_mark_attendance = False
                    # else:
                    #     # print("predicted_class:",predicted_class)
                        
                    #     face_area = target_obj['facial_area']
                        
                    #     if target_obj["confidence"]>10:
                    #         print("face_area:",face_area)
                    #         print("face_confidence:",target_obj["confidence"])
                    #         # print("unknown face detected")
                    #         x,y,w,h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                    #         cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    #         cv2.putText(roi, "Unknown Face", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    #         is_mark_attendance = False

                        
                # # Draw a red rectangle around the ROI
                # # cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 3)
                # Display the region of interest (ROI)
                # cv2.imshow("ROI", roi)
                cv2.imshow("Welcome To Face Recognition", img)
                processing_time = time.time() - start_time
                print("Processing Time:", processing_time)

                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
            frame_count = frame_count +1
        
        cap.release()
        cv2.destroyAllWindows()

    def preview_camera(self):
        # Get the selected camera name from the listbox
        selected_camera_name = self.camera_listbox.get(self.camera_listbox.curselection())
        # Get the corresponding IP address from the dictionary
        camera_ip = self.camera_dict.get(selected_camera_name)
        width, height = 600, 500
        # Open the selected camera stream in a preview window
        preview_window = tk.Toplevel(self.master)
        preview_window.title(f"Preview: {selected_camera_name}")
        preview_window.geometry(f"{width}x{height}")
        print("camera ip:", camera_ip)
        # check if local camera
        if camera_ip=="0":
            camera_ip=0
        print("camera ip:",camera_ip)
        cap = cv2.VideoCapture(camera_ip)
        while True:
            ret, frame = cap.read()
            cv2.imshow(f"Preview: {selected_camera_name}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def save_to_database(self, camera_name, camera_ip):
        # Establish a MySQL connection and save camera details in the database
        conn = mysql.connector.connect(host="localhost", username="root", password=DB_PASSWORD, database="face_recognition")
        cursor = conn.cursor()
        insert_query = "INSERT INTO cameras (camera_name, camera_ip) VALUES (%s, %s)"
        data = (camera_name, camera_ip)
        cursor.execute(insert_query, data)
        conn.commit()
        conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
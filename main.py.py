import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, redirect, url_for, request, send_file

app = Flask(__name__)

# Path to the directory containing student images
path = 'C:/Users/nabee/Desktop/cvproject/face_recognition_attendance/student_images'
images = []
classNames = []
mylist = os.listdir(path)
cap = None  # Declare cap as a global variable

# Load student images and class names
for cl in mylist:
    try:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        else:
            print(f"Error loading image: {cl}")
    except Exception as e:
        print(f"Error processing image: {cl}. Error: {str(e)}")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)
        if len(encoded_face) > 0:
            encodeList.append(encoded_face[0])
        else:
            print("No face found in the image.")
    return encodeList

encoded_face_train = findEncodings(images)

# Set to store unique attendance names
attendance_names_set = set()

def markAttendance(name, file_path):
    # Check if the file exists, and if not, create it with a header
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time,Date\n')

    if name not in attendance_names_set:
        with open(file_path, 'a') as f:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            entry = f'{name},{time},{date}\n'
            f.write(entry)
            attendance_names_set.add(name)  # Add the name to the set

        # Convert the data to a DataFrame
        df = pd.read_csv(file_path)

        # Save the DataFrame as an Excel file using the correct file path
        excel_file_path = file_path.replace('.csv', '.xlsx')
        df.to_excel(excel_file_path, index=False, engine='openpyxl')  # Use openpyxl engine

        # Update the global attendance_file_path variable
        global attendance_file_path
        attendance_file_path = excel_file_path

        print(f"Attendance marked for {name}. Excel file path: {attendance_file_path}")


# Define a route for the root URL '/'
@app.route('/')
def home():
    # You can pass data to your template like this
    mess = "Welcome to the Attendance System"
    return render_template('index.html', mess=mess)


# Define a route for starting attendance
@app.route('/start')
def start_attendance():
    global cap  # Use the global cap variable
    if cap is None:
        cap = cv2.VideoCapture(0)
        # Start capturing frames and processing attendance
        return redirect(url_for('process_attendance'))
    return redirect(url_for('home'))


# Define a route for processing attendance
@app.route('/process-attendance')
def process_attendance():
    global cap
    if cap is not None:
        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            faces_in_frame = face_recognition.face_locations(imgS)
            encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
            for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
                faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
                min_face_dist = np.min(faceDist)
                confidence = 1 - min_face_dist  # Calculate confidence score

                # You can set a threshold for confidence (e.g., 0.5) and only mark attendance if confidence is above the threshold
                if min_face_dist < 0.5:
                    matchIndex = np.argmin(faceDist)
                    name = classNames[matchIndex].upper().lower()
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{name} ({confidence:.2f})', (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name, attendance_file_path)

            cv2.imshow('webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save the attendance data to the Excel file and close the file
        print(f"Attendance saved to {attendance_file_path}")
        cap.release()
        cv2.destroyAllWindows()

        # Set a message to display on the HTML page
        mess = "Attendance completed!"
        return render_template('index.html', mess=mess)
    else:
        return redirect(url_for('home'))


# Define a route for stopping and saving attendance
@app.route('/stop-and-save')
def stop_and_save_attendance():
    global cap  # Use the global cap variable
    if cap is not None:
        # Release the webcam if it's initialized
        cap.release()
        cv2.destroyAllWindows()
        cap = None  # Reset the cap variable

        # Set a message to display on the HTML page
        mess = "Attendance stopped and saved!"
    else:
        mess = "Attendance was not running."

    return render_template('index.html', mess=mess)


# Define a route for downloading attendance
@app.route('/download-attendance')
def download_attendance():
    # Ensure that the attendance file exists
    if os.path.isfile(attendance_file_path):
        return send_file(attendance_file_path, as_attachment=True)
    else:
        return "Attendance file not found."


if __name__ == '__main__':
    # Create a new CSV file with a unique name based on the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    attendance_file_path = f'Attendance_{timestamp}.csv'

    app.run(debug=True)

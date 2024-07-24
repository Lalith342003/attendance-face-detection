from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from datetime import date
import cv2, os, numpy

ui,_=loadUiType('face-reco.ui')

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)
        self.LOGINBUTTON.clicked.connect(self.login)
        self.LOGOUTBUTTON.clicked.connect(self.logout)
        self.CLOSE.clicked.connect(self.close_window)
        self.TRAINLINK1.clicked.connect(self.show_training_form)
        self.ATTLINK1.clicked.connect(self.show_attendance_entry_form)
        self.REPORTSLINK1.clicked.connect(self.show_reports_form)
        self.TRAININGBACK.clicked.connect(self.show_mainform)
        self.ATTENDANCEBACK.clicked.connect(self.show_mainform)
        self.REPORTSBACK.clicked.connect(self.show_mainform)
        self.TRAININGBUTTON.clicked.connect(self.start_training)
        self.RECORD.clicked.connect(self.record_attendance)
        self.dateEdit.setDate(date.today())
        self.dateEdit.dateChanged.connect(self.show_selected_date_reports)
        self.tabWidget.setStyleSheet("QTabWidget::pane{border:0;}")
        try:
            con = sqlite3.connect("face-reco.db")
            con.execute("CREATE TABLE IF NOT EXISTS attendance(attendanceid INTEGER, name TEXT, attendancedate TEXT)")
            con.commit()
            print("Table created successfully")
        except:
            print("Error in database")

    ### LOGIN PROCESS ###
    def login(self):
        pw = self.PASSWORD.text()
        if(pw=="123"):
            self.PASSWORD.setText("")
            self.LOGININFO.setText("")
            self.tabWidget.setCurrentIndex(1)
        else:
            self.LOGININFO.setText("Invalid Password..")
            self.PASSWORD.setText("")

    ### LOG OUT PROCESS ###
    def logout(self):
        self.tabWidget.setCurrentIndex(0)

    ### CLOSE WINDOW PROCESS ###
    def close_window(self):
        self.close()

    ### SHOW MAIN FORM ###
    def show_mainform(self):
        self.tabWidget.setCurrentIndex(1)        

    ### SHOW TRAINING FORM ###
    def show_training_form(self):
        self.tabWidget.setCurrentIndex(2)

    ### SHOW ATTENDANCE ENTRY FORM ###
    def show_attendance_entry_form(self):
        self.tabWidget.setCurrentIndex(3)

    ### SHOW REPORTS FORM ###
    def show_reports_form(self):
        self.tabWidget.setCurrentIndex(4)
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance")
        result = cursor.fetchall()
        r=0
        c=0
        for row_number,row_data in enumerate(result):
            r+=1
            c=0
            for column_number,data in enumerate(row_data):
                c+=1
        self.REPORTS.setColumnCount(c)

        for row_number,row_data in enumerate(result):
            self.REPORTS.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTS.setItem(row_number,column_number,QTableWidgetItem(str(data)))

        self.REPORTS.setHorizontalHeaderLabels(['Id','Name','Date'])        
        self.REPORTS.setColumnWidth(0,10)
        self.REPORTS.setColumnWidth(1,60)
        self.REPORTS.setColumnWidth(2,70)
        self.REPORTS.verticalHeader().setVisible(False)
        

    ### SHOW SELECTED DATE REPORTS ###
    def show_selected_date_reports(self):
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance WHERE attendancedate = '"+ str((self.dateEdit.date()).toPyDate()) +"'")
        result = cursor.fetchall()
        r=0
        c=0
        for row_number,row_data in enumerate(result):
            r+=1
            c=0
            for column_number,data in enumerate(row_data):
                c+=1
        self.REPORTS.setColumnCount(c)

        for row_number,row_data in enumerate(result):
            self.REPORTS.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTS.setItem(row_number,column_number,QTableWidgetItem(str(data)))

        self.REPORTS.setHorizontalHeaderLabels(['Id','Name','Date'])        
        self.REPORTS.setColumnWidth(0,10)
        self.REPORTS.setColumnWidth(1,60)
        self.REPORTS.setColumnWidth(2,70)
        self.REPORTS.verticalHeader().setVisible(False)


    ### TRAINING PROCESS ###
    def start_training(self):
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'
        sub_data = self.traineeName.text()
        path = os.path.join(datasets,sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
            print("The new directory is created")
            (width,height) = (130,100)
            face_cascade = cv2.CascadeClassifier(haar_file)
            webcam = cv2.VideoCapture(0)
            count = 1
            while count < int(self.trainingCount.text()) + 1:
                print(count)
                (_,im) = webcam.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,4)
                for (x,y,w,h) in faces:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                    face = gray[y:y+h,x:x+w]
                    face_resize = cv2.resize(face,(width,height))
                    cv2.imwrite('%s/%s.png'%(path,count),face_resize)
                count += 1
                cv2.imshow('OpenCV',im)
                key = cv2.waitKey(10)
                if key == 27:
                    break
            webcam.release()
            cv2.destroyAllWindows()  
            path=""
            QMessageBox.information(self,"Attendance System","Training Completed Successfully")  
            self.traineeName.setText("")
            self.trainingCount.setText("100")

    ### RECORD ATTENDANCE ###
    def record_attendance(self):
        self.currentprocess.setText("Process started.. Waiting..")        
        haar_file = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)
        datasets = 'datasets'
        (images, labels, names, id) = ([], [], {}, 0)

        # Collect images and labels for training
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = os.path.join(subjectpath, filename)
                    label = id
                    images.append(cv2.imread(path, 0))
                    labels.append(int(label))
                id += 1

        # Convert images and labels to numpy arrays
        (images, labels) = [numpy.array(lis) for lis in [images, labels]]

        # Train the face recognition model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images, labels)

        webcam = cv2.VideoCapture(0)
        cnt = 0
        attendance_recorded = False  # Flag to track if attendance is recorded

        while not attendance_recorded:
            ret, frame = webcam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                face = gray[y:y+h, x:x+w]
                face_resize = cv2.resize(face, (130, 100))
                prediction = recognizer.predict(face_resize)
                # print("Prediction:", prediction)

                if prediction[1] < 72:
                    name = names[prediction[0]]
                    print("Name:", name)
                    cv2.putText(frame, '{}'.format(name), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Record attendance
                    attendance_id = 0
                    try:
                        connection = sqlite3.connect("face-reco.db")
                        cursor = connection.execute("SELECT MAX(attendanceid) FROM attendance")
                        result = cursor.fetchone()
                        if result[0]:
                            attendance_id = int(result[0]) + 1
                    except Exception as e:
                        print("Error in database:", e)
                        attendance_id = 1

                    try:
                        con = sqlite3.connect("face-reco.db")
                        cursor = con.execute("SELECT * FROM attendance WHERE name=? AND attendancedate=?", (name, str(date.today())))
                        result = cursor.fetchall()
                        if not result:
                            con.execute("INSERT INTO attendance VALUES (?, ?, ?)", (attendance_id, name, str(date.today())))
                            con.commit()
                            attendance_recorded = True  # Set flag to True
                    except Exception as e:
                        print("Error in database insert:", e)

                    print("Attendance Registered successfully")
                    self.currentprocess.setText("Attendance entered for " + name)
                    cnt = 0
                    webcam.release()
                    cv2.destroyAllWindows()
                    return
                else:
                    cnt += 1
                    if cnt > 100:
                        print("Unknown person")
                        self.currentprocess.setText("Unknown Person ")
                        cv2.imwrite('unKnown.jpg', frame)
                        cnt = 0
                        webcam.release()
                        cv2.destroyAllWindows()
                        return

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()




















def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()    
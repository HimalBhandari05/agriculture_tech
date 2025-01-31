import sys
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                            QVBoxLayout, QPushButton, QLineEdit, QLabel, QSpacerItem, 
                            QSizePolicy, QMessageBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import os
import main_real  # Import main_real instead of calculator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.main_real_window = None  # Renamed for clarity

    def initUI(self):
        self.setWindowTitle("AI Sign-In")
        #self.setGeometry(100, 100, 500, 500)
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen.x(), screen.y(), screen.width(), screen.height())

        # Set a colorful gradient background
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #FF6F61, stop:0.5 #6B5B95, stop:1 #88B04B
                );
            }
        """)

        # Main layout
        main_layout = QVBoxLayout()

        # Add spacer at the top to push content to center
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # "Please, Sign In" label
        sign_in_label = QLabel("Please, Register")
        sign_in_label.setFont(QFont("Arial", 20, QFont.Bold))
        sign_in_label.setStyleSheet("color: white; padding-bottom: 10px;")
        sign_in_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(sign_in_label)

        # "Button" background widget (Sign-in panel)
        self.center_button_widget = QWidget(self)
        self.center_button_widget.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.90);
            border-radius: 25px;
            padding: 20px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);
        """)
        self.center_button_widget.setFixedSize(400, 280)  

        # Layout inside the sign-in panel
        button_layout = QVBoxLayout(self.center_button_widget)

        # Username row
        username_layout = QHBoxLayout()
        username_label = QLabel("Username:")
        username_label.setFont(QFont("Arial", 12))
        self.username_input = QLineEdit()
        self.username_input.setStyleSheet("""
            background-color: white;
            color: black;
            padding: 8px;
            border-radius: 15px;
            border: none;
            box-shadow: inset 2px 2px 5px rgba(0,0,0,0.2);
        """)
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        button_layout.addLayout(username_layout)

        # Password row
        password_layout = QHBoxLayout()
        password_label = QLabel("Password:")
        password_label.setFont(QFont("Arial", 12))
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("""
            background-color: white;
            color: black;
            padding: 8px;
            border-radius: 15px;
            border: none;
            box-shadow: inset 2px 2px 5px rgba(0,0,0,0.2);
        """)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        button_layout.addLayout(password_layout)

        # Submit button (Larger & with hover effect)
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setFixedSize(200, 50)
        self.submit_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #007BFF, stop:1 #00C6FF
                );
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                border-radius: 25px;
            }
            QPushButton:hover {
                background: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #0056b3, stop:1 #0099ff
                );
                box-shadow: 0px 0px 15px rgba(0, 123, 255, 0.6);
            }
        """)
        self.submit_button.clicked.connect(self.save_credentials)
        button_layout.addWidget(self.submit_button, alignment=Qt.AlignCenter)

        # Add the sign-in panel to the main layout
        main_layout.addWidget(self.center_button_widget, alignment=Qt.AlignHCenter)

        # Add spacer at the bottom to keep the panel centered
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def save_credentials(self):
        try:
            username = self.username_input.text()
            password = self.password_input.text()
            
            if username and password:
                base_dir = r"C:\Users\CSE\OneDrive\Desktop\HACKATHON real"
                file_path = os.path.join(base_dir, "signin.txt")
                
                os.makedirs(base_dir, exist_ok=True)
                
                with open(file_path, 'a') as f:
                    f.write(f"Username: {username}, Password: {password}\n")
                
                self.username_input.clear()
                self.password_input.clear()
                
                # Open main_real.py window
                try:
                    self.main_real_window = main_real.MainWindow()  # Changed to reference main_real
                    self.main_real_window.show()
                    self.hide()  # Hide the login window
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to open main_real: {str(e)}")
                    raise  # Re-raise the exception to see the full error trace
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
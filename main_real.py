import sys
from PyQt5.QtGui import QGuiApplication, QFontDatabase, QFont, QIcon, QImage, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QWidget, QFileDialog, QProgressBar, QMessageBox, QScrollArea, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import tensorflow as tf

# Disable oneDNN optimization warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Variables for the model and file paths
        self.model = None
        self.train_data_dir = None
        self.num_images = 0  # Store the number of images used for training
        self.current_image = None  # Store the current image for manipulation
        self.dark_mode = False  # Track dark mode state

    def initUI(self):
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen.x(), screen.y(), screen.width(), screen.height())
        self.setWindowTitle("AI Image Editor")

        # Load custom font
        QFontDatabase.addApplicationFont(r"D:\HACKATHON real\Roboto-Bold.ttf")  # Replace with your font path
        self.custom_font = QFont("Roboto", 12)

        # Set light background for the main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 2px solid #45a049;
            }
            QPushButton:pressed {
                background-color: #387038;
                border: 2px solid #387038;
            }
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: rgba(255, 255, 255, 0.8);
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
            QCheckBox {
                font-size: 14px;
                margin: 5px;
            }
        """)

        # Create main vertical layout
        main_layout = QVBoxLayout()

        # Create a horizontal layout for the top buttons
        top_buttons_layout = QHBoxLayout()

        # Create buttons for AI functionality
        self.upload_train = QPushButton("Upload Training Images", self)
        self.upload_predict = QPushButton("Upload Prediction Image", self)
        self.ask_ai = QPushButton("Ask AI", self)
        self.ask_ai.setIcon(QIcon(r"D:\HACKATHON real\ai_icon.png"))  # Replace with your icon path
        self.ask_ai.setFixedHeight(80)  # Increased height for the "Ask AI" button

        # Apply styling for buttons
        self.style_button(self.upload_train)
        self.style_button(self.upload_predict)
        self.style_button(self.ask_ai)

        # Add the top buttons to the horizontal layout
        top_buttons_layout.addWidget(self.upload_train)
        top_buttons_layout.addWidget(self.upload_predict)
        top_buttons_layout.addWidget(self.ask_ai)

        # Add the top buttons layout to the main layout
        main_layout.addLayout(top_buttons_layout)

        # Add a toggle button for dark mode (aligned to the left)
        toggle_layout = QHBoxLayout()
        self.dark_mode_toggle = QCheckBox("Toggle Dark Mode", self)
        self.dark_mode_toggle.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                margin: 5px;
            }
        """)
        self.dark_mode_toggle.stateChanged.connect(self.toggleDarkMode)
        toggle_layout.addWidget(self.dark_mode_toggle, alignment=Qt.AlignLeft)
        toggle_layout.addStretch()  # Push the toggle button to the left
        main_layout.addLayout(toggle_layout)

        # Create and center the image label
        self.image_label = QLabel("Your selected image will appear here.", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 10px;
                padding: 10px;
                background-color: rgba(255, 255, 255, 0.8);
            }
        """)
        self.image_label.setFont(self.custom_font)

        # Add the image label to a scroll area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        main_layout.addWidget(scroll_area, stretch=1)

        # Create a progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(False)  # Hide by default
        main_layout.addWidget(self.progress_bar)

        # Create a label for model predictions
        self.predictions_label = QLabel("Model's Predictions", self)
        self.predictions_label.setAlignment(Qt.AlignCenter)
        self.predictions_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                margin-top: 10px;
                color: #333333;
            }
        """)
        self.predictions_label.setFont(self.custom_font)
        main_layout.addWidget(self.predictions_label)

        # Add a label for image manipulation buttons
        manipulate_label = QLabel("Manipulate Image", self)
        manipulate_label.setAlignment(Qt.AlignCenter)
        manipulate_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                margin-top: 10px;
                color: #333333;
            }
        """)
        manipulate_label.setFont(self.custom_font)
        main_layout.addWidget(manipulate_label)

        # Create basic image manipulation buttons
        self.left = QPushButton("Rotate Left", self)
        self.right = QPushButton("Rotate Right", self)
        self.mirror = QPushButton("Mirror", self)
        self.clear = QPushButton("Clear Image", self)
        self.save = QPushButton("Save Image", self)

        # Apply styling for buttons
        self.style_button(self.left)
        self.style_button(self.right)
        self.style_button(self.mirror)
        self.style_button(self.clear)
        self.style_button(self.save)

        # Create a horizontal layout for the image manipulation buttons
        image_buttons_layout = QHBoxLayout()
        image_buttons_layout.addWidget(self.left)
        image_buttons_layout.addWidget(self.right)
        image_buttons_layout.addWidget(self.mirror)
        image_buttons_layout.addWidget(self.clear)
        image_buttons_layout.addWidget(self.save)

        # Add the image buttons layout to the main layout
        main_layout.addLayout(image_buttons_layout)

        # Add a help button
        self.help_button = QPushButton("Help", self)
        self.style_button(self.help_button)
        self.help_button.clicked.connect(self.showHelp)
        main_layout.addWidget(self.help_button)

        # Set the layout to the central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        # Connect buttons to backend functionality
        self.upload_train.clicked.connect(self.selectTrainFolder)
        self.upload_predict.clicked.connect(self.uploadTestImage)
        self.left.clicked.connect(self.rotateLeft)
        self.right.clicked.connect(self.rotateRight)
        self.mirror.clicked.connect(self.mirrorImage)
        self.clear.clicked.connect(self.clearImage)
        self.save.clicked.connect(self.saveImage)

    def style_button(self, button):
        button.setFont(self.custom_font)

    def toggleDarkMode(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #333333;
                }
                QLabel {
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #555555;
                    border: 2px solid #555555;
                    color: white;
                    font-size: 16px;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #666666;
                    border: 2px solid #666666;
                }
                QPushButton:pressed {
                    background-color: #444444;
                    border: 2px solid #444444;
                }
                QProgressBar {
                    border: 2px solid #777777;
                    border-radius: 5px;
                    text-align: center;
                    background-color: rgba(255, 255, 255, 0.2);
                }
                QProgressBar::chunk {
                    background-color: #555555;
                    width: 10px;
                }
                QCheckBox {
                    font-size: 14px;
                    margin: 5px;
                    color: #ffffff;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f0f0f0;
                }
                QLabel {
                    color: #333333;
                }
                QPushButton {
                    background-color: #4CAF50;
                    border: 2px solid #4CAF50;
                    color: white;
                    font-size: 16px;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                    border: 2px solid #45a049;
                }
                QPushButton:pressed {
                    background-color: #387038;
                    border: 2px solid #387038;
                }
                QProgressBar {
                    border: 2px solid #cccccc;
                    border-radius: 5px;
                    text-align: center;
                    background-color: rgba(255, 255, 255, 0.8);
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    width: 10px;
                }
                QCheckBox {
                    font-size: 14px;
                    margin: 5px;
                }
            """)

    def selectTrainFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder for Training')
        if folder_path:
            self.train_data_dir = folder_path
            self.trainModel(folder_path)

    def trainModel(self, folder_path):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.updateProgress(10))
        self.timer.start(500)

        datagen = ImageDataGenerator(rescale=1.0/255.0)
        train_generator = datagen.flow_from_directory(
            folder_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical'
        )

        self.num_images = train_generator.samples
        if self.num_images == 0:
            self.image_label.setText('No images found in the selected folder.')
            self.progress_bar.setVisible(False)
            return

        print(f"Total images used for training: {self.num_images}")

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(train_generator, epochs=5)

        self.timer.stop()
        self.progress_bar.setVisible(False)

        for epoch in range(5):
            print(f"Epoch {epoch+1}: Loss = {history.history['loss'][epoch]:.4f}, Accuracy = {history.history['accuracy'][epoch]*100:.2f}%")

        self.model.save('trained_model.h5')
        self.image_label.setText(f'Model trained with {self.num_images} images! Now upload an image for prediction.')

    def updateProgress(self, value):
        if self.progress_bar.value() < 100:
            self.progress_bar.setValue(self.progress_bar.value() + value)
        else:
            self.timer.stop()

    def uploadTestImage(self):
        image_path, _ = QFileDialog.getOpenFileName(self, 'Select Image for Prediction', '', 'Images (*.png *.jpg *.bmp)')
        if image_path:
            print(f"Uploaded image: {image_path}")
            self.current_image = QImage(image_path)
            self.displayImage(self.current_image)
            self.predictImage(image_path)

    def displayImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def rotateLeft(self):
        if self.current_image:
            transform = QTransform().rotate(-90)
            self.current_image = self.current_image.transformed(transform)
            self.displayImage(self.current_image)

    def rotateRight(self):
        if self.current_image:
            transform = QTransform().rotate(90)
            self.current_image = self.current_image.transformed(transform)
            self.displayImage(self.current_image)

    def mirrorImage(self):
        if self.current_image:
            self.current_image = self.current_image.mirrored(True, False)
            self.displayImage(self.current_image)

    def clearImage(self):
        self.current_image = None
        self.image_label.setText("Your selected image will appear here.")
        self.predictions_label.setText("Model's Predictions")

    def saveImage(self):
        if self.current_image:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Images (*.png *.jpg *.bmp)')
            if file_path:
                self.current_image.save(file_path)
                QMessageBox.information(self, "Success", "Image saved successfully!")

    def predictImage(self, image_path):
        img = load_img(image_path, target_size=(150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        class_labels = list(os.listdir(self.train_data_dir))
        predicted_label = class_labels[predicted_class]

        confidence = predictions[0][predicted_class] * 100

        result_text = f'<b>Prediction:</b> {predicted_label} ({confidence:.2f}% confidence)<br>'
        result_text += f'<b>Trained with:</b> {self.num_images} images.<br><br>'
        result_text += '<b>Probabilities:</b><br>'

        for i, label in enumerate(class_labels):
            probability = predictions[0][i] * 100
            result_text += f'{label}: {probability:.2f}%<br>'

        self.predictions_label.setText(result_text)

    def showHelp(self):
        help_text = """
        <h2>Image Editor Help</h2>
        <p><b>1. Upload Training Images:</b> Select a folder containing images to train the AI model.</p>
        <p><b>2. Upload Prediction Image:</b> Select an image to predict its class using the trained model.</p>
        <p><b>3. Rotate Left/Right:</b> Rotate the uploaded image 90 degrees left or right.</p>
        <p><b>4. Mirror:</b> Flip the uploaded image horizontally.</p>
        <p><b>5. Clear Image:</b> Reset the displayed image.</p>
        <p><b>6. Save Image:</b> Save the manipulated image to your computer.</p>
        <p><b>7. Progress Bar:</b> Shows the progress of model training.</p>
        """
        QMessageBox.information(self, "Help", help_text)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
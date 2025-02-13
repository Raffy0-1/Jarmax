import speech_recognition as sr
import pyttsx3
import sqlite3
import cv2
import numpy as np
from transformers import pipeline
import os
from advance_thinking import start_chat, chat_with_bot
import sys
import logging
import re
import random
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def sanitize_input(user_input):
    """Sanitize input to allow only safe characters."""
    return re.sub(r'[^a-zA-Z0-9 .,!?\"]', '', user_input)

# Configure logging
logging.basicConfig(filename="jarmax.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

class Jarmax:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.sentiment_model = pipeline('sentiment-analysis', model='distilbert-base-uncased')
        
        # Database connection and table creation
        self.conn = sqlite3.connect('jarmax_memory.db', check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS preferences (id INTEGER PRIMARY KEY, preference TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, memory TEXT)''')
        # Table for ML model metrics (for deep learning feature)
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS ml_metrics (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                                    model_name TEXT, 
                                    accuracy REAL, 
                                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        # Table for face recognition metrics
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS face_ml_metrics (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    model_name TEXT,
                                    accuracy REAL,
                                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self.input_method = ""
        
        # For recording sentiment analysis history
        self.sentiment_history = []  # Each entry is a dict: {text, sentiment, confidence}
        
        # Face recognition initialization
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists("face_model.xml"):
            self.face_recognizer.read("face_model.xml")
        else:
            logging.warning("face_model.xml not found. Face authentication may not work until you train a model.")
            print("Warning: face_model.xml not found. Face authentication may not work until you train a model.")
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.questions, self.answers, self.vectorizer, self.question_matrix = start_chat()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio)
                print(f"User: {command}")
                return sanitize_input(command.lower())
            except sr.UnknownValueError:
                return "Sorry, I didn't get that."
            except sr.RequestError:
                return "Speech recognition service is unavailable."
            except Exception as e:
                logging.error(f"Error in listen(): {e}")
                return "An error occurred."

    def sentiment_analysis(self, text):
        result = self.sentiment_model(text)
        sentiment = result[0]['label']
        confidence = result[0]['score']
        sentiment_str = f"{sentiment.capitalize()} (Confidence: {confidence:.2f})"
        # Save to history
        self.sentiment_history.append({
            'text': text,
            'sentiment': sentiment.capitalize(),
            'confidence': confidence
        })
        return sentiment_str

    def face_authentication(self):
        if not os.path.exists("face_model.xml"):
            self.speak("Face model not found. Authentication unavailable.")
            return False
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self.speak("Error accessing camera.")
            return False
        
        ret, frame = camera.read()
        camera.release()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                # Resize face to 100x100 for consistency with our training
                face_roi = cv2.resize(face_roi, (100, 100))
                label, confidence = self.face_recognizer.predict(face_roi)
                # For LBPH, a lower confidence indicates a better match
                if confidence < 50:
                    return True
        return False

    def assist_with_typing(self):
        print("You can type your command now: ")
        user_input = input("Enter command: ")
        return sanitize_input(user_input.lower())

    def show_menu(self):
        self.speak("Welcome to Jarmax. Would you like to use voice commands or text commands? Please say 'voice' or 'text'.")
        command = self.listen()
        if 'voice' in command:
            self.input_method = "voice"
            self.speak("You have selected voice commands.")
        elif 'text' in command:
            self.input_method = "text"
            self.speak("You have selected text commands.")
        else:
            self.speak("I didn't understand. Please say 'voice' or 'text'.")
            self.show_menu()

    def healthcare_features(self):
        from health import RaymaxHealthcareRobot
        health_bot = RaymaxHealthcareRobot()
        health_bot.main_menu()

    def entertainment_features(self):
        """Provides jokes, compliments, and fun facts."""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "I told my computer I needed a break, and it said, 'No problem, I'll go to sleep.'",
            "Why did the tomato turn red? Because it saw the salad dressing!"
        ]
        compliments = [
            "You're looking fantastic today!",
            "Your positivity is infectious!",
            "You have a great sense of humor!"
        ]
        fun_facts = [
            "Did you know that honey never spoils?",
            "Bananas are berries, but strawberries aren't.",
            "A group of flamingos is called a 'flamboyance'."
        ]
        if self.input_method == "voice":
            while True:
                self.speak("Entertainment mode. Say 'joke' for a joke, 'compliment' for a compliment, 'fact' for a fun fact, or say 'back' to return to the main menu.")
                command = self.listen()
                if "back" in command:
                    self.speak("Exiting entertainment mode.")
                    break
                elif "joke" in command:
                    self.speak(random.choice(jokes))
                elif "compliment" in command:
                    self.speak(random.choice(compliments))
                elif "fact" in command:
                    self.speak(random.choice(fun_facts))
                else:
                    self.speak("I didn't understand. Please say 'joke', 'compliment', 'fact', or 'back'.")
        else:
            while True:
                print("\nEntertainment Menu:")
                print("1. Tell me a joke")
                print("2. Give me a compliment")
                print("3. Share a fun fact")
                print("4. Back to main menu")
                choice = input("Enter your choice (1-4): ").strip()
                if choice == '4':
                    print("Exiting entertainment mode.")
                    self.speak("Exiting entertainment mode.")
                    break
                elif choice == '1':
                    joke = random.choice(jokes)
                    print("Joke: " + joke)
                    self.speak(joke)
                elif choice == '2':
                    comp = random.choice(compliments)
                    print("Compliment: " + comp)
                    self.speak(comp)
                elif choice == '3':
                    fact = random.choice(fun_facts)
                    print("Fun Fact: " + fact)
                    self.speak(fact)
                else:
                    print("Invalid choice. Please try again.")
                    self.speak("Invalid choice. Please try again.")

    def learning_memory_features(self):
        """Allows Jarmax to learn new information and recall stored memories."""
        if self.input_method == "voice":
            while True:
                self.speak("Learning and Memory mode. Say 'add' to store a memory, 'retrieve' to hear stored memories, or 'back' to return to the main menu.")
                command = self.listen()
                if "back" in command:
                    self.speak("Exiting learning and memory mode.")
                    break
                elif "add" in command:
                    self.speak("What memory would you like me to remember?")
                    memory_text = self.listen()
                    memory_text = sanitize_input(memory_text)
                    try:
                        self.cursor.execute('INSERT INTO memory (memory) VALUES (?)', (memory_text,))
                        self.conn.commit()
                        self.speak("Memory added successfully.")
                    except Exception as e:
                        logging.error(f"Error adding memory: {e}")
                        self.speak("Failed to add memory.")
                elif "retrieve" in command:
                    try:
                        self.cursor.execute('SELECT memory FROM memory')
                        memories = self.cursor.fetchall()
                        if memories:
                            for mem in memories:
                                self.speak(mem[0])
                                print(f"Memory: {mem[0]}")
                        else:
                            self.speak("No memories found.")
                    except Exception as e:
                        logging.error(f"Error retrieving memories: {e}")
                        self.speak("Failed to retrieve memories.")
                else:
                    self.speak("I didn't understand. Please say 'add', 'retrieve', or 'back'.")
        else:
            while True:
                print("\nLearning and Memory Menu:")
                print("1. Add a memory")
                print("2. Retrieve memories")
                print("3. Back to main menu")
                choice = input("Enter your choice (1-3): ").strip()
                if choice == '3':
                    print("Exiting learning and memory mode.")
                    self.speak("Exiting learning and memory mode.")
                    break
                elif choice == '1':
                    memory_text = input("Enter the memory you want me to remember: ")
                    memory_text = sanitize_input(memory_text)
                    try:
                        self.cursor.execute('INSERT INTO memory (memory) VALUES (?)', (memory_text,))
                        self.conn.commit()
                        print("Memory added successfully.")
                        self.speak("Memory added successfully.")
                    except Exception as e:
                        logging.error(f"Error adding memory: {e}")
                        print("Failed to add memory.")
                        self.speak("Failed to add memory.")
                elif choice == '2':
                    try:
                        self.cursor.execute('SELECT memory FROM memory')
                        memories = self.cursor.fetchall()
                        if memories:
                            for mem in memories:
                                print(f"Memory: {mem[0]}")
                                self.speak(mem[0])
                        else:
                            print("No memories found.")
                            self.speak("No memories found.")
                    except Exception as e:
                        logging.error(f"Error retrieving memories: {e}")
                        print("Failed to retrieve memories.")
                        self.speak("Failed to retrieve memories.")
                else:
                    print("Invalid choice. Please try again.")
                    self.speak("Invalid choice. Please try again.")

    def visualization_features(self):
        """Displays visualizations for sentiment history and memory storage."""
        # Visualization for Sentiment History
        if self.sentiment_history:
            sentiments = [entry['sentiment'] for entry in self.sentiment_history]
            sentiment_counts = Counter(sentiments)
            labels = list(sentiment_counts.keys())
            counts = list(sentiment_counts.values())
            
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.bar(labels, counts, color=['green', 'red', 'gray'])
            plt.title('Sentiment Analysis History')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
        else:
            print("No sentiment history to display.")
        
        # Visualization for Memory Storage
        self.cursor.execute('SELECT id, memory FROM memory')
        memories = self.cursor.fetchall()
        if memories:
            memory_ids = [mem[0] for mem in memories]
            # Calculate word count for each memory
            word_counts = [len(mem[1].split()) for mem in memories]
            
            plt.subplot(1, 2, 2)
            plt.bar(memory_ids, word_counts, color='blue')
            plt.title('Memory Word Count')
            plt.xlabel('Memory ID')
            plt.ylabel('Word Count')
        else:
            print("No memories stored to display.")
        
        plt.tight_layout()
        plt.show()

    def ml_model_features(self):
        """Trains a machine learning model on the Iris dataset, stores its accuracy, and visualizes performance."""
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        # Load Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train an MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_name = "MLP_Iris"
        
        # Insert current accuracy into the ml_metrics table
        self.cursor.execute('INSERT INTO ml_metrics (model_name, accuracy) VALUES (?, ?)', (model_name, acc))
        self.conn.commit()
        
        # Plot the confusion matrix for the current run
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name} (Accuracy: {acc:.2f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Query historical accuracies from the ml_metrics table
        self.cursor.execute('SELECT id, accuracy FROM ml_metrics')
        records = self.cursor.fetchall()
        if records:
            ids = [r[0] for r in records]
            accuracies = [r[1] for r in records]
            plt.subplot(1, 2, 2)
            plt.bar(ids, accuracies, color='green')
            plt.title('Historical Model Accuracies')
            plt.xlabel('Run ID')
            plt.ylabel('Accuracy')
        else:
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, 'No historical data', horizontalalignment='center')
            plt.title('Historical Model Accuracies')
        
        plt.tight_layout()
        plt.show()
        
        self.speak(f"The current model achieved an accuracy of {acc:.2f}")

    def facial_recognition_training_features(self):
        """
        Trains a facial recognition model using LBPH on images in the 'faces_dataset' folder,
        evaluates its performance, saves the model, and visualizes the results.
        The folder structure should be: faces_dataset/<person_name>/image_files.
        """
        dataset_path = "faces_dataset"
        if not os.path.exists(dataset_path):
            self.speak("Face dataset not found. Please create a folder named 'faces_dataset' with subfolders for each person.")
            return

        images = []
        labels = []
        label_map = {}
        current_label = 0

        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                label_map[current_label] = folder
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    # Detect face in the image
                    faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                    if len(faces) == 0:
                        continue
                    (x, y, w, h) = faces[0]
                    face_img = img[y:y+h, x:x+w]
                    # Resize for consistency
                    face_img = cv2.resize(face_img, (100, 100))
                    images.append(face_img)
                    labels.append(current_label)
                current_label += 1

        if len(images) == 0:
            self.speak("No face images found in the dataset.")
            return

        # Split data into train and test sets
        from sklearn.model_selection import train_test_split
        images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Train LBPHFaceRecognizer model
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images_train, np.array(labels_train))

        # Evaluate on test set
        predictions = []
        for img in images_test:
            pred_label, conf = model.predict(img)
            predictions.append(pred_label)

        accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == labels_test[i]]) / len(predictions)
        model_name = "LBPH_Face"

        # Insert current accuracy into the face_ml_metrics table
        self.cursor.execute('INSERT INTO face_ml_metrics (model_name, accuracy) VALUES (?, ?)', (model_name, accuracy))
        self.conn.commit()

        # Visualize the confusion matrix and historical accuracies
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels_test, predictions)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
        plt.title(f'Confusion Matrix for {model_name} (Accuracy: {accuracy:.2f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        self.cursor.execute('SELECT id, accuracy FROM face_ml_metrics')
        records = self.cursor.fetchall()
        if records:
            ids = [r[0] for r in records]
            accuracies = [r[1] for r in records]
            plt.subplot(1, 2, 2)
            plt.bar(ids, accuracies, color='orange')
            plt.title('Historical Face Recognition Accuracies')
            plt.xlabel('Run ID')
            plt.ylabel('Accuracy')
        else:
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, 'No historical data', horizontalalignment='center')
            plt.title('Historical Face Recognition Accuracies')

        plt.tight_layout()
        plt.show()

        self.speak(f"Facial recognition training completed with an accuracy of {accuracy:.2f}")
        # Save the trained model and update the current recognizer
        model.write("face_model.xml")
        self.face_recognizer = model

    def run(self):
        self.show_menu()
        if self.input_method == "voice":
            self.speak("Voice mode activated. How can I assist you?")
            while True:
                command = self.listen()
                if "exit" in command:
                    self.speak("Goodbye!")
                    break
                elif "how are you" in command:
                    self.speak("I am functioning optimally. How about you?")
                elif "analyze sentiment" in command:
                    self.speak("Please provide text input.")
                    user_text = self.listen()
                    sentiment = self.sentiment_analysis(user_text)
                    self.speak(f"The sentiment of your text is: {sentiment}")
                elif "authenticate face" in command:
                    if self.face_authentication():
                        self.speak("Authentication successful!")
                    else:
                        self.speak("Authentication failed!")
                elif "chat" in command:
                    self.speak("Starting chatbot. Please ask a question.")
                    while True:
                        user_input = self.listen()
                        if "exit" in user_input:
                            self.speak("Exiting chat mode.")
                            break
                        response = chat_with_bot(user_input, self.questions, self.answers, self.vectorizer, self.question_matrix)
                        self.speak(response)
                elif "health" in command:
                    self.speak("Opening healthcare menu...")
                    self.healthcare_features()
                elif "entertainment" in command:
                    self.speak("Opening entertainment features...")
                    self.entertainment_features()
                elif "learn" in command or "memory" in command:
                    self.speak("Opening learning and memory features...")
                    self.learning_memory_features()
                elif "visual" in command:
                    self.speak("Opening visualization features...")
                    self.visualization_features()
                elif ("ml" in command or "model" in command or "machine learning" in command 
                      or "deep learning" in command):
                    self.speak("Training machine learning model and visualizing performance...")
                    self.ml_model_features()
                elif "facial recognition" in command and "train" in command:
                    self.speak("Starting facial recognition training and evaluation...")
                    self.facial_recognition_training_features()
                else:
                    self.speak("I did not understand that command.")
        elif self.input_method == "text":
            self.speak("Text mode activated. How can I assist you?")
            while True:
                print("\nMain Menu:")
                print("1. Analyze sentiment")
                print("2. Authenticate face")
                print("3. Chat")
                print("4. Healthcare menu")
                print("5. Entertainment")
                print("6. Learning and Memory")
                print("7. Visualizations")
                print("8. ML Model Feature")
                print("9. Facial Recognition Training")
                print("10. Exit")
                choice = input("Enter your choice (1-10): ").strip()
                if choice == '10':
                    self.speak("Goodbye!")
                    break
                elif choice == '1':
                    self.speak("Please provide text input.")
                    user_text = input("Enter text: ")
                    sentiment = self.sentiment_analysis(user_text)
                    print(f"The sentiment of your text is: {sentiment}")
                    self.speak(f"The sentiment of your text is: {sentiment}")
                elif choice == '2':
                    if self.face_authentication():
                        self.speak("Authentication successful!")
                    else:
                        self.speak("Authentication failed!")
                elif choice == '3':
                    self.speak("Starting chatbot. Please ask a question.")
                    while True:
                        user_input = input("Ask a question or type 'exit' to quit: ")
                        if user_input.lower() == 'exit':
                            self.speak("Exiting chat mode.")
                            break
                        response = chat_with_bot(user_input, self.questions, self.answers, self.vectorizer, self.question_matrix)
                        print(f"Bot: {response}")
                        self.speak(response)
                elif choice == '4':
                    self.healthcare_features()
                elif choice == '5':
                    self.entertainment_features()
                elif choice == '6':
                    self.learning_memory_features()
                elif choice == '7':
                    self.speak("Opening visualization features...")
                    self.visualization_features()
                elif choice == '8':
                    self.speak("Training machine learning model and visualizing performance...")
                    self.ml_model_features()
                elif choice == '9':
                    self.speak("Starting facial recognition training and evaluation...")
                    self.facial_recognition_training_features()
                else:
                    print("Invalid choice. Please try again.")
                    self.speak("Invalid choice. Please try again.")

if __name__ == "__main__":
    jarmax = Jarmax()
    jarmax.run()

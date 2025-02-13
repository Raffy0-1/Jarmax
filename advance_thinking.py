# Advanced_Thinking.py
import csv
import random
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download necessary NLP packages
nltk.download('punkt')

# Load dataset
def load_knowledge_base():
    questions, answers = [], []
    
    with open("Knowledge_base.csv", mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            questions.append(row[1])
            answers.append(row[2])
    
    return questions, answers

# Function to detect if the user input contains a mathematical operation
def detect_math_expression(user_input):
    # Regex to match common mathematical expressions
    math_expression = re.match(r'^\d+[\+\-\*/\^]\d+$', user_input.strip())
    if math_expression:
        return True
    return False

# Function to process and extract relevant keywords or patterns from the user input
def process_input(user_input):
    # Tokenize input and clean up
    tokens = word_tokenize(user_input.lower())
    return tokens

# Get response using NLP
def get_response(user_input, questions, answers, vectorizer, question_matrix):
    user_input_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_input_vec, question_matrix)
    index = np.argmax(similarity)
    
    return answers[index] if similarity[0, index] > 0.2 else "Sorry, I don't understand that."

# Handle user input
def handle_input(user_input, questions, answers, vectorizer, question_matrix):
    if detect_math_expression(user_input):  # Check if it's a math expression
        try:
            result = eval(user_input)  # Safely evaluate the mathematical expression
            return f"The answer to {user_input} is {result}"
        except:
            return "Sorry, there was an error with the math expression."
    
    # If not a math expression, proceed with standard question matching
    response = get_response(user_input, questions, answers, vectorizer, question_matrix)
    return response

# Main function to start the chat with a bot
def start_chat():
    questions, answers = load_knowledge_base()
    vectorizer = TfidfVectorizer()
    question_matrix = vectorizer.fit_transform(questions)
    
    return questions, answers, vectorizer, question_matrix

# Chat handler to return responses for the user
def chat_with_bot(user_input, questions, answers, vectorizer, question_matrix):
    response = handle_input(user_input, questions, answers, vectorizer, question_matrix)
    return response

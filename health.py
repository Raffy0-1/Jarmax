
import sys
class RaymaxHealthcareRobot:
    
    def __init__(self):
        self.menu_options = {
            "1": self.bmi_calculator,
            "2": self.water_intake_recommendation,
            "3": self.calorie_needs_estimation,
            "4": self.basic_symptom_checker,
            "5": self.heart_rate_checker,
            "6": self.exercise_suggestions,
            "7": self.sleep_tracker,
            "8": self.mental_health_tips,
            "9": self.medicine_suggestion_system,
            "10": self.first_aid_guide
        }

    def greetings(self):
        print("\nHi! I am Raymax, apka apna personal healthcare Robot (inspired from Baymax).")
        print("How are you feeling today?\n")
        user_response = input("Please tell me (e.g., yes I am fine, no I am not fine, etc.): ").lower()
        print("That's great to hear! Stay healthy and happy.\n" if "yes" in user_response else
              "I hope you feel better soon. Remember, seeking help is always a good option.\n")
        if input("Do you need any assistance from me today? (1 for Yes, 0 for No): ") == "1":
            self.main_menu()
        else:
            print("Goodbye! Take care of yourself. Stay safe and healthy!")

    def main_menu(self):
        print("\nPlease choose what care you need:")
        for key, value in self.menu_options.items():
            print(f"{key}. {value.__doc__}")
        print("0. Exit")
        choice = input("\nEnter your choice (0-10): ").strip()
        if choice == "0":
            print("Goodbye! Stay healthy!\n")
            sys.exit()
        elif choice in self.menu_options:
            self.menu_options[choice]()
        else:
            print("Invalid choice. Please try again.")
        print("-" * 50)
        self.main_menu()

    def bmi_calculator(self):
        """BMI Calculator - Know your Body Mass Index."""
        weight = float(input("Enter your weight (kg): "))
        height = float(input("Enter your height (m): "))
        bmi = weight / height**2
        print(f"Your BMI is: {bmi:.2f}")
        print("You are underweight." if 9 <= bmi <= 18.5 else  
              "You have a healthy weight." if bmi < 24.9 else
              "You are overweight." if bmi < 29.9 else "You are obese. Consult a healthcare provider.")

    def water_intake_recommendation(self):
        """Water Intake Recommendation - Prevent dehydration."""
        weight = float(input("Enter your weight (kg): "))
        print(f"Recommended daily water intake: {weight * 35 / 1000:.2f} liters.")

    def calorie_needs_estimation(self):
        """Calorie Needs Estimation - Manage weight."""
        gender = input("Enter your gender (Male/Female): ").lower()
        age = int(input("Enter your age: "))
        weight = float(input("Enter your weight (kg): "))
        height = float(input("Enter your height (cm): "))
        bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161)
        activity_multiplier = {
            "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
            "active": 1.725, "very active": 1.9
        }
        activity = input("Enter your activity level (Sedentary, Light, Moderate, Active, Very Active): ").lower()
        total_calories = bmr * activity_multiplier.get(activity, 0)
        if total_calories:
            print(f"Your estimated daily calorie needs: {total_calories:.2f} calories.")
        else:
            print("Invalid activity level. Please try again.")

    def basic_symptom_checker(self):
        """Basic Symptom Checker - Suggestions for common symptoms."""
        symptoms = {
            "headache": "Rest, hydrate, and avoid screens.",
            "fever": "Rest, hydrate, monitor temperature, and see a doctor if it persists.",
            "cough": "Hydrate and use cough syrup if needed. Consult a doctor if necessary."
        }
        symptom = input("Describe your symptom (e.g., headache, fever, cough): ").lower()
        print(symptoms.get(symptom, "Symptom not recognized. Consult a healthcare provider if concerned."))

    def heart_rate_checker(self):
        """Heart Rate Checker - Monitor your heart health."""
        hr = int(input("Enter your heart rate (bpm): "))
        print("Below normal, consult a healthcare provider." if hr < 60 else
              "Normal range, keep it up!" if hr <= 100 else
              "Above normal. Consult a doctor.")

    def exercise_suggestions(self):
        """Exercise Suggestions - Based on fitness level."""
        suggestions = {
            "beginner": "Start with walking, light jogging, or basic exercises.",
            "intermediate": "Try running, cycling, or resistance training.",
            "advanced": "High-intensity interval training or heavy weightlifting."
        }
        level = input("Enter your fitness level (Beginner, Intermediate, Advanced): ").lower()
        print(suggestions.get(level, "Invalid level. Choose Beginner, Intermediate, or Advanced."))

    def sleep_tracker(self):
        """Sleep Tracker - Analyze your sleep patterns."""
        hours = float(input("How many hours did you sleep last night? "))
        print("Less than recommended. Aim for 7-9 hours." if hours < 6 else
              "Healthy amount of sleep. Keep it up!" if hours <= 9 else
              "Oversleeping can lead to health issues. Adjust your routine.")

    def mental_health_tips(self):
        """Mental Health Tips - Suggestions for well-being."""
        mood = input("How are you feeling today? (e.g., stressed, anxious, happy): ").lower()
        print("Relaxation techniques or meditation may help." if "stressed" in mood else
              "Consider talking to a counselor or practicing mindfulness." if "anxious" in mood else
              "Keep up the positive vibes!" if "happy" in mood else
              "It's okay to seek help if you're feeling down.")

    def medicine_suggestion_system(self):
        """Medicine Suggestion System - Info about common medicines."""
        suggestions = {
            "headache": "Use paracetamol or ibuprofen.",
            "cold": "Take decongestants or warm fluids.",
            "fever": "Consider acetaminophen and ibuprofen."
        }
        symptom = input("Describe your symptom (e.g., headache, cold, fever): ").lower()
        print(suggestions.get(symptom, "Consult a healthcare provider for advice."))

    def first_aid_guide(self):
        """Basic First Aid Guide - Handle minor injuries."""
        guides = {
            "cut": "Clean with water, apply antiseptic, and bandage.",
            "burn": "Cool with water, cover with a clean cloth.",
            "sprain": "Rest, ice, and elevate the area."
        }
        injury = input("Describe the injury (e.g., cut, burn, sprain): ").lower()
        print(guides.get(injury, "For serious injuries, contact emergency services."))


if __name__ == "__main__":
    robot = RaymaxHealthcareRobot()
    robot.greetings()

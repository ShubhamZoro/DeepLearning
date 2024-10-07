import speech_recognition as sr
from gtts import gTTS
import difflib
import tkinter as tk
from tkinter import messagebox

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to capture speech and convert it to text
def get_speech_input():
    with sr.Microphone() as source:
        status_label.config(text="Please speak...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        spoken_text = recognizer.recognize_google(audio)
        status_label.config(text=f"You said: {spoken_text}")
        return spoken_text
    except sr.UnknownValueError:
        status_label.config(text="Sorry, I could not understand your speech.")
    except sr.RequestError:
        status_label.config(text="Request error from the speech recognition service.")
    return None

# Function to compare pronunciation and provide feedback
def pronunciation_feedback(spoken_text, correct_text):
    if spoken_text:
        spoken_text = spoken_text.lower()
        correct_text = correct_text.lower()
        if spoken_text == correct_text:
            messagebox.showinfo("Feedback", "Pronunciation is correct!")
        else:
            messagebox.showerror("Feedback", "Incorrect pronunciation.")
            diff = difflib.ndiff(spoken_text.split(), correct_text.split())
            differences = list(diff)
            diff_str = "\n".join(differences)
            messagebox.showinfo("Differences", diff_str)

# Function to handle speech input and feedback
def check_pronunciation():
    correct_text = entry_correct.get()
    if correct_text.strip() == "":
        messagebox.showwarning("Input Error", "Please enter the correct text.")
        return
    spoken_text = get_speech_input()
    pronunciation_feedback(spoken_text, correct_text)

# Create the Tkinter app window
root = tk.Tk()
root.title("Pronunciation Assessment")

# Create widgets
label_correct = tk.Label(root, text="Enter Correct Sentence:")
label_correct.pack(pady=5)

entry_correct = tk.Entry(root, width=50)
entry_correct.pack(pady=5)

status_label = tk.Label(root, text="Press 'Check Pronunciation' to start", fg="blue")
status_label.pack(pady=5)

button_check = tk.Button(root, text="Check Pronunciation", command=check_pronunciation)
button_check.pack(pady=10)

# Run the Tkinter app
root.mainloop()


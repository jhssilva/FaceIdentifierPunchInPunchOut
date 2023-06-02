from os import system
import threading
import speech_recognition as sr


def read_message(message):
    thread = threading.Thread(target=message_reading, args=(message,))
    thread.start()


def message_reading(message):
    system(f'say {message}')


def start_listen(callback):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        # audio = recognizer.listen(source)
        audio1 = recognizer.record(source, duration=4)
        word = recognizer.recognize_google(audio1)
        callback(word)

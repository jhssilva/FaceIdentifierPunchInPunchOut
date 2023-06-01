from os import system
import threading


def say_hello():
    system('say Hello world!')


def read_message(message):
    thread = threading.Thread(target=message_reading, args=(message,))
    thread.start()


def message_reading(message):
    system(f'say {message}')


def read_start_menu():
    read_message(
        "Welcome to the menu of the face recognition system that tracks the punch in and punch out of employees.")

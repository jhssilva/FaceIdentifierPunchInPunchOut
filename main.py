import sys
import random
from PySide6 import QtCore, QtWidgets
from voice import read_message, read_start_menu


from camera import camera
from manage_employee import add_employee, remove_employee, get_punch_in_out_info_employee
from train_model import train_model


class Menu(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # * Button creation
        self.buttonStart = QtWidgets.QPushButton("Start")
        self.buttonReadPunchInOutInfoEmployee = QtWidgets.QPushButton(
            "Read Punch In Out Info Employee")
        self.buttonTrainModel = QtWidgets.QPushButton("Train Model")
        self.buttonAddEmployee = QtWidgets.QPushButton("Add Employee")
        self.buttonRemoveEmployee = QtWidgets.QPushButton("Remove Employee")
        self.buttonExit = QtWidgets.QPushButton("Exit")

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(10)

        # * Add Widgets to layout
        self.layout.addWidget(self.buttonStart)
        self.layout.addWidget(self.buttonReadPunchInOutInfoEmployee)
        self.layout.addWidget(self.buttonTrainModel)
        self.layout.addWidget(self.buttonAddEmployee)
        self.layout.addWidget(self.buttonRemoveEmployee)
        self.layout.addWidget(self.buttonExit)

        # * Connect buttons to slots
        self.buttonStart.clicked.connect(self.handleButtonStart)
        self.buttonReadPunchInOutInfoEmployee.clicked.connect(
            self.handleButtonReadPunchInOutInfoEmployee)
        self.buttonTrainModel.clicked.connect(self.handleButtonTrainModel)
        self.buttonAddEmployee.clicked.connect(self.handleButtonAddEmployee)
        self.buttonRemoveEmployee.clicked.connect(
            self.handleButtonRemoveEmployee)
        self.buttonExit.clicked.connect(self.handleButtonExit)
        self.setLayout(self.layout)

    @QtCore.Slot()
    def handleButtonStart(self):
        read_message("Opening Camera!")
        has_worked = camera()
        if not has_worked:
            read_message("Error opening camera!")
            print("Error opening camera!")
        print("Camera Closed!")
        read_message("Camera Closed!")

    @QtCore.Slot()
    def handleButtonReadPunchInOutInfoEmployee(self):
        read_message("Opening Read Punch In Out Info Employee!")
        get_punch_in_out_info_employee()

    @QtCore.Slot()
    def handleButtonTrainModel(self):
        read_message("Opening Model Training!")
        train_model()

    @QtCore.Slot()
    def handleButtonAddEmployee(self):
        read_message("Opening Add Employee!")
        add_employee()

    @QtCore.Slot()
    def handleButtonRemoveEmployee(self):
        read_message("Opening Remove Employee!")
        remove_employee()

    @QtCore.Slot()
    def handleButtonExit(self):
        read_message("Exiting the application!")
        exit()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    # * Window creation
    widget = Menu()
    widget.resize(300, 200)
    widget.show()

    # * Read start menu
    read_start_menu()

    sys.exit(app.exec())

import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

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
        has_worked = camera()
        if not has_worked:
            print("Error opening camera!")
        print("Camera Closed!")

    @QtCore.Slot()
    def handleButtonReadPunchInOutInfoEmployee(self):
        get_punch_in_out_info_employee()

    @QtCore.Slot()
    def handleButtonTrainModel(self):
        train_model()

    @QtCore.Slot()
    def handleButtonAddEmployee(self):
        add_employee()

    @QtCore.Slot()
    def handleButtonRemoveEmployee(self):
        remove_employee()

    @QtCore.Slot()
    def handleButtonExit(self):
        exit()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Menu()
    widget.resize(300, 200)
    widget.show()

    sys.exit(app.exec())

import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

from camera import camera
from manage_employee import add_employee, remove_employee


class Menu(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # * Button creation
        self.buttonStart = QtWidgets.QPushButton("Start")
        self.buttonTrainModel = QtWidgets.QPushButton("Train Model")
        self.buttonAddEmployee = QtWidgets.QPushButton("Add Employee")
        self.buttonRemoveEmployee = QtWidgets.QPushButton("Remove Employee")
        self.buttonExit = QtWidgets.QPushButton("Exit")

        self.layout = QtWidgets.QVBoxLayout(self)

        # * Add Widgets to layout
        self.layout.addWidget(self.buttonStart)
        self.layout.addWidget(self.buttonTrainModel)
        self.layout.addWidget(self.buttonAddEmployee)
        self.layout.addWidget(self.buttonRemoveEmployee)
        self.layout.addWidget(self.buttonExit)

        # * Connect buttons to slots
        self.buttonStart.clicked.connect(self.handleButtonStart)
        self.buttonTrainModel.clicked.connect(self.handleButtonTrainModel)
        self.buttonAddEmployee.clicked.connect(self.handleButtonAddEmployee)
        self.buttonRemoveEmployee.clicked.connect(
            self.handleButtonRemoveEmployee)
        self.buttonExit.clicked.connect(self.handleButtonExit)

    @QtCore.Slot()
    def handleButtonStart(self):
        has_worked = camera()
        if not has_worked:
            print("Error")

    @QtCore.Slot()
    def handleButtonTrainModel(self):
        print("Train Model")

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
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())

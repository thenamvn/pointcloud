import sys
from gui import CylinderAnalyzerGUI
from PyQt6.QtWidgets import QApplication
import matplotlib
matplotlib.use('Qt5Agg')  # Set matplotlib backend

def main():
    app = QApplication(sys.argv)
    window = CylinderAnalyzerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
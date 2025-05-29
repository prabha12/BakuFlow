import sys
from PyQt5.QtWidgets import QApplication
# Ensure the import path is correct based on the new structure
from labelimg.gui.main_window import LabelingTool

def main():
    app = QApplication(sys.argv)
    ex = LabelingTool()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 
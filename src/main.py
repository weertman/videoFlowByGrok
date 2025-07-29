import sys
import logging
from PySide6.QtWidgets import QApplication
from src.ui import MainUI

# Set up logging
logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        # Optional: Load stylesheet if assets/styles.qss exists
        # with open('assets/styles.qss', 'r') as f:
        #     app.setStyleSheet(f.read())
        window = MainUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
        sys.exit(1)
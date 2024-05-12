import sys
from PyQt5.QtWidgets import QApplication
from ui import MainWindow
from threading import Thread
from data import DataHandler

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    # Data handling
    data_handler = DataHandler()

    # Start threads for data polling and updating plots
    polling_thread = Thread(target=data_handler.poll_data, daemon=True)
    processing_thread = Thread(target=data_handler.process_data, daemon=True)
    update_thread = Thread(target=main_window.update_plots, args=(data_handler,), daemon=True)

    polling_thread.start()
    processing_thread.start()
    update_thread.start()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

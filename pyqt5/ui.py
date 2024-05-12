from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg

class PlotWidget(pg.PlotWidget):
    def __init__(self, y_range, plot_type="line", **kwargs):
        super().__init__(**kwargs)
        self.plot_type = plot_type
        self.setRange(yRange=y_range)
        if plot_type == "spectrogram":
            self.img = pg.ImageItem()
            self.addItem(self.img)
        else:
            self.plot = self.plot()

    def update_plot(self, data):
        if self.plot_type == "spectrogram":
            self.img.setImage(data)
        else:
            self.plot.setData(data)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Real-time Plotting Application')
        self.general_layout = QVBoxLayout()

        # Create and add plots
        self.plots = [
            PlotWidget((-0.1, 0.1)),
            PlotWidget((0, 4000), plot_type="spectrogram"),
            PlotWidget((0, 100)),
            PlotWidget((0, 100))
        ]

        central_widget = QWidget()
        central_widget.setLayout(self.general_layout)
        self.setCentralWidget(central_widget)

        for plot in self.plots:
            self.general_layout.addWidget(plot)

    def update_plots(self, data_handler):
        while True:
            data = data_handler.get_data()
            for i, plot in enumerate(self.plots):
                plot.update_plot(data[i])


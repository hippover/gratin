import socket
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)
from gratin.training.callbacks import LatentSpaceSaver, Plotter
from gratin.standard import train_model, load_model, plot_demo, get_predictions


if __name__ == "__main__":

    if "jbmasson" in socket.gethostname():
        export_path = "/home/hverdier/Gaia/hecat/hippo/models/andi_17may2022"
    else:
        export_path = "/Users/hverdier/models/demo"

    train_model(
        export_path=export_path,
        max_n_epochs=50,
        length_range=(7, 50),
        num_workers=4,
        time_delta_range=(0.005, 1.0),
    )

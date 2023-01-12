import socket
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)
from gratin.training.callbacks import LatentSpaceSaver, Plotter
from gratin.standard import train_model, load_model, plot_demo, get_predictions


if __name__ == "__main__":

    if "jbmasson" in socket.gethostname():
        export_path = "/home/hverdier/Gaia/hecat/hippo/models/gratin_20sept"
    else:
        export_path = "/Users/hverdier/models/demo"

    train_model(
        export_path=export_path,
        max_n_epochs=50,
        length_range=(7, 100),
        num_workers=4,
        time_delta_range=(0.015, 0.016),
        noise_range=(0.015, 0.12),
    )

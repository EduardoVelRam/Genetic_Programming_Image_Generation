import kagglehub
from pathlib import Path
import idx2numpy

mnist_path = Path(
    kagglehub.dataset_download("hojjatk/mnist-dataset")
)

def load_mnist(path):
    X_train = idx2numpy.convert_from_file(
        str(path / "train-images-idx3-ubyte" / "train-images-idx3-ubyte")
    )
    y_train = idx2numpy.convert_from_file(
        str(path / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte")
    )
    X_test = idx2numpy.convert_from_file(
        str(path / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte")
    )
    y_test = idx2numpy.convert_from_file(
        str(path / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte")
    )

    return X_train, y_train, X_test, y_test

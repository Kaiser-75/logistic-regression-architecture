import numpy as np

class MNISTLoader:
    def __init__(self, train_img_path, train_lbl_path, test_img_path, test_lbl_path):
        self.train_img_path = train_img_path
        self.train_lbl_path = train_lbl_path
        self.test_img_path = test_img_path
        self.test_lbl_path = test_lbl_path

    def _read_labels(self, path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            assert magic == 2049, "Invalid label file"
            num_items = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def _read_images(self, path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            assert magic == 2051, "Invalid image file"
            num_images = int.from_bytes(f.read(4), 'big')
            num_rows = int.from_bytes(f.read(4), 'big')
            num_cols = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape((num_images, num_rows * num_cols)).astype(np.float32) / 255.0
        return images

    def load_data(self):
        X_train = self._read_images(self.train_img_path)
        y_train = self._read_labels(self.train_lbl_path)
        X_test = self._read_images(self.test_img_path)
        y_test = self._read_labels(self.test_lbl_path)
        return X_train, y_train, X_test, y_test

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.experimental.numpy import deg2rad
from tensorflow.math import asin, cos, sin
from typing import Optional
import math


### Random Fourier Features ###
def gaussian_encoding(v: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    vp = 2 * math.pi * tf.matmul(v, tf.transpose(b))
    return tf.concat((tf.cos(vp), tf.sin(vp)), axis=-1)


class GaussianEncoding(layers.Layer):
    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[float] = None,
        encoded_size: Optional[float] = None,
        b: Optional[tf.Tensor] = None,
    ):
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError('Arguments "sigma," "input_size," and "encoded_size" are required.')
            b = tf.random.normal(shape=(encoded_size, input_size), mean=0.0, stddev=sigma)
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = self.add_weight(
            name="b", shape=(encoded_size, input_size), initializer=tf.keras.initializers.Constant(b), trainable=False
        )

    def call(self, v: tf.Tensor) -> tf.Tensor:
        return gaussian_encoding(v, self.b)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"b": self.b.numpy()})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 2 * self.b.shape[0])


# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def equal_earth_projection(L: tf.Tensor) -> tf.Tensor:
    lat = deg2rad(L[:, 0])
    lon = deg2rad(L[:, 1])
    theta = asin(math.sqrt(3) / 2 * sin(lat))

    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * math.sqrt(3) * lon * cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta

    return tf.stack((x, y), axis=1) * SF / 180


class LocationEncoderCapsule(layers.Layer):
    def __init__(self, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self.rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = tf.keras.Sequential(
            [
                self.rff_encoding,
                layers.Dense(1024, activation="relu", name="dense_1"),
                layers.Dense(1024, activation="relu", name="dense_2"),
                layers.Dense(1024, activation="relu", name="dense_3"),
                layers.Dense(512, name="dense_4"),
            ]
        )

    def call(self, x):
        x = self.capsule(x)
        # x = self.head(x)
        return x


class LocationEncoder(tf.keras.Model):
    def __init__(self, sigma=[2**0, 2**4, 2**8], from_pretrained=True):
        super().__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        self.loc_enc_layers = []
        for i, s in enumerate(self.sigma):
            self.loc_enc_layers.append(LocationEncoderCapsule(sigma=s, name=f"location_encoder_{i}"))

        self(tf.random.uniform((4, 2), minval=-180, maxval=180))

    def call(self, location):
        location = equal_earth_projection(location)
        location_features = self.loc_enc_layers[0](location)

        for i in range(1, self.n):
            location_features += self.loc_enc_layers[i](location)

        return location_features

    def build_graph(self):
        x = tf.keras.Input(shape=(2,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    location_encoder = LocationEncoder()
    random_locations = tf.random.uniform((32, 2), minval=-180, maxval=180)
    location_features = location_encoder(random_locations)
    print(location_features.shape)

    import torch

    path = "models/location_encoder_weights.pth"
    state_dict = torch.load(path)

    wmap = {
        (0, 0): "b:0",
        (1, 0): "dense_1/bias:0",
        (1, 1): "dense_1/kernel:0",
        (2, 0): "dense_2/bias:0",
        (2, 1): "dense_2/kernel:0",
        (3, 0): "dense_3/bias:0",
        (3, 1): "dense_3/kernel:0",
        (4, 0): "dense_4/bias:0",
        (4, 1): "dense_4/kernel:0",
    }

    for key in state_dict.keys():
        i = int(key.split(".")[0][-1])
        j = int(key.split(".")[2])
        j = int(j / 2 + 0.5)
        if "head" in key:
            j = 4
        k = 1 if "weight" in key else 0

        for w in location_encoder.loc_enc_layers[i].weights:
            if w.name == wmap[(j, k)]:
                # check if the shapes match else transpose
                if w.shape == state_dict[key].shape:
                    w.assign(state_dict[key])
                else:
                    w.assign(tf.transpose(state_dict[key]))
                pass

    # Save the weights
    location_encoder.save_weights("models/location_encoder_weights.h5")

    # Reinitialize and load the weights
    location_encoder = LocationEncoder()
    location_encoder.load_weights("models/location_encoder_weights.h5")

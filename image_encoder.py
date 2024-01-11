# Tensorflow imports
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

# ViT-Keras: Library that implements Vision Transformer on Keras
from vit_keras.layers import MultiHeadSelfAttention
from vit_keras.layers import TransformerBlock


@tf.keras.utils.register_keras_serializable()
class IA3MultiHeadSelfAttention(MultiHeadSelfAttention):
    """
    A MultiHeadSelfAttention layer with IA3 vectors for scaling key and value features.

    Added Functionalies
    -------------------
    - `copy_weights_from`: Copy weights from a MultiHeadSelfAttention layer
    - `extend`: Add :math:`l_k` and :math:`l_v` vectors to scale key and value features
    - `call`: Change the forward pass to include :math:`l_k` and :math:`l_v` vectors
    """

    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, num_heads=num_heads, **kwargs)
        self.is_extended = False
        # Forward pass to initialize weights
        self(tf.random.normal((1, 196, 768)))

    def copy_weights_from(self, model):
        for attr in ["query_dense", "key_dense", "value_dense", "combine_heads"]:
            layer = getattr(self, attr)
            layer.set_weights(getattr(model, attr).get_weights())

    def extend(self, model: MultiHeadSelfAttention):
        self.is_extended = True

        # Copy Attributes
        self.hidden_size = model.hidden_size
        self.projection_dim = model.projection_dim

        # Parameters
        self.copy_weights_from(model)

        # IA3, TODO: change initializer
        self.lk = self.add_weight("lk", shape=(self.hidden_size,), initializer="ones")
        self.lv = self.add_weight("lv", shape=(self.hidden_size,), initializer="ones")

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # IA3: Scale key and value features
        if self.is_extended:
            key = tf.multiply(key, self.lk)
            value = tf.multiply(value, self.lv)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights


class ElementWiseMultiplyLayer(tf.keras.layers.Layer):
    """Layer that performs element-wise multiplication with a learnable vector."""

    def __init__(self, dim, **kwargs):
        super(ElementWiseMultiplyLayer, self).__init__(**kwargs)
        self.lff = self.add_weight("lff", shape=(dim,), initializer="ones")

    def call(self, inputs):
        return tf.multiply(inputs, self.lff)


@tf.keras.utils.register_keras_serializable()
class IA3TransformerBlock(TransformerBlock):
    """
    A TransformerBlock layer with IA3 vectors for scaling key and value features.

    Added Functionalies
    -------------------
    - `copy_weights_from`: Copy weights from a TransformerBlock layer
    - `extend`: Add :math:`l_ff` inside the MLP block and replace the MultiHeadSelfAttention layer with IA3MultiHeadSelfAttention
    that adds :math:`l_k` and :math:`l_v` vectors
    """

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, **kwargs)
        self.is_extended = False
        self(tf.random.normal((1, 196, 768)))

    def copy_weights_from(self, model):
        for attr in ["att", "mlpblock", "layernorm1", "layernorm2", "dropout_layer"]:
            layer = getattr(self, attr)
            layer.set_weights(getattr(model, attr).get_weights())

    def extend(self, model: TransformerBlock):
        self.is_extended = True

        # Attributes
        self.num_heads = model.num_heads
        self.mlp_dim = model.mlp_dim
        self.dropout = model.dropout

        # Parameters
        self.copy_weights_from(model)

        # IA3, Attention
        ia3mhsa = IA3MultiHeadSelfAttention(num_heads=self.num_heads)
        ia3mhsa.extend(self.att)
        self.att = ia3mhsa

        # IA3 MLP Block
        mlplayers = []
        for l in self.mlpblock.layers:
            mlplayers.append(l)
            # if l.name == "lambda":
            if l.name.startswith("lambda"):
                mlplayers.append(ElementWiseMultiplyLayer(self.mlp_dim))

        self.mlpblock = tf.keras.Sequential(mlplayers, name="MlpBlock_3")


def build_ia3_model(model, add_aligment_layer=False):
    inputs = Input(shape=(384, 384, 3))
    x = model.layers[0](inputs)
    for layer in model.layers[1:]:
        if layer.name.startswith("Transformer/encoderblock"):
            tb = layer
            ia3tb = IA3TransformerBlock(num_heads=tb.num_heads, mlp_dim=tb.mlp_dim, dropout=tb.dropout)
            ia3tb.extend(tb)
            x, _ = ia3tb(x)
        else:
            x = layer(x)

    if add_aligment_layer:
        # Align to 512 dimensions
        x = tf.keras.layers.Dense(512, name="geoaligner")(x)

    return Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    from vit_keras import vit

    # The initial vectors are ones so we can test if everything is working
    # by checking if the output is the same
    image_size = 384
    model = vit.vit_b16(image_size=image_size, include_top=False, pretrained_top=False)
    new_model = build_ia3_model(model)

    # Dry run to see if it works
    x = tf.random.normal((1, 384, 384, 3))
    out1 = model(x)
    out2 = new_model(x)

    # Check if the output is the same
    print("Is the output the same?", tf.reduce_all(tf.equal(out1, out2)))

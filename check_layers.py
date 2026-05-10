import tensorflow as tf
import numpy as np

m          = tf.keras.models.load_model("model/aquatic_model_v2.keras")
base_model = m.get_layer("mobilenetv2_1.00_224")

# Run input through base model up to out_relu using a submodel
grad_model = tf.keras.models.Model(
    inputs=base_model.inputs,
    outputs=[
        base_model.get_layer("out_relu").output,
        base_model.output
    ]
)

dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)

with tf.GradientTape() as tape:
    conv_outputs, base_out = grad_model(dummy)
    # Now pass base_out through the rest of the sequential model
    x = m.get_layer("global_average_pooling2d_1")(conv_outputs)
    x = m.get_layer("dense_2")(x)
    x = m.get_layer("dropout_1")(x, training=False)
    predictions = m.get_layer("dense_3")(x)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)
print("grads:", grads.shape if grads is not None else "None")
print("conv_outputs:", conv_outputs.shape)
print("SUCCESS")
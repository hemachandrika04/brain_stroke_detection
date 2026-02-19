from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os


class PatchEmbedding(tf.keras.layers.Layer):
    """
    Patch embedding layer compatible with the saved model.
    It must accept the same init args used during training
    (e.g. PatchEmbedding(patch_size=2, embed_dim=128, ...)).
    """

    def __init__(self, patch_size=16, embed_dim=768, **kwargs):
        super().__init__(**kwargs)

        # Allow both int and tuple for patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Dense projection will be built once we know input shape
        self.projection = None

    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        patch_h, patch_w = self.patch_size
        channels = int(input_shape[-1])
        patch_dim = patch_h * patch_w * channels

        # Kernel shape becomes (patch_dim, embed_dim), e.g. (512, 128),
        # which matches the stored weights.
        self.projection = tf.keras.layers.Dense(self.embed_dim)
        self.projection.build((None, patch_dim))

        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, height, width, channels)
        patch_h, patch_w = self.patch_size

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, patch_h, patch_w, 1],
            strides=[1, patch_h, patch_w, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # patches: (batch, num_patches_h, num_patches_w, patch_dim)
        batch_size = tf.shape(patches)[0]
        num_patches_h = tf.shape(patches)[1]
        num_patches_w = tf.shape(patches)[2]

        patches = tf.reshape(
            patches, [batch_size, num_patches_h * num_patches_w, -1]
        )  # (batch, num_patches, patch_dim)

        x = self.projection(patches)  # (batch, num_patches, embed_dim)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


app = Flask(__name__, template_folder='../templates', static_folder='../static')

UPLOAD_FOLDER = '../static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the correct model file from the backend folder.
# Note: the actual filename has a space after the underscore.
MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'brain_stroke_ detection_model.h5')
model = tf.keras.models.load_model(
    MODEL_FILENAME, custom_objects={"PatchEmbedding": PatchEmbedding}
)

IMG_SIZE = 224

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    height, width, _ = image.shape

    img = preprocess_image(file_path)
    prediction = model.predict(img)

    result = "Stroke Detected" if prediction[0][0] > 0.5 else "No Stroke Detected"

    return jsonify({
        "result": result,
        "confidence": float(prediction[0][0]),
        "image_path": file_path.replace('../', ''),
        "file_name": file.filename,
        "dimensions": f"{width} × {height}",
        "file_size": round(os.path.getsize(file_path) / 1024, 2)
    })

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

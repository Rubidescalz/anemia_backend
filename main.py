from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown

app = FastAPI()

# =============================
# CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Directorio para modelos
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# DESCARGA Y CARGA DE MODELOS
# =============================

# EfficientNetB0
EFFNET_PATH = os.path.join(MODEL_DIR, "EfficientNetB0.h5")
FILE_ID_EFFNET = "1cgx38zVfWJIKCmHFa2ento6SVa5PVDk2"

if not os.path.exists(EFFNET_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID_EFFNET}"
    gdown.download(url, EFFNET_PATH, quiet=False)

effnet_model = load_model(EFFNET_PATH)

# UNet Conjuntiva
UNET_PATH = os.path.join(MODEL_DIR, "unet_conjuntiva_palpebral.h5")
FILE_ID_UNET = "1vYLwAIxeW-76_Z9MP5QFlHapw8xHKueI"

if not os.path.exists(UNET_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID_UNET}"
    gdown.download(url, UNET_PATH, quiet=False)

unet_model = load_model(UNET_PATH)

# =============================
# CONSTANTES
# =============================
IMG_SIZE = 256
FINAL_SIZE = 224
EFFNET_SIZE = 150

# =============================
# Función de segmentación
# =============================
def process_image_no_background(image_array, model):
    h, w = image_array.shape[:2]
    img_resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_input)[0]
    pred_mask = (pred > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_contour)

    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w_box = min(w - x, w_box + 2 * padding)
    h_box = min(h - y, h_box + 2 * padding)

    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    mask_bool = mask_resized.astype(bool)
    rgba_image[mask_bool, :3] = image_array[mask_bool]
    rgba_image[mask_bool, 3] = 255

    cropped = rgba_image[y:y+h_box, x:x+w_box]

    size = max(h_box, w_box)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    y_offset = (size - cropped.shape[0]) // 2
    x_offset = (size - cropped.shape[1]) // 2
    canvas[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped

    final_img = cv2.resize(canvas, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_AREA)
    return final_img, mask_resized

# =============================
# Función Grad-CAM
# =============================
def get_grad_cam(model, img_array, layer_name='block7a_project_conv'):
    img_array = cv2.resize(img_array[:, :, :3], (EFFNET_SIZE, EFFNET_SIZE))
    img_array = np.expand_dims(preprocess_input(img_array), axis=0)

    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), int(class_idx)

# =============================
# ENDPOINT /predict
# =============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            return {"error": "Imagen inválida"}

        segmented_img, mask = process_image_no_background(img_array, unet_model)
        if segmented_img is None:
            return {"error": "No se detectó conjuntiva"}

        effnet_img = cv2.resize(segmented_img[:, :, :3], (EFFNET_SIZE, EFFNET_SIZE))
        effnet_img = np.expand_dims(preprocess_input(effnet_img), axis=0)

        prediction = effnet_model.predict(effnet_img)[0]
        anemia = "CON ANEMIA" if prediction[0] > 0.5 else "SIN ANEMIA"

        heatmap, class_idx = get_grad_cam(effnet_model, effnet_img[0])

        return {
            "classification": prediction.tolist(),
            "anemia": anemia,
            "confidence": float(prediction[class_idx]),
            "heatmap": heatmap.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

# =============================
# Run local (solo desarrollo)
# =============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

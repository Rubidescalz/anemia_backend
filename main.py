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
# Directorios
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =============================
# MODELOS (paths + Drive IDs)
# =============================
EFFNET_PATH = os.path.join(MODEL_DIR, "EfficientNetB0.h5")
UNET_PATH = os.path.join(MODEL_DIR, "unet_conjuntiva_palpebral.h5")

FILE_ID_EFFNET = "1cgx38zVfWJIKCmHFa2ento6SVa5PVDk2"
FILE_ID_UNET = "1vYLwAIxeW-76_Z9MP5QFlHapw8xHKueI"

# =============================
# Descargar modelos si no existen
# =============================
if not os.path.exists(EFFNET_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID_EFFNET}", EFFNET_PATH, quiet=False)

if not os.path.exists(UNET_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID_UNET}", UNET_PATH, quiet=False)

# =============================
# Lazy loading (CLAVE PARA RENDER)
# =============================
effnet_model = None
unet_model = None

def get_effnet():
    global effnet_model
    if effnet_model is None:
        print("📦 Cargando EfficientNet...")
        effnet_model = load_model(EFFNET_PATH)
    return effnet_model

def get_unet():
    global unet_model
    if unet_model is None:
        print("📦 Cargando UNet...")
        unet_model = load_model(UNET_PATH)
    return unet_model

# =============================
# CONSTANTES
# =============================
IMG_SIZE = 256
FINAL_SIZE = 224
EFFNET_SIZE = 150

# =============================
# Segmentación
# =============================
def process_image_no_background(image_array, model):
    h, w = image_array.shape[:2]

    img_resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_input, verbose=0)[0]
    mask = (pred > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    cropped = image_array[y:y+bh, x:x+bw]
    final_img = cv2.resize(cropped, (FINAL_SIZE, FINAL_SIZE))

    return final_img, mask_resized

# =============================
# Grad-CAM
# =============================
def get_grad_cam(model, img_array, layer_name="block7a_project_conv"):
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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    return heatmap.numpy(), int(class_idx)

# =============================
# ENDPOINT
# =============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Imagen inválida"}

        unet = get_unet()
        effnet = get_effnet()

        segmented, _ = process_image_no_background(img, unet)
        if segmented is None:
            return {"error": "No se detectó conjuntiva"}

        img_eff = cv2.resize(segmented, (EFFNET_SIZE, EFFNET_SIZE))
        img_eff = np.expand_dims(preprocess_input(img_eff), axis=0)

        prediction = effnet.predict(img_eff, verbose=0)[0]
        anemia = "CON ANEMIA" if prediction[0] > 0.5 else "SIN ANEMIA"

        heatmap, class_idx = get_grad_cam(effnet, img_eff[0])

        return {
            "classification": prediction.tolist(),
            "anemia": anemia,
            "confidence": float(prediction[class_idx]),
            "heatmap": heatmap.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

# =============================
# Render
# =============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================= MODEL =================
model = load_model("model/keras_model.h5")

# ================= KAMERA =================
kamera = cv2.VideoCapture(0)

while True:
    ret, frame = kamera.read()
    if not ret:
        break

    # BGR → RGB
    goruntu = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Yeniden boyutlandır
    goruntu = cv2.resize(goruntu, (224, 224))
    goruntu = (goruntu.astype(np.float32) / 127.0) - 1
    goruntu = np.expand_dims(goruntu, axis=0)

    # Tahmin
    tahmin = model.predict(goruntu, verbose=0)
    if tahmin[0][0] > tahmin[0][1]:
        etiket = "Maskeli"
        renk = (0, 255, 0)
    else:
        etiket = "Maskesiz"
        renk = (0, 0, 255)

    # Ekrana yaz
    cv2.putText(frame, etiket, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)

    # Görüntüyü göster
    cv2.imshow("Maske Tespit", frame)

    # q tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
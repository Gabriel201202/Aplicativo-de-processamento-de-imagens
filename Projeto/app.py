from flask import Flask, render_template, request
import cv2
import numpy as np
from scipy import stats

app = Flask(__name__)

INPUT_IMAGE = "static/pessoa.jpg"
OUTPUT_IMAGE = "static/output.jpg"


def aplicar_filtro(filtro, k):
    img = cv2.imread(INPUT_IMAGE)

    # Kernel deve ser ímpar
    if k % 2 == 0:
        k += 1

    if filtro == "media":
        img_filtrada = cv2.blur(img, (k, k))

    elif filtro == "gauss":
        img_filtrada = cv2.GaussianBlur(img, (k, k), 0)

    elif filtro == "mediana":
        img_filtrada = cv2.medianBlur(img, k)

    elif filtro == "moda":
        img_filtrada = img.copy()
        h, w, c = img.shape
        
        offset = k // 2

        for y in range(offset, h - offset):
            for x in range(offset, w - offset):
                for ch in range(c):
                    janela = img[y-offset:y+offset+1, x-offset:x+offset+1, ch].flatten()
                    moda = stats.mode(janela, keepdims=True)[0][0]
                    img_filtrada[y, x, ch] = moda

    else:
        img_filtrada = img

    cv2.imwrite(OUTPUT_IMAGE, img_filtrada)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/processar", methods=["POST"])
def processar():
    filtro = request.form.get("filtro")
    kernel = int(request.form.get("kernel"))

    aplicar_filtro(filtro, kernel)

    return render_template("index.html", filtro=filtro, kernel=kernel)


if __name__ == "__main__":
    app.run(debug=True)

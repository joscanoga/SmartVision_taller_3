

from PIL import Image
import numpy as np

## funcion de normalizacion de tamaño con padding
def normalizar_tamano(imagen, tamano=(512, 512)):
    ancho_original, alto_original = imagen.size
    ancho_objetivo, alto_objetivo = tamano

    # Calcular la relación de aspecto
    relacion_aspecto = min(ancho_objetivo / ancho_original, alto_objetivo / alto_original)

    # Calcular el nuevo tamaño manteniendo la relación de aspecto
    nuevo_ancho = int(ancho_original * relacion_aspecto)
    nuevo_alto = int(alto_original * relacion_aspecto)

    # Redimensionar la imagen
    # Seleccionar filtro de remuestreo compatible con distintas versiones de Pillow
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # Pillow < 9.1 expone LANCZOS directamente en Image
        resample_filter = getattr(Image, 'LANCZOS', Image.BICUBIC)

    imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), resample=resample_filter)

    # Crear una nueva imagen con el tamaño objetivo y rellenar con negro
    imagen_normalizada = Image.new("RGB", tamano)
    imagen_normalizada.paste(imagen_redimensionada, ((ancho_objetivo - nuevo_ancho) // 2, (alto_objetivo - nuevo_alto) // 2))

    return imagen_normalizada

def normalizacion_min_max(imagen):
    # Convertir la imagen a un array numpy
    img_array = np.array(imagen).astype(np.float32)

    # Normalizar los valores de píxeles a [0, 1]
    img_min = img_array.min()
    img_max = img_array.max()
    img_normalizada = (img_array - img_min) / (img_max - img_min)

    # Escalar de vuelta a [0, 255] y convertir a uint8
    img_normalizada = (img_normalizada * 255).astype(np.uint8)

    # Convertir de vuelta a imagen PIL
    imagen_normalizada = Image.fromarray(img_normalizada)

    return imagen_normalizada

def preprocesar_imagen(ruta):
    with Image.open(ruta) as img:
        img = img.convert("RGB")  # Asegurarse de que la imagen esté en modo RGB
        img_normalizada = normalizar_tamano(img)
        img_normalizada = normalizacion_min_max(img_normalizada)
        return img_normalizada
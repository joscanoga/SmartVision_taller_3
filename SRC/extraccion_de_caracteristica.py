# clase para manejar el extractor de caracteristicas con restnet50
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm # Para una barra de progreso

class RXFeatureExtractor:
    """
    Clase para cargar un modelo preentrenado (ResNet50), congelar sus pesos
    y extraer vectores de características (features) de imágenes RX.
    """
    def __init__(self, target_size=(512, 512), model_name='ResNet50'):
        # 1. Parámetros de inicialización
        self.target_size = target_size
        self.model_name = model_name
        self.feature_extractor = self._load_model()
        
    def _load_model(self):
        # 2. Carga y Congelamiento del Modelo Base
        if self.model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False, 
                input_shape=(self.target_size[0], self.target_size[1], 3)
            )
        elif self.model_name == 'EfficientNetB0':
             base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False, 
                input_shape=(self.target_size[0], self.target_size[1], 3)
            )
        else:
            raise ValueError(f"Modelo '{self.model_name}' no soportado.")
            
        base_model.trainable = False
        
        # 3. Aplicar Pooling Global (opcional pero recomendado para feature vectors)
        # Esto reduce el tensor final a un vector plano.
        extractor = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        print(f"Extractor {self.model_name} cargado con éxito.")
        return extractor
        
    def _preprocess_image(self, img_path):
        # 4. Preprocesamiento: Redimensionamiento y Estandarización
        try:
            # Cargar y convertir a RGB (necesario para ImageNet weights)
            img = Image.open(img_path).convert('RGB')
            # Redimensionar (aquí se manejaría el padding si fuera necesario,
            # pero para simplificar usamos resize directo)
            img = img.resize(self.target_size) 
            
            # Convertir a array de numpy y expandir las dimensiones para el batch (1, H, W, C)
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            # Estandarización según el modelo base (ImageNet)
            if self.model_name == 'ResNet50':
                return tf.keras.applications.resnet50.preprocess_input(img_array)
            elif self.model_name == 'EfficientNetB0':
                return tf.keras.applications.efficientnet.preprocess_input(img_array)
            
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
            return None

    def extract_features(self, data_directory):
        """
        Extrae características y etiquetas de todas las imágenes en el directorio.
        
        data_directory debe contener subcarpetas con los nombres de las clases (ej: 'Normal', 'Anormal').
        """
        all_features = []
        all_labels = []
        
        # Obtener los nombres de las clases (carpetas)
        class_names = sorted(os.listdir(data_directory))
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        # Iterar sobre las clases y archivos
        for class_name, label in class_to_idx.items():
            class_path = os.path.join(data_directory, class_name)
            if not os.path.isdir(class_path):
                continue
                
            print(f"Iniciando extracción para la clase: {class_name} ({label})...")
            
            # Listar imágenes y usar tqdm para la barra de progreso
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
            
            for filename in tqdm(image_files):
                img_path = os.path.join(class_path, filename)
                
                processed_img = self._preprocess_image(img_path)
                
                if processed_img is not None:
                    # Extraer el vector de características
                    features = self.feature_extractor.predict(processed_img, verbose=0)
                    
                    all_features.append(features[0])
                    all_labels.append(label)

        # Convertir listas a arrays de NumPy
        return np.array(all_features), np.array(all_labels)




# Punto 2 A 3 Descriptores de contorno

def extraer_descriptores_contorno(img_gray):
    """
    Calcula Área, Perímetro, Circularidad y Excentricidad.
    Requiere una imagen en escala de grises.
    """
    # 1. Segmentación simple (Binarización)
    # Usamos Otsu para encontrar el umbral automático que separa cuerpo de fondo
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si no encuentra contornos, retornamos ceros
    if not contours:
        return [0, 0, 0, 0]
    
    # 3. Nos quedamos con el contorno más grande (asumimos que es el tórax/pulmón)
    c = max(contours, key=cv2.contourArea)
    
    # --- CALCULO DE MÉTRICAS ---
    
    # A. Área
    area = cv2.contourArea(c)
    
    # B. Perímetro
    perimeter = cv2.arcLength(c, True)
    
    # C. Circularidad: (4 * pi * Area) / (Perímetro^2)
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # D. Excentricidad
    # Ajustamos una elipse al contorno para medir sus ejes
    if len(c) < 5: # Se necesitan al menos 5 puntos para ajustar una elipse
        eccentricity = 0
    else:
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        # MA = Eje Menor, ma = Eje Mayor
        a = ma / 2
        b = MA / 2
        if a > 0:
            eccentricity = np.sqrt(1 - (b**2 / a**2))
        else:
            eccentricity = 0
            
    # Retornamos las 4 características en una lista
    return [area, perimeter, circularity, eccentricity]

# Punto 2 B 3: Filtros de Gabor

def extraer_descriptores_gabor(img_gray):
    """
    Aplica un banco de filtros de Gabor y retorna media y varianza de la respuesta.
    """
    filters = []
    ksize = 31  # Tamaño del kernel
    
    # Definimos parámetros del banco de filtros
    # Variamos la orientación (theta) y la frecuencia (sigma/lambd)
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 grados
    sigmas = [1, 3] # Diferentes escalas
    
    features = []
    
    # Creamos y aplicamos cada filtro
    for theta in thetas:
        for sigma in sigmas:
            lambd = np.pi/4
            gamma = 0.5
            psi = 0
            
            # Crear el kernel de Gabor
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            
            # Filtrar la imagen
            fimg = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
            
            # Calcular estadísticas de respuesta (Media y Desviación Estándar)
            mean = np.mean(fimg)
            std = np.std(fimg)
            
            # Guardamos ambas estadísticas
            features.extend([mean, std])
            
    return features
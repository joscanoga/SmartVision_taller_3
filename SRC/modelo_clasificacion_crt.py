import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

class CustomClassifier:
    """
    Clase para construir y gestionar el modelo clasificador personalizado 
    que se entrena sobre las características extraídas (Transfer Learning).
    """
    def __init__(self, input_shape=2048):
        """
        Inicializa la clase y construye el modelo.
        :param input_shape: Dimensión del vector de características de entrada (por defecto 2048 para ResNet50).
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Define y construye la arquitectura del clasificador.
        """
        model = Sequential([
            # 1. Capa de entrada (directamente el vector de 2048)
            Input(shape=(self.input_shape,)),
            
            # 2. Primera capa densa (Aprendizaje complejo)
            Dense(512, activation='relu'),
            
            # 3. Regularización para evitar sobreajuste
            Dropout(0.5), 
            
            # 4. Capa intermedia para refinamiento
            Dense(128, activation='relu'),

            # 5. Capa de salida binaria (Normal/Anormal)
            Dense(1, activation='sigmoid')
        ], name="RX_Custom_Classifier")
        
        return model

    def compile_model(self, learning_rate=0.001):
        """
        Compila el modelo con optimizador, pérdida y métricas adecuadas para el desbalance.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
    def show_architecture(self):
        """
        Muestra un resumen de la arquitectura del modelo.
        """
        if self.model:
            print("--- Resumen del Modelo Clasificador Personalizado ---")
            self.model.summary()
        else:
            print("El modelo no ha sido construido aún.")


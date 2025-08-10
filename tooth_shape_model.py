from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

class ToothShapeModel:
    """
    Implementación mejorada de un Active Shape Model (ASM) para segmentación dental.
    """
    def __init__(self, n_landmarks=40):
        self.n_landmarks = n_landmarks  # Número de puntos de referencia
        self.mean_shape = None          # Forma promedio
        self.eigenvectors = None        # Modos de variación
        self.eigenvalues = None         # Valores propios
        self.pca = None                 # Modelo PCA
        self.aligned_shapes = []        # Formas alineadas usadas durante el entrenamiento
        
    def _normalize_shape(self, shape):
        """Normaliza la forma centrándola y escalándola"""
        centroid = np.mean(shape, axis=0)
        centered = shape - centroid
        scale = np.sqrt(np.sum(centered**2)) / self.n_landmarks
        normalized = centered / scale if scale > 0 else centered
        return normalized, centroid, scale
    
    def _procrustes(self, shape, reference):
        """
        Análisis Procrustes mejorado: alinea shape a reference
        retornando la forma transformada
        """
        # Centramos ambas formas
        shape_norm, shape_centroid, shape_scale = self._normalize_shape(shape)
        ref_norm, ref_centroid, ref_scale = self._normalize_shape(reference)
        
        # Calculamos la matriz de covarianza
        cov = np.dot(ref_norm.T, shape_norm)
        
        # Descomposición SVD para encontrar la rotación óptima
        u, s, vt = np.linalg.svd(cov)
        
        # Verificamos si necesitamos una reflexión (det<0)
        reflection = np.eye(2)
        if np.linalg.det(np.dot(u, vt)) < 0:
            reflection[1, 1] = -1
            
        # Matriz de rotación final
        rotation = np.dot(np.dot(u, reflection), vt)
        
        # Calculamos el factor de escala óptimo
        scale_factor = np.sum(s * np.diag(reflection)) / np.sum(np.square(shape_norm))
        
        # Aplicamos la transformación completa: escala, rotación y traslación
        aligned = scale_factor * np.dot(shape_norm, rotation.T)
        
        # Calculamos el error residual como medida de similitud
        residual = np.sum(np.square(ref_norm - aligned))
        
        return aligned, residual
    
    def _align_shapes(self, shapes):
        """
        Alinea todas las formas mediante análisis Procrustes generalizado.
        Más robusto que la implementación anterior.
        """
        n_shapes = len(shapes)
        if n_shapes < 2:
            return shapes
            
        # Inicializamos con formas normalizadas
        aligned = []
        for shape in shapes:
            norm_shape, _, _ = self._normalize_shape(shape)
            aligned.append(norm_shape)
            
        prev_mean = np.zeros_like(aligned[0])
        mean_shape = np.mean(aligned, axis=0)
        
        # Iteramos hasta convergencia
        max_iterations = 50
        convergence_threshold = 1e-6
        
        for iteration in range(max_iterations):
            # Guardamos la media anterior para verificar convergencia
            prev_mean = mean_shape.copy()
            
            # Alineamos cada forma con la forma media
            for i in range(n_shapes):
                aligned[i], _ = self._procrustes(aligned[i], mean_shape)
            
            # Recalculamos la forma media
            mean_shape = np.mean(aligned, axis=0)
            
            # Normalizamos la forma media
            mean_shape, _, _ = self._normalize_shape(mean_shape)
            
            # Verificamos convergencia
            if np.sum((mean_shape - prev_mean) ** 2) < convergence_threshold:
                print(f"Alineación convergida en {iteration+1} iteraciones")
                break
                
        return aligned
    
    def _ensure_consistent_orientation(self, shapes):
        """
        Asegura que todos los contornos tengan la misma orientación
        basándose en el área (regla de la mano derecha)
        """
        consistent_shapes = []
        
        for shape in shapes:
            # Calculamos el área usando la fórmula del producto cruzado
            area = 0
            for i in range(len(shape)):
                j = (i + 1) % len(shape)
                area += shape[i, 0] * shape[j, 1]
                area -= shape[j, 0] * shape[i, 1]
            area /= 2
            
            # Si el área es negativa, invertimos el orden de los puntos
            if area < 0:
                shape = np.flipud(shape)
                
            consistent_shapes.append(shape)
            
        return consistent_shapes
        
    def _resample_shape(self, shape, n_points=None):
        """
        Remuestrea la forma para tener un número específico de puntos
        distribuidos uniformemente a lo largo del contorno
        """
        if n_points is None:
            n_points = self.n_landmarks
            
        # Calculamos la longitud total del contorno
        perimeter = 0
        for i in range(len(shape)):
            j = (i + 1) % len(shape)
            segment = np.linalg.norm(shape[i] - shape[j])
            perimeter += segment
            
        # Remuestreamos los puntos a distancias iguales
        resampled = np.zeros((n_points, 2))
        point_dist = perimeter / n_points
        
        curr_point = 0
        curr_dist = 0
        
        for i in range(n_points):
            target_dist = i * point_dist
            
            # Avanzamos hasta encontrar el segmento correcto
            while curr_dist + np.linalg.norm(shape[curr_point] - shape[(curr_point + 1) % len(shape)]) < target_dist:
                curr_dist += np.linalg.norm(shape[curr_point] - shape[(curr_point + 1) % len(shape)])
                curr_point = (curr_point + 1) % len(shape)
                
            # Interpolamos para encontrar el punto exacto
            next_point = (curr_point + 1) % len(shape)
            segment_length = np.linalg.norm(shape[curr_point] - shape[next_point])
            
            if segment_length > 0:
                fraction = (target_dist - curr_dist) / segment_length
                resampled[i] = shape[curr_point] + fraction * (shape[next_point] - shape[curr_point])
            else:
                resampled[i] = shape[curr_point]
                
        return resampled
        
    def train(self, training_shapes, align_rotation=True, resample=True):
        """
        Entrena el modelo usando un conjunto de formas anotadas
        
        Args:
            training_shapes: Lista de arrays numpy, cada uno con forma (n_points, 2)
                           representando los contornos de los dientes
            align_rotation: Si es True, alinea las rotaciones de las formas
            resample: Si es True, remuestrea los puntos para distribuirlos uniformemente
        """
        if not training_shapes:
            raise ValueError("Se necesitan formas de entrenamiento")
        
        preprocessed_shapes = []
        
        # Preprocesamiento de formas
        for shape in training_shapes:
            processed = shape.copy()
            
            # Remuestrear si es necesario
            if resample:
                processed = self._resample_shape(processed)
            
            preprocessed_shapes.append(processed)
            
        # Asegurar orientación consistente
        preprocessed_shapes = self._ensure_consistent_orientation(preprocessed_shapes)
        
        # Alinear todas las formas
        if align_rotation:
            self.aligned_shapes = self._align_shapes(preprocessed_shapes)
        else:
            self.aligned_shapes = preprocessed_shapes
        
        # Calcular la forma media
        self.mean_shape = np.mean(self.aligned_shapes, axis=0)
        
        # Crear matriz de formas para PCA
        shape_matrix = np.vstack([shape.flatten() for shape in self.aligned_shapes])
        
        # Realizar PCA para extraer modos de variación
        n_components = min(len(self.aligned_shapes)-1, self.n_landmarks*2-1)
        
        if n_components < 1:
            raise ValueError("No hay suficientes formas para extraer componentes principales")
            
        self.pca = PCA(n_components=n_components)
        self.pca.fit(shape_matrix)
        
        # Guardar eigenvectores y eigenvalores
        self.eigenvectors = self.pca.components_
        self.eigenvalues = self.pca.explained_variance_
        
        print(f"Modelo entrenado con {len(training_shapes)} formas.")
        var_explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"Varianza explicada: {var_explained:.2f}")
        
        if var_explained < 0.9:
            print("ADVERTENCIA: El modelo explica menos del 90% de la varianza.")
            print("Considere incluir más ejemplos de entrenamiento o revisar la consistencia de las anotaciones.")
     
        self.visualize_model_variations(n_modes=3, scale_factor=3.0)
           
        return self
    
    def visualize_model_variations(self, n_modes=3, scale_factor=3.0):
        """
        Visualiza la forma media y los principales modos de variación
        
        Args:
            n_modes: Número de modos principales a visualizar
            scale_factor: Factor de escala para la visualización de variaciones
        """
        if self.mean_shape is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        n_modes = min(n_modes, len(self.eigenvalues))
        
        plt.figure(figsize=(10, 4 * (n_modes+1)))
        
        # Visualizar forma media
        plt.subplot(n_modes+1, 1, 1)
        x = self.mean_shape[:, 0]
        y = self.mean_shape[:, 1]
        plt.plot(x, y, 'b-', linewidth=2)
        plt.plot(x, y, 'bo', markersize=4)
        
        # Conectar el último punto con el primero para cerrar el contorno
        plt.plot([x[-1], x[0]], [y[-1], y[0]], 'b-', linewidth=2)
        
        plt.title("Forma Media")
        plt.axis('equal')
        plt.grid(True)
        
        # Añadir marcador especial para el punto inicial
        plt.plot(x[0], y[0], 'rs', markersize=8)
        
        # Visualizar modos de variación
        for i in range(n_modes):
            plt.subplot(n_modes+1, 1, i+2)
            
            # Obtener eigenvector y eigenvalue
            eigenvector = self.eigenvectors[i].reshape(-1, 2)
            eigenvalue = self.eigenvalues[i]
            std_dev = np.sqrt(eigenvalue)
            
            # Visualizar variaciones: -3*std, media, +3*std
            variations = []
            for j, factor in enumerate([-scale_factor, 0, scale_factor]):
                variation = self.mean_shape + factor * std_dev * eigenvector
                variations.append(variation)
                
                color = ['r', 'b', 'g'][j]
                x = variation[:, 0]
                y = variation[:, 1]
                
                plt.plot(x, y, f'{color}-', linewidth=2, 
                         label=f"{factor:+.1f} σ" if factor != 0 else "Media")
                plt.plot(x, y, f'{color}o', markersize=4)
                
                # Cerrar el contorno
                plt.plot([x[-1], x[0]], [y[-1], y[0]], f'{color}-', linewidth=2)
                
                # Marcar el punto inicial
                plt.plot(x[0], y[0], f'{color}s', markersize=8)
            
            var_ratio = 100 * self.pca.explained_variance_ratio_[i]
            plt.title(f"Modo de Variación {i+1} ({var_ratio:.1f}% de varianza)")
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def reconstruct_shape(self, weights):
        """
        Reconstruye una forma usando los pesos de los modos de variación
        
        Args:
            weights: Array de pesos para cada modo de variación
        
        Returns:
            Forma reconstruida
        """
        if self.mean_shape is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        # Aseguramos que weights tenga la longitud correcta
        n_modes = len(self.eigenvalues)
        if len(weights) > n_modes:
            weights = weights[:n_modes]
        elif len(weights) < n_modes:
            weights = np.pad(weights, (0, n_modes - len(weights)))
        
        # Limitamos los pesos a un rango razonable (típicamente ±3 desviaciones estándar)
        constrained_weights = np.clip(weights, -3, 3)
        
        # Reconstruimos la forma
        shape_params = constrained_weights * np.sqrt(self.eigenvalues)
        shape_variation = np.dot(shape_params, self.eigenvectors)
        reconstructed = self.mean_shape.flatten() + shape_variation
        
        return reconstructed.reshape(-1, 2)
        
    def match_to_new_image(self, initial_shape, image, n_iterations=10):
        """
        Ajusta el modelo a una nueva imagen
        
        Args:
            initial_shape: Forma inicial para empezar la búsqueda
            image: Imagen donde buscar
            n_iterations: Número de iteraciones para el ajuste
            
        Returns:
            Forma ajustada
        """
        # Esta es una función simplificada para ilustrar el proceso
        # En una implementación real, se usaría búsqueda de perfiles locales
        
        if self.mean_shape is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        # Alineamos la forma inicial a la forma media
        aligned_initial, _ = self._procrustes(initial_shape, self.mean_shape)
        
        # Inicializamos con la forma media
        current_shape = self.mean_shape.copy()
        
        # Iteraciones de ajuste
        for _ in range(n_iterations):
            # En una implementación real, aquí buscaríamos mejores posiciones 
            # para cada landmark basándonos en perfiles de intensidad
            
            # Proyectamos la forma actual en el espacio del modelo
            shape_vector = current_shape.flatten() - self.mean_shape.flatten()
            weights = np.dot(shape_vector, self.eigenvectors.T) / np.sqrt(self.eigenvalues)
            
            # Limitamos los pesos a un rango razonable
            constrained_weights = np.clip(weights, -3, 3)
            
            # Reconstruimos la forma con los pesos constrained
            current_shape = self.reconstruct_shape(constrained_weights)
            
        return current_shape
    
    def visualize_fitting_process(self, image, initial_shape, n_iterations=10, step=2):
        """
        Visualiza el proceso de ajuste del modelo a una nueva imagen
        
        Args:
            image: Imagen donde buscar
            initial_shape: Forma inicial
            n_iterations: Número total de iteraciones
            step: Mostrar cada cuántas iteraciones
        """
        if self.mean_shape is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        plt.figure(figsize=(12, 8))
        
        # Mostramos la imagen
        plt.imshow(image, cmap='gray')
        
        # Mostramos la forma inicial
        x, y = initial_shape[:, 0], initial_shape[:, 1]
        plt.plot(x, y, 'b-', linewidth=1)
        plt.plot(x, y, 'bo', markersize=3, label='Inicial')
        
        # Proceso de ajuste
        current_shape = initial_shape.copy()
        
        for i in range(n_iterations):
            # En una implementación real, aquí se haría el ajuste
            # Para la visualización, simulamos el proceso
            
            if i % step == 0 or i == n_iterations - 1:
                color = plt.cm.jet(i / n_iterations)
                x, y = current_shape[:, 0], current_shape[:, 1]
                plt.plot(x, y, '-', color=color, linewidth=1)
                plt.plot(x, y, 'o', color=color, markersize=3, 
                         label=f'Iteración {i+1}' if i == n_iterations - 1 else None)
            
            # Simulamos un paso de ajuste (en la implementación real sería diferente)
            shape_vector = current_shape.flatten() - self.mean_shape.flatten()
            weights = np.dot(shape_vector, self.eigenvectors.T) / np.sqrt(self.eigenvalues)
            constrained_weights = np.clip(weights, -3, 3)
            current_shape = self.reconstruct_shape(constrained_weights)
        
        plt.title('Proceso de Ajuste del Modelo ASM')
        plt.legend()
        plt.axis('equal')
        plt.show()
        
    def save_model(self, filename):
        """Guarda el modelo entrenado"""
        if self.mean_shape is None:
            raise ValueError("No hay modelo para guardar")
            
        np.savez(filename, 
                 mean_shape=self.mean_shape,
                 eigenvectors=self.eigenvectors,
                 eigenvalues=self.eigenvalues,
                 n_landmarks=self.n_landmarks)
        print(f"Modelo guardado en {filename}")
        
    def load_model(self, filename):
        """Carga un modelo previamente entrenado"""
        data = np.load(filename)
        self.mean_shape = data['mean_shape']
        self.eigenvectors = data['eigenvectors']
        self.eigenvalues = data['eigenvalues']
        self.n_landmarks = int(data['n_landmarks'])
        
        # Reconstruir el objeto PCA
        n_components = self.eigenvectors.shape[0]
        self.pca = PCA(n_components=n_components)
        self.pca.components_ = self.eigenvectors
        self.pca.explained_variance_ = self.eigenvalues
        self.pca.explained_variance_ratio_ = self.eigenvalues / np.sum(self.eigenvalues)
        self.pca.mean_ = self.mean_shape.flatten()
        
        print(f"Modelo cargado con {self.n_landmarks} landmarks")
        
    def evaluate_model(self):
        """Evalúa la calidad del modelo entrenado"""
        if self.mean_shape is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        # Calculamos la varianza explicada acumulada
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Análisis de Componentes Principales')
        plt.grid(True)
        
        # Añadir líneas de referencia
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% de varianza')
        plt.axhline(y=0.95, color='g', linestyle='--', label='95% de varianza')
        
        # Encontrar cuántos componentes explican el 90% y 95% de la varianza
        n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        plt.legend()
        plt.show()
        
        print(f"El {cumulative_variance[0]*100:.1f}% de la varianza es explicada por el primer componente")
        print(f"Se necesitan {n_components_90} componentes para explicar el 90% de la varianza")
        print(f"Se necesitan {n_components_95} componentes para explicar el 95% de la varianza")
        
        return cumulative_variance

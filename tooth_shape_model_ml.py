import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class ToothContourDataset(Dataset):
    """Dataset para contornos dentales"""
    def __init__(self, contours, transform=None):
        """
        Args:
            contours: Lista de arrays numpy con landmarks de contornos dentales
            transform: Transformaciones opcionales a aplicar
        """
        # Convertimos todos los contornos a tensores
        self.contours = [torch.tensor(contour, dtype=torch.float32) for contour in contours]
        self.transform = transform
        
    def __len__(self):
        return len(self.contours)
    
    def __getitem__(self, idx):
        contour = self.contours[idx]
        
        if self.transform:
            contour = self.transform(contour)
            
        # Aplanamos el contorno para facilitar el procesamiento
        return contour.flatten()

class ToothContourAutoencoder(nn.Module):
    """
    Autoencoder para modelar contornos dentales
    
    Este modelo puede:
    1. Comprimir la representación de contornos a un espacio latente de baja dimensión
    2. Reconstruir contornos completos a partir de este espacio latente
    3. Generar nuevos contornos variando el espacio latente
    """
    def __init__(self, n_landmarks=40, latent_dim=8):
        """
        Args:
            n_landmarks: Número de puntos en el contorno
            latent_dim: Dimensión del espacio latente
        """
        super(ToothContourAutoencoder, self).__init__()
        
        self.n_landmarks = n_landmarks
        self.input_dim = n_landmarks * 2  # x,y para cada landmark
        self.latent_dim = latent_dim
        
        # Encoder (reduce la dimensionalidad de los contornos)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.latent_dim)
        )
        
        # Decoder (reconstruye contornos a partir del espacio latente)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.input_dim)
        )
        
    def encode(self, x):
        """Codifica un contorno al espacio latente"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decodifica del espacio latente a un contorno"""
        return self.decoder(z)
    
    def forward(self, x):
        """Pasa completo: codificación y decodificación"""
        z = self.encode(x)
        return self.decode(z)
    
    def generate_tooth(self, z=None):
        """
        Genera un contorno dental desde el espacio latente.
        Si z es None, genera un vector aleatorio del espacio latente.
        """
        if z is None:
            # Generamos un vector aleatorio con distribución normal
            z = torch.randn(1, self.latent_dim)
            
        with torch.no_grad():
            contour = self.decode(z)
            
        # Reshape de vuelta a la forma del contorno (n_landmarks, 2)
        return contour.view(-1, 2)
    
    def interpolate_teeth(self, contour1, contour2, steps=10):
        """
        Genera contornos intermedios interpolando entre dos contornos existentes
        
        Args:
            contour1, contour2: Arrays o tensores de contornos
            steps: Número de pasos intermedios a generar
        
        Returns:
            Lista de contornos interpolados
        """
        # Convertimos a tensores si no lo son ya
        if not isinstance(contour1, torch.Tensor):
            contour1 = torch.tensor(contour1, dtype=torch.float32)
        if not isinstance(contour2, torch.Tensor):
            contour2 = torch.tensor(contour2, dtype=torch.float32)
            
        # Aplanamos los contornos
        contour1_flat = contour1.flatten().unsqueeze(0)
        contour2_flat = contour2.flatten().unsqueeze(0)
        
        # Codificamos en el espacio latente
        z1 = self.encode(contour1_flat)
        z2 = self.encode(contour2_flat)
        
        # Generamos interpolaciones en el espacio latente
        interpolated_contours = []
        
        with torch.no_grad():
            for alpha in np.linspace(0, 1, steps):
                # Interpolación lineal
                z_interp = z1 * (1 - alpha) + z2 * alpha
                
                # Decodificamos para obtener el contorno
                contour_interp = self.decode(z_interp)
                
                # Convertimos de vuelta a la forma (n_landmarks, 2)
                contour_reshaped = contour_interp.view(-1, 2)
                interpolated_contours.append(contour_reshaped)
                
        return interpolated_contours

class DeepToothModel:
    """
    Modelo completo para segmentación dental basado en deep learning
    Extiende el autoencoder con funcionalidades adicionales
    """
    def __init__(self, n_landmarks=40, latent_dim=8, device=None):
        """
        Args:
            n_landmarks: Número de landmarks por contorno
            latent_dim: Dimensión del espacio latente
            device: Dispositivo para ejecutar los cálculos (CPU/GPU)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.n_landmarks = n_landmarks
        self.latent_dim = latent_dim
        self.model = ToothContourAutoencoder(n_landmarks, latent_dim).to(self.device)
        self.optimizer = None
        self.trained = False
        self.contours_training = None
        
    def preprocess_contours(self, contours):
        """
        Preprocesa los contornos para uniformizar su representación
        """
        processed = []
        
        for contour in contours:
            # Resampleamos si es necesario para tener exactamente n_landmarks
            if len(contour) != self.n_landmarks:
                # Aquí se podría implementar el resampling
                # Por simplicidad asumimos que ya tienen el número correcto
                pass
                
            # Normalizamos la escala y posición
            centered = contour - np.mean(contour, axis=0)
            scale = np.max(np.abs(centered))
            normalized = centered / scale if scale > 0 else centered
            
            processed.append(normalized)
            
        return processed
    
    def train(self, contours, batch_size=4, epochs=100, lr=0.001):
        """
        Entrena el modelo con un conjunto de contornos dentales
        
        Args:
            contours: Lista de arrays numpy con landmarks
            batch_size: Tamaño del batch para entrenamiento
            epochs: Número de épocas
            lr: Tasa de aprendizaje
        """
        # Preprocesamos los contornos
        self.contours_training = contours
        processed_contours = self.preprocess_contours(contours)
        
        # Creamos el dataset y dataloader
        dataset = ToothContourDataset(processed_contours)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Configuramos el optimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Ciclo de entrenamiento
        losses = []
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                # Movemos los datos al dispositivo
                batch = batch.to(self.device)
                
                # Reset de gradientes
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model(batch)
                
                # Calculamos la pérdida (MSE)
                loss = F.mse_loss(reconstructed, batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            # Pérdida promedio de la época
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Imprimimos progreso cada 10 épocas
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch+1}/{epochs}, Pérdida: {avg_loss:.6f}")
                
        self.trained = True
        
        # Visualizamos la curva de aprendizaje
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Curva de Aprendizaje")
        plt.xlabel("Época")
        plt.ylabel("Pérdida (MSE)")
        plt.grid(True)
        plt.show()
        
        print(f"Entrenamiento completado. Pérdida final: {losses[-1]:.6f}")
        
        return losses
    
    def visualize_reconstructions(self, contours, n_samples=5):
        """
        Visualiza contornos originales vs reconstruidos
        
        Args:
            contours: Lista de contornos a reconstruir
            n_samples: Número de muestras a visualizar
        """
        if not self.trained:
            print("El modelo no ha sido entrenado aún.")
            return
            
        # Preprocesamos los contornos
        processed = self.preprocess_contours(contours)
        
        # Seleccionamos muestras aleatorias
        indices = np.random.choice(len(processed), min(n_samples, len(processed)), replace=False)
        
        self.model.eval()
        plt.figure(figsize=(15, 3*n_samples))
        
        for i, idx in enumerate(indices):
            # Original
            original = torch.tensor(processed[idx].flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Reconstrucción
            with torch.no_grad():
                reconstructed = self.model(original).cpu().numpy().reshape(-1, 2)
                
            # Visualizamos
            plt.subplot(n_samples, 2, 2*i+1)
            plt.plot(processed[idx][:, 0], processed[idx][:, 1], 'b-o', markersize=4)
            plt.title(f"Original #{idx}")
            plt.axis('equal')
            plt.grid(True)
            
            plt.subplot(n_samples, 2, 2*i+2)
            plt.plot(reconstructed[:, 0], reconstructed[:, 1], 'r-o', markersize=4)
            plt.title(f"Reconstruido #{idx}")
            plt.axis('equal')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()
        
    def visualize_latent_space(self, contours, labels=None):
        """
        Visualiza el espacio latente de los contornos
        
        Args:
            contours: Lista de contornos a proyectar en espacio latente
            labels: Etiquetas opcionales para los contornos
        """
        if not self.trained:
            print("El modelo no ha sido entrenado aún.")
            return
            
        # Preprocesamos los contornos
        processed = self.preprocess_contours(contours)
        
        # Codificamos en el espacio latente
        self.model.eval()
        latent_vectors = []
        
        for contour in processed:
            contour_tensor = torch.tensor(contour.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                z = self.model.encode(contour_tensor).cpu().numpy()
                latent_vectors.append(z[0])
                
        latent_vectors = np.array(latent_vectors)
        
        # Visualizamos (solo las 2 o 3 primeras dimensiones)
        if self.latent_dim >= 2:
            plt.figure(figsize=(12, 10))
            
            # Visualización 2D
            plt.subplot(2, 1, 1)
            scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=labels if labels is not None else np.arange(len(latent_vectors)), 
                         cmap='viridis', alpha=0.8, s=80)
            
            if labels is not None:
                plt.colorbar(scatter, label='Tipo de Diente')
                
            plt.title("Proyección 2D del Espacio Latente")
            plt.xlabel("Dimensión 1")
            plt.ylabel("Dimensión 2")
            plt.grid(True)
            
            # Añadimos anotaciones numéricas
            for i, (x, y) in enumerate(zip(latent_vectors[:, 0], latent_vectors[:, 1])):
                plt.annotate(str(i), (x, y), fontsize=10)
            
            # Visualización 3D si hay suficientes dimensiones
            if self.latent_dim >= 3:
                ax = plt.subplot(2, 1, 2, projection='3d')
                scatter = ax.scatter(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2],
                          c=labels if labels is not None else np.arange(len(latent_vectors)), 
                          cmap='viridis', alpha=0.8, s=80)
                
                # Añadimos anotaciones numéricas
                for i, (x, y, z) in enumerate(zip(latent_vectors[:, 0], latent_vectors[:, 1], latent_vectors[:, 2])):
                    ax.text(x, y, z, str(i), fontsize=10)
                
                ax.set_title("Proyección 3D del Espacio Latente")
                ax.set_xlabel("Dimensión 1")
                ax.set_ylabel("Dimensión 2")
                ax.set_zlabel("Dimensión 3")
                
            plt.tight_layout()
            plt.show()
            
    def generate_variations(self, n_variations=5, std_dev=1.0):
        """
        Genera variaciones aleatorias de contornos dentales
        
        Args:
            n_variations: Número de variaciones a generar
            std_dev: Desviación estándar para la distribución normal
            
        Returns:
            Lista de contornos generados
        """
        if not self.trained:
            print("El modelo no ha sido entrenado aún.")
            return []
            
        variations = []
        self.model.eval()
        
        # Generamos muestras aleatorias del espacio latente
        for _ in range(n_variations):
            # Vector aleatorio con distribución normal
            z = torch.randn(1, self.latent_dim).to(self.device) * std_dev
            
            # Generamos el contorno
            with torch.no_grad():
                contour = self.model.decode(z).cpu().numpy().reshape(-1, 2)
                variations.append(contour)
                
        return variations
    
    def visualize_mode_variations(self, n_modes=3, std_dev=2.0):
        """
        Visualiza variaciones a lo largo de los modos principales del espacio latente.
        """
        if not self.trained:
            print("El modelo no ha sido entrenado aún.")
            return

        self.model.eval()
        n_modes = min(n_modes, self.latent_dim)
        factors = [-std_dev, 0.0, std_dev]
        colors  = ['r', 'b', 'g']

        plt.figure(figsize=(12, 4 * n_modes))
        for mode in range(n_modes):
            plt.subplot(n_modes, 1, mode+1)

            for j, factor in enumerate(factors):
                # Construye el vector latente
                z = torch.zeros(1, self.latent_dim, device=self.device)
                z[0, mode] = factor

                # Decodifica y convierte a numpy
                with torch.no_grad():
                    contour = self.model.decode(z).cpu().numpy().reshape(-1, 2)

                # Plot
                plt.plot(contour[:, 0], contour[:, 1], f'{colors[j]}-o',
                        markersize=4,
                        label=f"{factor:+.1f} σ" if factor != 0 else "Media")

            plt.title(f"Modo de Variación {mode+1}")
            plt.axis('equal')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, filename):
        """Guarda el modelo entrenado"""
        if not self.trained:
            print("El modelo no ha sido entrenado aún.")
            return
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_landmarks': self.n_landmarks,
            'latent_dim': self.latent_dim,
            'contours_training': self.contours_training
        }, filename)
        
        print(f"Modelo guardado en {filename}")
        
    def load_model(self, filename):
        """Carga un modelo previamente entrenado"""
        checkpoint = torch.load(filename)
        
        self.n_landmarks = checkpoint['n_landmarks']
        self.latent_dim = checkpoint['latent_dim']
        self.contours_training = checkpoint['contours_training']
        
        # Reconstruimos el modelo
        self.model = ToothContourAutoencoder(self.n_landmarks, self.latent_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.trained = True
        print(f"Modelo cargado con {self.n_landmarks} landmarks y dimensión latente {self.latent_dim}")
        
    def match_to_image(self, image, side, initial_contour, iterations=100, learning_rate=0.01):
        if not self.trained:
            print("El modelo no ha sido entrenado aún.")
            return initial_contour

        # 1) Consigue tus bordes y mapa de distancias
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 25, 85)
        inv = cv2.bitwise_not(edges)
        dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
        cv2.imwrite(f"debug/dist_map_raw_{side}.png", (dist_map / dist_map.max() * 255).astype(np.uint8))
        H, W = dist_map.shape

        # 2) Tensor de distancias en GPU
        dist_map_tensor = (
            torch.from_numpy(dist_map) 
                 .unsqueeze(0)   # batch dim
                 .unsqueeze(0)   # channel dim
                 .to(self.device)
        )

        # 3) Prepara el contorno inicial en latente
        proc_init = self.preprocess_contours([initial_contour])[0]
        contour_tensor = torch.tensor(proc_init.flatten(),
                                      dtype=torch.float32,
                                      device=self.device)
        with torch.no_grad():
            z = self.model.encode(contour_tensor.unsqueeze(0))
        z_opt = z.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([z_opt], lr=learning_rate)

        # 4) Define image_loss usando dist_map_tensor
        def image_loss(contour_pts, dist_map_tensor):
            # contour_pts: (N,2) en pixeles
            _, _, h, w = dist_map_tensor.shape
            xs = contour_pts[:, 0].clamp(0, w-1)
            ys = contour_pts[:, 1].clamp(0, h-1)

            # Normaliza coords a [-1,1] para grid_sample
            grid_x = 2.0 * (xs / (w-1)) - 1.0
            grid_y = 2.0 * (ys / (h-1)) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=1).view(1, -1, 1, 2)

            sampled = F.grid_sample(
                dist_map_tensor,
                grid,
                mode='bilinear',
                align_corners=True
            )  # (1,1,N,1)
            sampled = sampled.view(-1)  # (N,)
            return -sampled.mean()

        # 5) Bucle de optimización
        for it in range(iterations):
            optimizer.zero_grad()

            # Decodifica y mapea a píxeles
            contour_flat = self.model.decode(z_opt)       # (1,2N)
            pts = contour_flat.view(-1, 2)               # (N,2) en [-1,1]
            xs = (pts[:,0] + 1)*0.5*(W-1)
            ys = (1 - (pts[:,1] + 1)*0.5)*(H-1)
            contour_pts_pixels = torch.stack([xs, ys], dim=1)

            # Calcula pérdidas
            img_l = image_loss(contour_pts_pixels, dist_map_tensor)
            reg_l = torch.mean(z_opt**2)
            loss = img_l + 0.1 * reg_l

            loss.backward()
            optimizer.step()

            if (it+1) % 10 == 0:
                print(f"Iter {it+1}/{iterations}  img_loss={img_l.item():.3f}  reg={reg_l.item():.3f}")
            # if (it+1) % 20 == 0:
            # # recupera el contorno en pixeles
            #     cp = contour_pts_pixels.detach().cpu().numpy().astype(int)
            #     img_debug = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
            #     cv2.polylines(img_debug, [cp], isClosed=True, color=(255,0,0), thickness=1)
            #     cv2.imwrite(f"debug/iter_{it+1:03d}_{side}.png", img_debug)

        with torch.no_grad():
            final_flat = self.model.decode(z_opt).cpu().numpy().reshape(-1,2)
        # Mapea de nuevo a píxeles:
        final_pts = np.empty_like(final_flat)
        final_pts[:,0] = (final_flat[:,0] + 1)*0.5*(W-1)
        final_pts[:,1] = (1 - (final_flat[:,1] + 1)*0.5)*(H-1)

        return final_pts
    
    def get_mean_shape(self, contours):
        """
        Calcula y devuelve la forma media decodificando el vector latente promedio.
        """
        if not self.trained:
            print("El modelo no ha sido entrenado aún.")
            return None

        # Preprocesa los contornos (igual que en el entrenamiento)
        processed = self.preprocess_contours(contours)
    
        # Codifica todos los contornos en el espacio latente
        self.model.eval()
        latent_vectors = []
        for contour in processed:
            contour_tensor = torch.tensor(contour.flatten(), 
                                    dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            with torch.no_grad():
                z = self.model.encode(contour_tensor).cpu().numpy()
                latent_vectors.append(z)
    
        # Calcula el vector latente promedio
        z_mean = np.mean(latent_vectors, axis=0)
        z_mean_tensor = torch.tensor(z_mean, 
                               dtype=torch.float32, 
                               device=self.device)
    
        # Decodifica el vector promedio
        with torch.no_grad():
            mean_contour = self.model.decode(z_mean_tensor).cpu().numpy().reshape(-1, 2)
    
        return mean_contour

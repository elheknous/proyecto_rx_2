import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm

# Definición del modelo U-Net para segmentación
class DoubleConv(nn.Module):
    """Doble bloque de convolución (conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Capa de Downscaling: MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Capa de Upscaling: Upsample -> Concatenate -> DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Diferencia en dimensiones
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # Concatenación
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Convolución de salida 1x1"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Modelo U-Net completo para segmentación"""
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        # Camino de contracción (encoder)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # Camino de expansión (decoder)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        # Sigmoid para obtener probabilidades
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Aplicar sigmoid para obtener mapa de probabilidades
        return self.sigmoid(logits)

# Dataset personalizado para radiografías de caninos
class CanineDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Cargar imagen y máscara
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Leer imagen y convertir a escala de grises
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Leer máscara (binaria)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar a tamaño estándar
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)
        
        # Binarizar máscara si es necesario
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Normalizar valores
        image = image / 255.0
        mask = mask / 255.0
        
        # Aplicar transformaciones
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convertir a tensor y añadir dimensión de canal
        image = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(mask).float().unsqueeze(0)    # [1, H, W]
        
        return image, mask

# Función de pérdida Dice para segmentación
def dice_loss(pred, target, smooth=1e-5):
    """
    Calcula el coeficiente Dice (similar a IoU) y devuelve 1 - Dice como pérdida
    """
    pred = pred.contiguous()
    target = target.contiguous()    
    
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Función de evaluación
def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Binarizar predicciones con umbral 0.5
            pred_masks = (outputs > 0.5).float()
            
            # Calcular coeficiente Dice (métrica de segmentación)
            batch_dice = 1 - dice_loss(pred_masks, masks)
            
            total_dice += batch_dice.item() * images.size(0)
            total_samples += images.size(0)
    
    avg_dice = total_dice / total_samples
    return avg_dice

# Función para entrenar el modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # Seguimiento de pérdidas y métricas
    history = {
        'train_loss': [],
        'val_dice': []
    }
    
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Barra de progreso
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, masks in pbar:
            # Mover datos al dispositivo
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Actualizar pérdida
            train_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        
        # Calcular pérdida promedio
        train_loss = train_loss / len(train_loader.dataset)
        
        # Evaluar en conjunto de validación
        val_dice = evaluate_model(model, val_loader, device)
        
        # Guardar métricas
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        
        # Mostrar progreso
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Dice: {val_dice:.4f}')
        
        # Guardar mejor modelo
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_canine_model.pth')
            print(f'Model saved with Dice score: {val_dice:.4f}')
    
    return history

# Función para visualizar resultados
def visualize_results(model, test_loader, device, num_samples=5):
    model.eval()
    
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))
        
        for i, (images, masks) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_masks = (outputs > 0.5).float()
            
            # Mostrar imagen original
            axes[i, 0].imshow(images[0, 0].cpu().numpy(), cmap='gray')
            axes[i, 0].set_title('Radiografía Original')
            axes[i, 0].axis('off')
            
            # Mostrar máscara real
            axes[i, 1].imshow(masks[0, 0].cpu().numpy(), cmap='gray')
            axes[i, 1].set_title('Máscara Real')
            axes[i, 1].axis('off')
            
            # Mostrar predicción
            axes[i, 2].imshow(pred_masks[0, 0].cpu().numpy(), cmap='gray')
            axes[i, 2].set_title('Predicción')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('test_results.png')
        plt.show()

# Función para extraer contorno a partir de la máscara
def extract_contour(mask, min_contour_len=100):
    # Convertir a uint8 y asegurar valores binarios
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Seleccionar el contorno más grande
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Verificar que el contorno tenga un tamaño mínimo
        if len(largest_contour) >= min_contour_len:
            return largest_contour
        
    return None

# Función para inferencia en nuevas imágenes
def predict_canine_contour(model, img_path, device, target_size=(256, 256)):
    # Cargar imagen
    image = cv2.imread(img_path)
    orig_h, orig_w = image.shape[:2]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocesar
    img_resized = cv2.resize(gray, target_size)
    img_normalized = img_resized / 255.0
    
    # Convertir a tensor
    img_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Predicción
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = (output > 0.5).float()
    
    # Convertir a numpy
    pred_mask_np = pred_mask[0, 0].cpu().numpy()
    
    # Redimensionar al tamaño original
    pred_mask_orig = cv2.resize(pred_mask_np, (orig_w, orig_h))
    
    # Extraer contorno
    contour = extract_contour(pred_mask_orig)
    
    # Visualizar resultado
    result_img = image.copy()
    if contour is not None:
        cv2.drawContours(result_img, [contour], 0, (0, 255, 0), 2)
    
    # Mostrar resultados
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask_orig, cmap='gray')
    axes[1].set_title('Máscara Predicha')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Contorno Predicho')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    return contour, pred_mask_orig

# Función principal que ejecuta el flujo completo
def main():
    # Configuración
    img_size = (256, 256)
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001
    
    # Paths - AJUSTAR SEGÚN TU ESTRUCTURA DE DIRECTORIOS
    data_dir = 'datos_caninos/'
    img_dir = os.path.join(data_dir, 'radiografias')
    mask_dir = os.path.join(data_dir, 'mascaras')
    
    # Listar archivos (asumiendo nombres coincidentes entre imágenes y máscaras)
    all_images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
    all_masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    # Dividir datos
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        all_images, all_masks, test_size=0.2, random_state=42
    )
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs, train_masks, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(train_imgs)}, Validation: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    # Transformaciones para aumentación de datos (solo en entrenamiento)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.3),
    ])
    
    # Crear datasets
    train_dataset = CanineDataset(train_imgs, train_masks, transform=train_transform, size=img_size)
    val_dataset = CanineDataset(val_imgs, val_masks, size=img_size)
    test_dataset = CanineDataset(test_imgs, test_masks, size=img_size)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    
    # Configurar dispositivo (GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Crear modelo
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = model.to(device)
    
    # Optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = dice_loss
    
    # Entrenar modelo
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_canine_model.pth'))
    
    # Evaluar en conjunto de prueba
    test_dice = evaluate_model(model, test_loader, device)
    print(f"Test Dice Score: {test_dice:.4f}")
    
    # Visualizar resultados
    visualize_results(model, test_loader, device)
    
    # Ejemplo de inferencia en una nueva imagen
    sample_img = test_imgs[0]  # Usar primera imagen de prueba como ejemplo
    contour, mask = predict_canine_contour(model, sample_img, device)
    
    # Graficar curvas de aprendizaje
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_dice'], label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    plt.show()

if __name__ == "__main__":
    main()
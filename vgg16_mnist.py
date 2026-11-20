# -*- coding: utf-8 -*-
"""Arquitectura VGG16 para MNIST y adaptador compatible con predict_proba."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16MNIST(nn.Module):
    """13 capas convolucionales + 3 capas densas (VGG16), adaptada a 10 clases.

    Se conservan las 13 capas convolucionales de VGG16. Se reemplaza
    la cabeza original de ImageNet por tres capas densas más pequeñas
    (512->256->128->10) para clasificar MNIST con 10 clases.
    """
    def __init__(self, image_size=48):
        super().__init__()
        self.image_size = image_size
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
               512, 512, 512, "M", 512, 512, 512, "M"]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)])
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(128, 10)
        )

    def forward(self, images):
        # Entrada: (N,1,28,28) en [0,255]. Las 3 bandas usan el mismo canal gris.
        x = F.interpolate(images.float(), size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1) / 255.0
        # Normalización adecuada para los pesos oficiales de ImageNet.
        mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)  # Logits; probabilidades con softmax en el adaptador.


def to_mnist_tensor(images):
    """Acepta (28,28), (N,28,28), (784,), (N,784), (N,1,28,28).

    Si los píxeles están en [0,1], los devuelve a la escala [0,255].
    """
    a = np.asarray(images, dtype=np.float32)
    if a.ndim == 1 and a.size == 784:
        a = a.reshape(1, 1, 28, 28)
    elif a.ndim == 2 and a.shape == (28, 28):
        a = a.reshape(1, 1, 28, 28)
    elif a.ndim == 2 and a.shape[1] == 784:
        a = a.reshape(-1, 1, 28, 28)
    elif a.ndim == 3 and a.shape[1:] == (28, 28):
        a = a[:, None, :, :]
    elif a.ndim == 4 and a.shape[1:] == (1, 28, 28):
        pass
    else:
        raise ValueError(f"Forma no soportada: {a.shape}; se requieren imágenes MNIST 28x28.")
    a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
    a = np.clip(a, 0.0, 255.0)
    if a.size and a.max() <= 1.0:
        a = a * 255.0
    return torch.from_numpy(np.ascontiguousarray(a))


class VGG16Classifier:
    """Modelo persistente con interfaz semejante a sklearn y predict_proba."""
    def __init__(self, model, batch_size=32):
        self.model = model.cpu().eval()
        self.batch_size = batch_size
        self.classes_ = np.arange(10)
        self._runtime_device = None

    def predict_proba(self, images):
        batch = to_mnist_tensor(images)
        if self._runtime_device is None:
            self._runtime_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self._runtime_device)
            self.model.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(batch), self.batch_size):
                x = batch[start:start + self.batch_size].to(self._runtime_device)
                logits = self.model(x)
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(chunks) if chunks else np.empty((0, 10), dtype=np.float32)

    def predict(self, images):
        return self.classes_[np.argmax(self.predict_proba(images), axis=1)]

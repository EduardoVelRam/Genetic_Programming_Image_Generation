# -*- coding: utf-8 -*-
"""Entrena VGG16 sobre MNIST y guarda vgg16_mnist.joblib para gp_generation_phase.py.

Requiere los módulos import_data.py y vgg16_mnist.py en el mismo directorio.
Dependencias: torch torchvision numpy scikit-learn joblib
"""
import argparse
import copy
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import VGG16_Weights, vgg16

from import_data import load_mnist, mnist_path
from vgg16_mnist import VGG16Classifier, VGG16MNIST, to_mnist_tensor

BASE_DIR = Path(__file__).resolve().parent


def make_loader(images, labels, batch_size, shuffle, device):
    inputs = to_mnist_tensor(images)
    targets = torch.as_tensor(np.asarray(labels, dtype=np.int64).copy())
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=(device.type == "cuda"))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_ok = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)
        total_loss += loss.item() * len(y)
        total_ok += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, total_ok / total


def train_stage(model, train_loader, val_loader, epochs, optimizer,
                device, best_state, best_val, label):
    if epochs <= 0:
        return best_state, best_val
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(y)
            seen += len(y)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"{label} | época {epoch}/{epochs} | loss_train={running_loss/seen:.4f} "
              f"| loss_val={val_loss:.4f} | accuracy_val={val_acc*100:.2f}%", flush=True)
        # Se conserva la mejor generalización, no necesariamente la última época.
        if val_acc > best_val[0] or (val_acc == best_val[0] and val_loss < best_val[1]):
            best_state = copy.deepcopy(model.state_dict())
            best_val = (val_acc, val_loss)
    return best_state, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs-finetune", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Limitar MNIST para pruebas rápidas (por defecto, todo).")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Entrenar todas las capas sin pesos de ImageNet (más lento).")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs_head < 0 or args.epochs_finetune < 0:
        parser.error("batch-size debe ser positivo y las épocas no pueden ser negativas.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Semilla={args.seed} | dispositivo={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    X_train, y_train, X_test, y_test = load_mnist(mnist_path)
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)

    if args.max_train is not None:
        if not 100 <= args.max_train <= len(X_train):
            parser.error("--max-train debe estar entre 100 y el total disponible.")
        if args.max_train < len(X_train):
            indices, _ = train_test_split(
                np.arange(len(y_train)), train_size=args.max_train,
                random_state=args.seed, stratify=y_train)
            X_train, y_train = X_train[indices], y_train[indices]

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.10,
        stratify=y_train, random_state=args.seed)
    train_loader = make_loader(X_train[train_idx], y_train[train_idx],
                               args.batch_size, True, device)
    val_loader = make_loader(X_train[val_idx], y_train[val_idx],
                             args.batch_size, False, device)
    test_loader = make_loader(X_test, y_test, args.batch_size, False, device)
    print(f"MNIST: {len(train_idx)} entrenamiento, {len(val_idx)} validación, "
          f"{len(y_test)} prueba", flush=True)

    model = VGG16MNIST(image_size=48)
    if not args.from_scratch:
        print("Cargando pesos preentrenados de VGG16 en ImageNet...", flush=True)
        pretrained = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        model.features.load_state_dict(pretrained.features.state_dict(), strict=True)
        del pretrained
    model = model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = (-1.0, float("inf"))

    if args.from_scratch:
        # No se congelan capas que todavía tienen pesos aleatorios.
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        best_state, best_val = train_stage(
            model, train_loader, val_loader,
            args.epochs_head + args.epochs_finetune, opt, device,
            best_state, best_val, "Desde cero")
    else:
        # Etapa 1: aprender únicamente la cabeza MNIST.
        for p in model.features.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        best_state, best_val = train_stage(
            model, train_loader, val_loader, args.epochs_head, opt,
            device, best_state, best_val, "Cabeza")
        # Etapa 2: ajustar bloque convolucional 5 (capas features[24:]).
        for p in model.features[24:].parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW([
            {"params": model.features[24:].parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 2e-4},
        ], weight_decay=1e-4)
        best_state, best_val = train_stage(
            model, train_loader, val_loader, args.epochs_finetune, opt,
            device, best_state, best_val, "Fine-tuning")

    if best_val[0] < 0:
        parser.error("Especifique al menos una época de entrenamiento.")
    model.load_state_dict(best_state)
    _, accuracy_test = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Mejor accuracy de validación: {best_val[0]*100:.2f}%", flush=True)
    print(f"Accuracy de prueba (MNIST): {accuracy_test*100:.2f}%", flush=True)

    # Para que joblib pueda cargarlo incluso en equipos sin CUDA.
    model = model.cpu().eval()
    output = BASE_DIR / "vgg16_mnist.joblib"
    joblib.dump(VGG16Classifier(model), output, compress=3)
    print(f"Modelo guardado en: {output}", flush=True)
    print("Prueba: joblib.load('vgg16_mnist.joblib').predict_proba(X_test[0])")


if __name__ == "__main__":
    main()

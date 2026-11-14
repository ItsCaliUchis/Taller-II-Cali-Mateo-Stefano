"""Entrena MLPs de sentimiento sobre embeddings densos y dispersos."""

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import optuna
from scipy.sparse import issparse, load_npz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from utils import agregar_argumento_limit, ejecutar_etapa, obtener_limit
from .data import CLASS_NAMES, load_sentiment_labels
from .graphics import graficar_historial_rna, graficar_matrices_confusion


class SentimentMLP(nn.Module):
    def __init__(self, input_size, hidden_dims, dropout):
        super().__init__()
        layers = []
        size = input_size
        for hidden_size in hidden_dims:
            layers.extend((nn.Linear(size, hidden_size), nn.ReLU(), nn.Dropout(dropout)))
            size = hidden_size
        layers.append(nn.Linear(size, len(CLASS_NAMES)))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        if features.layout in (torch.sparse_coo, torch.sparse_csr):
            first = self.network[0]
            # BOW/TF-IDF suelen tener menos de 0.1% de valores distintos de cero.
            # Multiplicar la matriz dispersa evita calcular sobre sus ceros.
            with torch.autocast(device_type="cuda", enabled=False) if features.is_cuda else nullcontext():
                hidden = torch.sparse.mm(features.float(), first.weight.float().t())
                if first.bias is not None:
                    hidden = hidden + first.bias.float()
            hidden = self.network[2](self.network[1](hidden))
            return self.network[3:](hidden)
        return self.network(features)


class _IndexedDataset(Dataset):
    """Vista de índices; el minibatch completo se extrae de la matriz de una vez."""

    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return int(self.indices[index])


class _BatchCollator:
    """Extrae y prepara una minibatch completa, evitando trabajo fila por fila."""

    def __init__(self, features, labels, scaler=None, sparse_batches=False):
        self.features = features
        self.labels = labels
        self.scaler = scaler
        self.sparse_batches = sparse_batches

    def __call__(self, rows):
        row_indices = np.asarray(rows, dtype=np.int64)
        batch = self.features[row_indices]
        if issparse(batch):
            batch = batch.tocsr()
            if self.scaler is not None:
                batch = self.scaler.transform(batch).tocsr()
            if self.sparse_batches:
                crow = torch.from_numpy(batch.indptr.astype(np.int64, copy=False))
                columns = torch.from_numpy(batch.indices.astype(np.int64, copy=False))
                values = torch.from_numpy(np.asarray(batch.data, dtype=np.float32))
                batch_tensor = torch.sparse_csr_tensor(crow, columns, values, size=batch.shape)
            else:
                batch_tensor = torch.from_numpy(np.asarray(batch.toarray(), dtype=np.float32))
        else:
            values = np.asarray(batch, dtype=np.float32)
            if self.scaler is not None:
                values = self.scaler.transform(values)
            batch_tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
        labels_tensor = torch.from_numpy(np.asarray(self.labels[row_indices], dtype=np.int64))
        return batch_tensor, labels_tensor


def _parse_hidden_dims(value):
    try:
        dimensions = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use dimensiones enteras separadas por comas, ej. 256,128.") from error
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("Las dimensiones ocultas deben ser enteros positivos.")
    return dimensions


def _select_device(requested):
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Se solicitó CUDA, pero torch.cuda.is_available() es False. "
            "Instala una build CUDA de PyTorch y verifica el driver NVIDIA."
        )
    return device


def _input_signature(rep, matrix_path):
    dataset_stat = config.dataset_clean.stat()
    matrix_stat = Path(matrix_path).stat()
    return {
        "dataset": [str(config.dataset_clean.resolve()), dataset_stat.st_size, dataset_stat.st_mtime_ns],
        "embedding": [str(Path(matrix_path).resolve()), matrix_stat.st_size, matrix_stat.st_mtime_ns],
        "representation": rep,
    }


def _run_id(signature, args):
    parameters = {
        "signature": signature,
        "limit": args.limit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "amp": args.amp,
        "scale": getattr(args, "scale", None),
        "phase": getattr(args, "phase", "baseline"),
    }
    if getattr(args, "feature_selection", None):
        parameters["feature_selection"] = args.feature_selection
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20], parameters


def _reduce_sparse_features(features, labels, train_indices, signature, args):
    """Selecciona columnas por chi-cuadrado usando solo filas de entrenamiento."""
    original_dim = features.shape[1]
    if not issparse(features) or original_dim <= args.max_features:
        return features, None
    if features.data.size and np.min(features.data) < 0:
        raise ValueError("La selección chi2 requiere embeddings dispersos no negativos.")

    cache_payload = {
        "input_signature": signature,
        "seed": args.seed,
        "max_features": args.max_features,
        "selection": "chi2-train-only-v1",
    }
    cache_id = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    cache_path = config.TRAINING_CACHE / f"rna_feature_columns_{cache_id}.npy"
    selected = None
    if cache_path.is_file():
        try:
            candidate = np.load(cache_path, allow_pickle=False)
            if (
                candidate.ndim == 1
                and len(candidate) == args.max_features
                and candidate.min() >= 0
                and candidate.max() < original_dim
            ):
                selected = candidate.astype(np.int64, copy=False)
        except (OSError, ValueError):
            cache_path.unlink(missing_ok=True)

    if selected is None:
        selector = SelectKBest(score_func=chi2, k=args.max_features)
        selector.fit(features[train_indices], labels[train_indices])
        selected = selector.get_support(indices=True).astype(np.int64, copy=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as cache_file:
            np.save(cache_file, selected, allow_pickle=False)
        temporary.replace(cache_path)
        print(
            f"[{args.representation}] selección chi2: {original_dim:,} → "
            f"{len(selected):,} características (ajustada solo con train).",
            flush=True,
        )

    return features[:, selected].tocsr(), selected


def _artifact_paths(rep, limit, *, scale=False, phase="baseline"):
    suffix = f"_limit_{limit}" if limit is not None else ""
    suffix += f"_{phase}_{'scaled' if scale else 'unscaled'}"
    report_dir = config.rna_report_dir(rep)
    result_dir = config.RNA_RESULTS
    return {
        "report": report_dir / f"rna_{rep}_report{suffix}.csv",
        "confusion_csv": report_dir / f"rna_{rep}_confusion{suffix}.csv",
        "confusion_png": report_dir / f"rna_{rep}_confusion{suffix}.png",
        "history_csv": report_dir / f"rna_{rep}_history{suffix}.csv",
        "curves_png": report_dir / f"rna_{rep}_curves{suffix}.png",
        "model": result_dir / f"rna_{rep}_best_model{suffix}.pt",
        "metadata": result_dir / f"rna_{rep}_metadata{suffix}.json",
        "complete": result_dir / f"rna_{rep}_complete{suffix}.json",
    }


def _atomic_json(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(contents, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(contents, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(contents, temporary)
    temporary.replace(path)


def _checkpoint_paths(rep, run_id):
    directory = config.RNA_CHECKPOINTS / rep / run_id
    return directory, directory / "last.pt", directory / "best.pt"


def _save_checkpoint(path, *, model, optimizer, scaler, epoch, best_f1, best_epoch,
                     bad_epochs, history, generator, completed=False):
    checkpoint = {
        "version": 1,
        "epoch": epoch,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator_state": generator.get_state(),
        "completed": completed,
    }
    _atomic_torch_save(checkpoint, path)


def _make_loader(features, labels, indices, *, batch_size, shuffle, device, generator, num_workers, scaler=None):
    dataset = _IndexedDataset(indices)
    sparse_batches = issparse(features) and features.nnz / max(features.shape[0] * features.shape[1], 1) < 0.20
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda" and not sparse_batches,
        persistent_workers=num_workers > 0,
        collate_fn=_BatchCollator(features, labels, scaler=scaler, sparse_batches=sparse_batches),
    )


def _evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    losses = 0.0
    count = 0
    targets = []
    predictions = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            amp_context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_context:
                logits = model(features)
                loss = criterion(logits, labels)
            losses += float(loss.item()) * len(labels)
            count += len(labels)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return losses / max(count, 1), f1_score(targets, predictions, average="macro", zero_division=0), targets, predictions


def _fit_representation(
    rep, features, labels, matrix_path, args, device, *, scale=False,
    phase="baseline", trial=None, save_artifacts=True,
):
    if features.shape[0] != len(labels):
        raise ValueError(
            f"La matriz {rep} tiene {features.shape[0]} filas pero hay {len(labels)} etiquetas. "
            "Comprueba que la matriz y el dataset usan el mismo --limit y orden de filas."
        )

    signature = _input_signature(rep, matrix_path)
    args.scale = scale
    args.phase = phase
    args.representation = rep
    args.feature_selection = (
        {"method": "chi2", "max_features": args.max_features, "version": 1}
        if issparse(features) and features.shape[1] > args.max_features
        else None
    )
    run_id, parameters = _run_id(signature, args)
    output_paths = _artifact_paths(rep, args.limit, scale=scale, phase=phase)
    checkpoint_dir, latest_path, best_path = _checkpoint_paths(rep, run_id)

    if args.restart and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    if output_paths["complete"].exists() and not args.overwrite and not args.restart:
        try:
            previous = json.loads(output_paths["complete"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("run_id") == run_id and all(
            output_paths[key].is_file() and output_paths[key].stat().st_size > 0
            for key in ("report", "confusion_csv", "confusion_png", "history_csv", "curves_png", "model", "metadata")
        ):
            if trial is None and checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            print(f"[{rep}] resultados válidos existentes; se omite.")
            return previous
        if previous and previous.get("run_id") != run_id:
            raise FileExistsError(f"[{rep}] ya hay resultados de otra configuración; usa --overwrite.")

    generated_paths = [path for key, path in output_paths.items() if key != "complete"]
    partial_outputs = [path for path in generated_paths if path.exists()]
    if partial_outputs and not (args.overwrite or args.restart or latest_path.exists()):
        raise FileExistsError(f"[{rep}] hay resultados parciales; usa --overwrite para regenerarlos.")

    trainval_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.20,
        shuffle=True,
        stratify=labels,
        random_state=102,
    )
    train_indices, val_indices = train_test_split(
        trainval_indices,
        test_size=0.10,
        shuffle=True,
        stratify=labels[trainval_indices],
        random_state=args.seed,
    )
    original_input_dim = features.shape[1]
    features, selected_features = ejecutar_etapa(
        f"RNA: selección de características {rep}",
        _reduce_sparse_features,
        features,
        labels,
        train_indices,
        signature,
        args,
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    if not issparse(features):
        features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    scaler = StandardScaler(with_mean=False).fit(features[train_indices]) if scale else None
    features_tensor = features
    train_loader = _make_loader(
        features_tensor, labels, train_indices,
        batch_size=args.batch_size, shuffle=True, device=device,
        generator=generator, num_workers=args.num_workers,
        scaler=scaler,
    )
    train_eval_loader = _make_loader(
        features_tensor, labels, train_indices,
        batch_size=args.batch_size, shuffle=False, device=device,
        generator=generator, num_workers=args.num_workers,
        scaler=scaler,
    )
    val_loader = _make_loader(
        features_tensor, labels, val_indices,
        batch_size=args.batch_size, shuffle=False, device=device,
        generator=generator, num_workers=args.num_workers,
        scaler=scaler,
    )
    test_loader = _make_loader(
        features_tensor, labels, test_indices,
        batch_size=args.batch_size, shuffle=False, device=device,
        generator=generator, num_workers=args.num_workers,
        scaler=scaler,
    )

    model = SentimentMLP(features.shape[1], args.hidden_dims, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_macro_f1": [], "val_macro_f1": []}
    start_epoch = 0
    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    completed_checkpoint = False

    if trial is None and latest_path.exists() and not args.restart:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError(f"Checkpoint incompatible: {latest_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"])
        best_f1 = float(checkpoint["best_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = checkpoint["history"]
        generator.set_state(checkpoint["loader_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        completed_checkpoint = bool(checkpoint.get("completed", False))
        print(f"[{rep}] reanudando desde época {start_epoch}; mejor F1 macro {best_f1:.4f}.")

    best_state = None
    if trial is None and best_path.exists() and not args.restart:
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        best_state = best_checkpoint["model_state_dict"]

    if not completed_checkpoint:
        for epoch in range(start_epoch, args.epochs):
            started = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_rows = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device, non_blocking=device.type == "cuda")
                batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                amp_context = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
                with amp_context:
                    logits = model(batch_features)
                    loss = criterion(logits, batch_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item()) * len(batch_labels)
                total_rows += len(batch_labels)

            train_loss, train_f1, _, _ = _evaluate(model, train_eval_loader, criterion, device, use_amp)
            val_loss, val_f1, _, _ = _evaluate(model, val_loader, criterion, device, use_amp)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss if total_rows else 0.0)
            history["val_loss"].append(val_loss)
            history["train_macro_f1"].append(train_f1)
            history["val_macro_f1"].append(val_f1)

            improved = val_f1 > best_f1
            if improved:
                best_f1 = val_f1
                best_epoch = epoch + 1
                bad_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if trial is None:
                    _atomic_torch_save({"version": 1, "model_state_dict": best_state, "epoch": best_epoch, "val_macro_f1": best_f1}, best_path)
            else:
                bad_epochs += 1

            if trial is None:
                _save_checkpoint(
                    latest_path, model=model, optimizer=optimizer, scaler=scaler,
                    epoch=epoch + 1, best_f1=best_f1, best_epoch=best_epoch,
                    bad_epochs=bad_epochs, history=history, generator=generator,
                    completed=bad_epochs >= args.patience or epoch + 1 >= args.epochs,
                )
            if trial is not None:
                trial.report(val_f1, step=epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            print(
                f"[{rep}] época {epoch + 1}/{args.epochs} | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"F1 macro {train_f1:.4f}/{val_f1:.4f} | "
                f"{f'VRAM {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f}/{torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB asignada/reservada | ' if device.type == 'cuda' else ''}"
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
            if bad_epochs >= args.patience:
                print(f"[{rep}] early stopping: sin mejora durante {args.patience} épocas.")
                break

        if best_state is None:
            if best_path.exists():
                best_state = torch.load(best_path, map_location="cpu", weights_only=True)["model_state_dict"]
            else:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if trial is None:
            _save_checkpoint(
                latest_path, model=model, optimizer=optimizer, scaler=scaler,
                epoch=len(history["epoch"]), best_f1=best_f1, best_epoch=best_epoch,
                bad_epochs=bad_epochs, history=history, generator=generator,
                completed=True,
            )

    model.load_state_dict(best_state)
    if not save_artifacts:
        return {"best_validation_macro_f1": best_f1, "best_epoch": best_epoch}

    test_loss, test_f1, y_test, y_pred = _evaluate(model, test_loader, criterion, device, use_amp)
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_paths["report"], index_label="class")
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_paths["confusion_csv"], index_label="actual")
    pd.DataFrame(history).to_csv(output_paths["history_csv"], index=False)

    show = args.show_plots
    graficar_matrices_confusion(
        [matrix], [rep.upper()], list(CLASS_NAMES),
        titulo=f"RNA — matriz de confusión ({rep.upper()})",
        save_path=output_paths["confusion_png"], show=show,
    )
    graficar_historial_rna(
        history,
        f"RNA — entrenamiento y validación ({rep.upper()})",
        output_paths["curves_png"],
        show=show,
    )
    _atomic_torch_save({
        "model_state_dict": best_state,
        "representation": rep,
        "input_dim": int(features.shape[1]),
        "original_input_dim": int(original_input_dim),
        "selected_features": selected_features,
    }, output_paths["model"])

    metadata = {
        "version": 1,
        "run_id": run_id,
        "representation": rep,
        "scaled": scale,
        "phase": phase,
        "input_signature": signature,
        "parameters": parameters,
        "device": str(device),
        "amp": use_amp,
        "rows": int(len(labels)),
        "input_dim": int(features.shape[1]),
        "original_input_dim": int(original_input_dim),
        "selected_feature_count": int(len(selected_features)) if selected_features is not None else None,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test_loss": test_loss,
        "test_macro_f1": test_f1,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "best_model": str(output_paths["model"]),
    }
    _atomic_json(output_paths["metadata"], metadata)
    _atomic_json(output_paths["complete"], {
        "version": 1,
        "run_id": run_id,
        "representation": rep,
        "outputs": {key: str(path) for key, path in output_paths.items() if key != "complete"},
        "test_macro_f1": test_f1,
    })
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    print(
        f"[{rep}] test F1 macro={test_f1:.4f}, accuracy={accuracy_score(y_test, y_pred):.4f}; "
        f"mejor época={best_epoch}; modelo={output_paths['model']}",
        flush=True,
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entrena MLPs de sentimiento sobre embeddings densos y dispersos.")
    agregar_argumento_limit(parser)
    parser.add_argument("--embedding", choices=("idf-bigram", "idf-monogram", "bow", "fasttext", "all"), default="all", help="Representación; all ejecuta BOW, TF-IDF monograma/bigrama y FastText.")
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--scale", dest="scale", action="store_true", help="Evalúa solo con escalado.")
    scaling.add_argument("--no-scale", dest="scale", action="store_false", help="Evalúa solo sin escalado.")
    parser.set_defaults(scale=None)
    parser.add_argument("--epochs", type=int, default=30, help="Máximo de épocas por representación.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Filas por minibatch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate de AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Regularización L2 de AdamW.")
    parser.add_argument("--hidden-dims", type=_parse_hidden_dims, default=(256, 128), help="Capas ocultas, ej. 256,128.")
    parser.add_argument("--max-features", type=int, default=100_000, help="Tope de columnas dispersas; si se supera, selecciona por chi2 usando solo train.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout entre capas ocultas.")
    parser.add_argument("--patience", type=int, default=5, help="Épocas sin mejora antes de detenerse.")
    parser.add_argument("--optimization-trials", type=int, default=20, help="Cantidad máxima de pruebas Optuna por embedding y escalado.")
    parser.add_argument("--optimization-timeout", type=int, default=None, help="Tiempo máximo en segundos para cada búsqueda Optuna.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de entrenamiento y validación.")
    parser.add_argument("--device", type=str.lower, choices=("auto", "cuda", "cpu"), default="auto", help="Dispositivo de entrenamiento.")
    parser.add_argument("--num-workers", type=int, default=0, help="Procesos para preparar minibatches (en Windows se fuerza 0 para evitar copiar matrices grandes entre procesos).")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision cuando se usa CUDA.")
    parser.add_argument("--overwrite", action="store_true", help="Regenera reportes y modelos; reanuda checkpoints compatibles.")
    parser.add_argument("--restart", action="store_true", help="Descarta el checkpoint compatible y entrena desde cero.")
    parser.add_argument("--show-plots", action="store_true", help="Muestra las gráficas además de guardarlas.")
    args = parser.parse_args(argv)
    args.limit = obtener_limit(args)

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.num_workers < 0 or args.optimization_trials < 1:
        parser.error("epochs, batch-size y patience deben ser positivos; num-workers no puede ser negativo.")
    if args.optimization_timeout is not None and args.optimization_timeout < 1:
        parser.error("optimization-timeout debe ser positivo.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("learning-rate debe ser positivo, weight-decay no negativo y dropout entre 0 y 1.")
    if args.max_features < 1:
        parser.error("max-features debe ser positivo.")
    if args.restart and not args.overwrite:
        args.overwrite = True
    if os.name == "nt" and args.num_workers > 0:
        print("Windows: se usará num-workers=0 para no serializar matrices grandes a procesos hijos.", flush=True)
        args.num_workers = 0

    try:
        device = _select_device(args.device)
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            print(
                f"RNA usará {torch.cuda.get_device_name(device)} "
                f"({properties.total_memory / (1024 ** 3):.1f} GB VRAM).",
                flush=True,
            )
        else:
            print("RNA usará CPU.", flush=True)
        labels = ejecutar_etapa(
            "RNA: cargando etiquetas",
            load_sentiment_labels,
            config.dataset_clean,
            limit=args.limit,
            as_indices=True,
        )
        paths = {
            "idf-bigram": config.idf_bigram, "idf-monogram": config.idf_monogram,
            "bow": config.bow_vect, "fasttext": config.fasttext_embeddings,
        }
        four_embeddings = ("bow", "idf-monogram", "idf-bigram", "fasttext")
        representations = four_embeddings if args.embedding == "all" else (args.embedding,)
        missing = [str(config.ruta_limitada(paths[name], args.limit)) for name in representations
                   if not config.ruta_limitada(paths[name], args.limit).is_file()]
        if missing:
            raise FileNotFoundError("Faltan matrices de embeddings:\n" + "\n".join(missing))

        results = []
        scale_modes = (False, True) if args.scale is None else (args.scale,)
        for rep in representations:
            matrix_path = config.ruta_limitada(paths[rep], args.limit)
            matrix = ejecutar_etapa("RNA: cargando matriz " + rep.upper(), load_npz, matrix_path)
            if matrix.shape[0] != len(labels):
                raise ValueError(f"{rep} tiene {matrix.shape[0]} filas pero hay {len(labels)} etiquetas; comprueba --limit.")
            for scale in scale_modes:
                results.append(ejecutar_etapa(
                    f"RNA: baseline {rep.upper()} {'con' if scale else 'sin'} escalado",
                    _fit_representation, rep, matrix, labels, matrix_path, args, device,
                    scale=scale, phase="baseline",
                ))

                study_dir = config.RNA_RESULTS / "optuna"
                study_dir.mkdir(parents=True, exist_ok=True)
                study_material = [
                    _input_signature(rep, matrix_path), scale, args.seed, args.epochs,
                    args.batch_size, args.patience, args.amp,
                ]
                if matrix.shape[1] > args.max_features:
                    study_material.append({"method": "chi2", "max_features": args.max_features, "version": 1})
                study_key = hashlib.sha256(
                    json.dumps(study_material, sort_keys=True).encode()
                ).hexdigest()[:20]
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=args.seed),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
                    study_name=f"rna_{rep}_{'scaled' if scale else 'unscaled'}",
                    storage=f"sqlite:///{(study_dir / f'rna_{study_key}.db').as_posix()}",
                    load_if_exists=True,
                )

                def objective(trial):
                    trial_args = copy.copy(args)
                    trial_args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
                    trial_args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
                    trial_args.dropout = trial.suggest_float("dropout", 0.0, 0.6)
                    hidden_options = {
                        "128,64": (128, 64),
                        "256,128": (256, 128),
                        "512,256": (512, 256),
                    }
                    hidden_choice = trial.suggest_categorical("hidden_dims", list(hidden_options))
                    trial_args.hidden_dims = hidden_options[hidden_choice]
                    outcome = _fit_representation(
                        rep, matrix, labels, matrix_path, trial_args, device,
                        scale=scale, phase=f"trial_{trial.number}", trial=trial,
                        save_artifacts=False,
                    )
                    return outcome["best_validation_macro_f1"]

                ejecutar_etapa(
                    f"RNA: optimización Optuna {rep.upper()} {'con' if scale else 'sin'} escalado",
                    study.optimize,
                    objective,
                    n_trials=max(0, args.optimization_trials - len(study.trials)),
                    timeout=args.optimization_timeout,
                    gc_after_trial=True,
                    show_progress_bar=False,
                )
                if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
                    raise RuntimeError(f"Optuna no completó ningún trial para {rep} con scale={scale}.")
                best_args = copy.copy(args)
                best_args.learning_rate = study.best_params["learning_rate"]
                best_args.weight_decay = study.best_params["weight_decay"]
                best_args.dropout = study.best_params["dropout"]
                best_args.hidden_dims = {
                    "128,64": (128, 64), "256,128": (256, 128), "512,256": (512, 256),
                }[study.best_params["hidden_dims"]]
                optimized = ejecutar_etapa(
                    f"RNA: reentrenando mejor configuración {rep.upper()}",
                    _fit_representation, rep, matrix, labels, matrix_path, best_args, device,
                    scale=scale, phase="optimized",
                )
                optimized["optimization_best_params"] = study.best_params
                results.append(optimized)
            del matrix

        summary = pd.DataFrame(results)
        summary_path = config.ruta_limitada(config.RNA_RESULTS / "rna_summary.csv", args.limit)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Resumen guardado: {summary_path}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Entrena MLPs de sentimiento sobre matrices densas BERT y ELMo."""

import argparse
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from utils import agregar_argumento_limit, ejecutar_etapa, obtener_limit
from .data import CLASS_NAMES, load_dense_embeddings, load_sentiment_labels
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
        return self.network(features)


class _IndexedDataset(Dataset):
    """Vista indexada de las filas, para no copiar train/valid/test completos."""

    def __init__(self, features, labels, indices):
        self.features = features
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        return torch.from_numpy(self.features[row]), int(self.labels[row])


def _parse_hidden_dims(value):
    try:
        dimensions = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use dimensiones enteras separadas por comas, ej. 256,128.") from error
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("Las dimensiones ocultas deben ser enteros positivos.")
    return dimensions


def _select_device(requested):
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
    }
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20], parameters


def _artifact_paths(rep, limit):
    suffix = f"_limit_{limit}" if limit is not None else ""
    report_dir = config.CLASSIFICATION_REPORTS
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


def _make_loader(features, labels, indices, *, batch_size, shuffle, device, generator, num_workers):
    dataset = _IndexedDataset(features, labels, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
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


def _fit_representation(rep, features, labels, matrix_path, args, device):
    if features.shape[0] != len(labels):
        raise ValueError(
            f"BERT/ELMo tiene {features.shape[0]} filas pero hay {len(labels)} etiquetas. "
            "Comprueba que ambas salidas usan el mismo --limit y orden de dataset."
        )

    signature = _input_signature(rep, matrix_path)
    run_id, parameters = _run_id(signature, args)
    output_paths = _artifact_paths(rep, args.limit)
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

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    features_tensor = features
    train_loader = _make_loader(
        features_tensor, labels, train_indices,
        batch_size=args.batch_size, shuffle=True, device=device,
        generator=generator, num_workers=args.num_workers,
    )
    train_eval_loader = _make_loader(
        features_tensor, labels, train_indices,
        batch_size=args.batch_size, shuffle=False, device=device,
        generator=generator, num_workers=args.num_workers,
    )
    val_loader = _make_loader(
        features_tensor, labels, val_indices,
        batch_size=args.batch_size, shuffle=False, device=device,
        generator=generator, num_workers=args.num_workers,
    )
    test_loader = _make_loader(
        features_tensor, labels, test_indices,
        batch_size=args.batch_size, shuffle=False, device=device,
        generator=generator, num_workers=args.num_workers,
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

    if latest_path.exists() and not args.restart:
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
    if best_path.exists() and not args.restart:
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
                _atomic_torch_save({"version": 1, "model_state_dict": best_state, "epoch": best_epoch, "val_macro_f1": best_f1}, best_path)
            else:
                bad_epochs += 1

            _save_checkpoint(
                latest_path, model=model, optimizer=optimizer, scaler=scaler,
                epoch=epoch + 1, best_f1=best_f1, best_epoch=best_epoch,
                bad_epochs=bad_epochs, history=history, generator=generator,
                completed=bad_epochs >= args.patience or epoch + 1 >= args.epochs,
            )
            print(
                f"[{rep}] época {epoch + 1}/{args.epochs} | "
                f"loss {train_loss:.4f}/{val_loss:.4f} | "
                f"F1 macro {train_f1:.4f}/{val_f1:.4f} | "
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

        _save_checkpoint(
            latest_path, model=model, optimizer=optimizer, scaler=scaler,
            epoch=len(history["epoch"]), best_f1=best_f1, best_epoch=best_epoch,
            bad_epochs=bad_epochs, history=history, generator=generator,
            completed=True,
        )

    model.load_state_dict(best_state)
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
    _atomic_torch_save({"model_state_dict": best_state, "representation": rep, "input_dim": int(features.shape[1])}, output_paths["model"])

    metadata = {
        "version": 1,
        "run_id": run_id,
        "representation": rep,
        "input_signature": signature,
        "parameters": parameters,
        "device": str(device),
        "amp": use_amp,
        "rows": int(len(labels)),
        "input_dim": int(features.shape[1]),
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
    print(
        f"[{rep}] test F1 macro={test_f1:.4f}, accuracy={accuracy_score(y_test, y_pred):.4f}; "
        f"mejor época={best_epoch}; modelo={output_paths['model']}",
        flush=True,
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entrena MLPs de sentimiento sobre embeddings densos.")
    agregar_argumento_limit(parser)
    parser.add_argument("--embedding", choices=("bert", "elmo", "both"), default="both", help="Representación a entrenar.")
    parser.add_argument("--epochs", type=int, default=30, help="Máximo de épocas por representación.")
    parser.add_argument("--batch-size", type=int, default=512, help="Filas por minibatch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate de AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Regularización L2 de AdamW.")
    parser.add_argument("--hidden-dims", type=_parse_hidden_dims, default=(256, 128), help="Capas ocultas, ej. 256,128.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout entre capas ocultas.")
    parser.add_argument("--patience", type=int, default=5, help="Épocas sin mejora antes de detenerse.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de entrenamiento y validación.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Dispositivo de entrenamiento.")
    parser.add_argument("--num-workers", type=int, default=0, help="Procesos de carga por minibatch.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision cuando se usa CUDA.")
    parser.add_argument("--overwrite", action="store_true", help="Regenera reportes y modelos; reanuda checkpoints compatibles.")
    parser.add_argument("--restart", action="store_true", help="Descarta el checkpoint compatible y entrena desde cero.")
    parser.add_argument("--show-plots", action="store_true", help="Muestra las gráficas además de guardarlas.")
    args = parser.parse_args(argv)
    args.limit = obtener_limit(args)

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.num_workers < 0:
        parser.error("epochs, batch-size y patience deben ser positivos; num-workers no puede ser negativo.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("learning-rate debe ser positivo, weight-decay no negativo y dropout entre 0 y 1.")
    if args.restart and not args.overwrite:
        args.overwrite = True

    try:
        device = _select_device(args.device)
        labels = ejecutar_etapa(
            "RNA: cargando etiquetas",
            load_sentiment_labels,
            config.dataset_clean,
            limit=args.limit,
            as_indices=True,
        )
        representations = ("bert", "elmo") if args.embedding == "both" else (args.embedding,)
        missing = [
            str(config.ruta_limitada(path, args.limit))
            for name, path in (("bert", config.bert_vect), ("elmo", config.elmo_vect))
            if name in representations and not config.ruta_limitada(path, args.limit).is_file()
        ]
        if missing:
            raise FileNotFoundError("Faltan matrices de embeddings:\n" + "\n".join(missing))

        results = []
        for rep in representations:
            matrix, matrix_path = ejecutar_etapa(
                f"RNA: cargando matriz {rep.upper()}",
                load_dense_embeddings,
                rep,
                limit=args.limit,
            )
            results.append(ejecutar_etapa(
                f"RNA: entrenando {rep.upper()}",
                _fit_representation,
                rep,
                matrix,
                labels,
                matrix_path,
                args,
                device,
            ))
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

# Quellen zum Projekt „MNIST Fault Injection“

Stand: 28.10.2026

## 1. Entwicklungsumgebung und Python-Grundlagen

| ID | Quelle | Verwendung im Projekt |
| --- | --- | --- |
| Q01 | Python Software Foundation: [venv — Creation of virtual environments, Python 3.10](https://docs.python.org/3.10/library/venv.html) | Virtuelle Umgebung `.venv`, Aktivierung und getrennte Paketinstallation. |
| Q02 | PyTorch: [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/) | Zusammenpassende Versionen von `torch` und `torchvision` sowie CPU- und CUDA-Installationsvarianten für das Einrichtungsskript. |
| Q03 | Python Software Foundation: [pathlib — Object-oriented filesystem paths, Python 3.10](https://docs.python.org/3.10/library/pathlib.html) | `Path`, `resolve()`, `parents`, Zusammensetzen von Pfaden und Erstellen des Checkpoint-Ordners mit `mkdir()`. |
| Q04 | Python Software Foundation: [Modules, Python 3.10](https://docs.python.org/3.10/tutorial/modules.html) | Imports zwischen den Projektdateien, Python-Pakete, `__name__ == "__main__"` und Ausführen mit `python -m`. |
| Q05 | PyTorch: [torch.cuda.is_available, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.is_available.html) | Auswahl von CUDA oder CPU bei der Programmausführung. |

## 2. MNIST, Vorverarbeitung und Datenaufteilung

| ID | Quelle | Verwendung im Projekt |
| --- | --- | --- |
| Q06 | torchvision: [MNIST, Version 0.24](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.MNIST.html) | `datasets.MNIST`, Parameter `root`, `train`, `download` und `transform`; Abrufen von Bild und Label. |
| Q07 | torchvision: [ToTensor, Version 0.24](https://docs.pytorch.org/vision/0.24/generated/torchvision.transforms.ToTensor.html) | Umwandeln der MNIST-Bilder in Tensoren mit Kanal-, Höhen- und Breitendimension sowie Skalierung der Bildwerte auf den Bereich von 0 bis 1. |
| Q08 | PyTorch: [torch.utils.data, Version 2.9](https://docs.pytorch.org/docs/2.9/data.html) | `DataLoader`, Batchgröße, Mischen der Trainingsdaten und `random_split` für die Trainings- und Validierungsaufteilung. |
| Q09 | PyTorch: [Reproducibility, Version 2.9](https://docs.pytorch.org/docs/2.9/notes/randomness.html) | Festlegen von Seeds und Einordnen der Grenzen der Wiederholbarkeit, insbesondere zwischen Geräten und Versionen. |

## 3. CNN und PyTorch-Schichten

| ID | Quelle | Verwendung im Projekt |
| --- | --- | --- |
| Q10 | PyTorch: [nn.Module, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html) | Basisklasse von `MNIST_CNN`, Initialisierung mit `super().__init__()`, Registrieren der Schichten, `forward`, Parameterverwaltung, Gerätewechsel sowie Trainings- und Auswertungsmodus. |
| Q11 | PyTorch: [Conv2d, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html) | Faltungsschichten, Ein- und Ausgabekanäle, Filtergröße, Padding, Gewichtsformen und Berechnung der Ausgabegrößen. |
| Q12 | PyTorch: [ReLU, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.ReLU.html) | Elementweise Aktivierungsfunktion `max(0, x)`. |
| Q13 | PyTorch: [MaxPool2d, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.MaxPool2d.html) | Max-Pooling über 2 × 2 große Fenster zur Reduktion der räumlichen Auflösung. |
| Q14 | PyTorch: [Linear, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html) | Klassifikationsschicht mit 784 Eingabewerten und zehn Ausgabewerten. |
| Q15 | PyTorch: [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | Offizielles Beispiel für das Zusammenspiel von Modell, `nn.Flatten`, Verlustfunktion, Trainingsschleife und Evaluation. |

## 4. Training, Evaluation und Checkpoints

| ID | Quelle | Verwendung im Projekt |
| --- | --- | --- |
| Q16 | PyTorch: [CrossEntropyLoss, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html) | Verlustberechnung aus Logits und richtigen Klassenlabels; keine vorgeschaltete Softmax-Schicht im Modell. |
| Q17 | PyTorch: [Adam, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html) | Optimierer und Bedeutung des Lernratenparameters `lr`. |
| Q18 | PyTorch: [no_grad, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.no_grad.html) | Deaktivieren der Gradientenaufzeichnung bei Evaluation und Inferenztests. |
| Q19 | PyTorch: [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | Speichern und Laden von `state_dict`, Checkpoint-Dictionaries, `map_location`, `weights_only=True` und anschließendes Umschalten in den Auswertungsmodus. |

Die Reihenfolge `zero_grad()`, Vorwärtsrechnung, Verlustberechnung, `backward()` und `step()` orientiert sich an Q15. Die Verwendung von Adam und die Lernrate `0.001` sind unsere konkrete Konfiguration.

## 5. PyTorchFI und Beobachtung der Aktivierungen

| ID | Quelle | Verwendung im Projekt |
| --- | --- | --- |
| Q20 | PyTorchFI: [Veröffentlichung 0.6.0 auf PyPI einschließlich Paketdateien](https://pypi.org/project/pytorchfi/0.6.0/#files) | Verwendete PyTorchFI-Version; maßgeblicher Quellcode in `pytorchfi/core.py` innerhalb des veröffentlichten Pakets. |
| Q21 | PyTorch: [nn.Module.register_forward_hook, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) | Beobachten der Schichtausgaben, Hook-Argumente, Ausführungsreihenfolge und Entfernen der Beobachtungs-Hooks über Handles. |
| Q22 | PyTorch: [Tensor.detach, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.detach.html) | Lösen der beobachteten Aktivierungen von der Verbindung zur Gradientenberechnung. |
| Q23 | PyTorch: [torch.clone, Version 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.clone.html) | Unabhängige Kopie der Aktivierungsdaten für den Vergleich von Referenz- und Fehlerlauf. |
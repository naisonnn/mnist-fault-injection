# Fehlerinjektion mit MNIST

Dieses Projektseminar an der HTW Dresden untersucht, wie sich simulierte Fehler in neuronalen Netzen auf die Erkennung handgeschriebener Ziffern auswirken. Dafür vergleichen wir zwei Modelle auf demselben MNIST-Datensatz:

- **Convolutional Neural Network (CNN):** Verarbeitet die Bilder mit Faltungsschichten und nutzt dabei die räumliche Anordnung der Pixel.
- **Multilayer Perceptron (MLP):** Verarbeitet dieselben Bilder als Vektoren aus jeweils 784 Pixelwerten in voll verbundenen Schichten.

Für beide Modelle verwenden wir dieselbe Datenaufteilung und dieselben Auswertungsmetriken. Mit PyTorchFI verändern wir während der Vorwärtsrechnung gezielt einzelne Zwischenwerte. Für jedes Modell vergleichen wir die Ergebnisse mit einem Lauf ohne Injektion. So lässt sich untersuchen, wie häufig sich einzelne Vorhersagen ändern und wie sich die Erkennungsgenauigkeit entwickelt.

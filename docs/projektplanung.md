# Projektplan: MNIST Fault Injection

**Team:** Vincent Dalchau, Nathan Richter  
**Projektdauer:** 10 Wochen  
**Technologien:** Python, PyTorch, torchvision, PyTorchFI, Pandas, Matplotlib  
**Stand:** 22.10.2026

## 1. Projektziel und Forschungsfrage

Ziel ist die experimentelle Untersuchung, wie simulierte Bitfehler in den Zwischenergebnissen eines kleinen Convolutional Neural Networks (CNN) dessen Klassifikationen beeinflussen. Im Mittelpunkt steht außerdem die Frage, ob Training unter Fehlerinjektion die Robustheit bei späterer Inferenz verbessert.

Als Testdatensatz, auf dem das CNN klassifiziert, verwenden wir den MNIST-Datensatz (Modified National Institute of Standards and Technology database), welcher aus 70.000 handgeschriebenen Ziffern als Graustufen-Bildern besteht.

Wir verwenden eine kleine CNN-Vorlage aus normalen PyTorch-Schichten, passen sie an MNIST an und trainieren sie selbst. Die eigene Entwicklungsarbeit konzentriert sich auf Trainingsablauf, Fehlerinjektion, Experimentsteuerung und Auswertung. Vortrainierte Gewichte werden für das Kernprojekt nicht benötigt.

**Zentrale Forschungsfrage:**  
Wie beeinflussen Bitfehler in unterschiedlichen CNN-Schichten die Ziffernerkennung, und verändert Training unter solchen Fehlern die Robustheit des Modells bei der Inferenz?

Daraus ergeben sich vier Teilfragen:

1. Wie unterscheiden sich die Auswirkungen von Fehlern in der ersten und zweiten Faltungsschicht?
2. Welchen Einfluss haben Bitbereich und Injektionsrate?
3. Ist ein unter Fehlerinjektion trainiertes Modell gegenüber einem normal trainierten Modell robuster, und wie verändert sich seine Genauigkeit ohne Fehler?
4. Wie entwickeln sich Genauigkeit und Robustheit mit zunehmender Trainingsdauer?

Eine Verbesserung durch Training unter Fehlern ist eine zu prüfende Hypothese. Auch unveränderte Robustheit, schlechtere Genauigkeit oder instabiles Training sind aussagekräftige Ergebnisse.

## 2. Modell und Datengrundlage

### CNN aus Standardbausteinen

Als Startpunkt dient folgende kleine Architektur. Die Kanalzahlen werden nach einem kurzen Pilotversuch festgelegt und anschließend für alle Vergleiche beibehalten.

| Schritt | PyTorch-Baustein | Ausgabe pro Bild |
|---|---|---|
| Eingabe | MNIST-Graustufenbild | 1 × 28 × 28 |
| Erste Faltung | nn.Conv2d(1, 8, kernel_size=3, padding=1) | 8 × 28 × 28 |
| Aktivierung und Pooling | nn.ReLU(), nn.MaxPool2d(2) | 8 × 14 × 14 |
| Zweite Faltung | nn.Conv2d(8, 16, kernel_size=3, padding=1) | 16 × 14 × 14 |
| Aktivierung und Pooling | nn.ReLU(), nn.MaxPool2d(2) | 16 × 7 × 7 |
| Zusammenfassen | nn.Flatten() | 784 |
| Klassifikation | nn.Linear(784, 10) | 10 Klassenwerte |

Die Klassifikationsschicht liefert Logits für die Ziffern 0 bis 9. Als Verlustfunktion wird nn.CrossEntropyLoss verwendet. Dropout und Batch-Normalisierung werden im ersten Aufbau weggelassen, um den Versuchsaufbau überschaubar zu halten.

### Aufteilung der Daten

- Die 60.000 MNIST-Trainingsbilder werden mit festem Seed in 50.000 Trainings- und 10.000 Validierungsbilder aufgeteilt.
- Die 10.000 offiziellen Testbilder bleiben für die abschließende Auswertung reserviert.
- Architektur, Lernrate, Trainingsdauer und Injektionsparameter werden mit Trainings- und Validierungsdaten festgelegt.
- Für umfangreiche Injektionsexperimente wird vorab eine feste, nach Ziffern ausgewogene Teilmenge des Testsets definiert, zunächst 1.000 Bilder.
- Vorverarbeitung und Datenauswahl sind für beide Trainingsvarianten identisch.

## 3. Fehlermodell und Integration

### Was wird injiziert?

Untersucht werden transiente Fehler in Float32-Zwischenergebnissen: Ein einzelnes Bit eines einzelnen Wertes wird während eines Vorwärtsdurchlaufs umgedreht.

Die beiden primären Injektionspunkte sind die Ausgaben der Faltungsschichten, jeweils **vor der nachfolgenden ReLU-Aktivierung**. So ist die Position des Eingriffs eindeutig festgelegt.

| Eigenschaft | Festlegung |
|---|---|
| Fehlerobjekt | Ein skalarer Wert in der Ausgabe einer Faltungsschicht |
| Fehlerwirkung | Ein Bitflip im aktuellen Vorwärtsdurchlauf |
| Schichten | Erste und zweite nn.Conv2d-Schicht |
| Datentyp | Float32, ohne Mixed Precision |
| Bitbereiche | Mantisse und Exponent, getrennt ausgewertet |
| Unveränderte Größen | Eingabebild, Klassenlabel und gespeicherte Modellgewichte beim reinen Inferenzexperiment |
| Ausführungsphase | Vorwärtsrechnung während Training oder Inferenz |

Bei der Bitnummerierung vom niederwertigsten Bit aus umfasst die Mantisse die Bits 0–22, der Exponent 23–30 und das Vorzeichen Bit 31. Das Vorzeichenbit ist eine mögliche spätere Erweiterung.

**Definition der Injektionsrate:** Die Wahrscheinlichkeit p beschreibt, ob ein bestimmtes Bild bei einem Vorwärtsdurchlauf eine Injektion erhält. Falls ja, wird genau ein Wert in genau einer ausgewählten Schicht mit genau einem Bitflip verändert. p = 0,1 bedeutet somit ungefähr ein betroffenes Bild unter zehn Bildern, unabhängig von der Batchgröße.

Beim Fehlertraining wird die Zielschicht zufällig und gleichverteilt aus den beiden Faltungsschichten gewählt. Ebenso wird zunächst einer der beiden Bitbereiche gleichverteilt ausgewählt und anschließend eine Bitposition innerhalb dieses Bereichs. Die Werteposition wird gleichverteilt aus dem Ausgabetensor des jeweiligen Bildes gezogen. Bei der Auswertung werden Zielschicht und Bitbereich gezielt vorgegeben.

Ein Fehler in einem Zwischenergebnis wird beim nächsten Durchlauf neu ausgewählt. Während des Trainings können seine Auswirkungen über die anschließende Gewichtsaktualisierung dauerhaft in das Modell eingehen.

### Technischer Machbarkeitstest

PyTorchFI wird frühzeitig an genau dieser Architektur und einer festgehaltenen Version erprobt. Bereits in Woche 1 werden sowohl eine Inferenzinjektion als auch ein Trainingsschritt unter Injektion ausprobiert.

Besonders zu prüfen sind:

1. Der injizierte Wert besitzt tatsächlich das gewünschte geänderte Bitmuster; die übrigen Werte bleiben unverändert.
2. Bei deaktivierter Injektion entspricht das Ergebnis dem normalen Modell.
3. Hooks und Injektionszustände sammeln sich zwischen Durchläufen nicht an.
4. Optimierer und Checkpoint beziehen sich auf das tatsächlich ausgeführte Modell. Bei Modellkopien durch PyTorchFI darf der Optimierer nicht weiterhin ein anderes Parameterobjekt aktualisieren.
5. Gradienten und Gewichtsaktualisierungen funktionieren auch unter Injektion; Optimiererzustände bleiben über Trainingsschritte hinweg erhalten.

Ein Bitflip ist keine gewöhnliche differenzierbare Operation. Für das Fehlertraining wird die Vorwärtsrechnung mit dem veränderten Wert ausgeführt; lokal an der Bitmanipulation wird eine Identitäts-Ersatzableitung verwendet. Diese Annahme wird ausdrücklich dokumentiert. Die übrigen Ableitungen berechnet PyTorch. Fehler in Gradienten, Optimiererzuständen oder der Rückwärtsrechnung selbst werden nicht zusätzlich injiziert.

Falls die gewählte PyTorchFI-Version Anpassungen benötigt, wird eine kleine, dokumentierte Erweiterung der Injektionsfunktion vorgenommen. Ein Wechsel auf reine PyTorch-Hooks wäre eine begründete Ersatzlösung und müsste als methodische Änderung ausgewiesen werden.

## 4. Versuchsdesign

### Vergleich der Trainingsvarianten

Es werden zwei Modelle derselben Architektur miteinander verglichen:

- **Modell A:** Training ohne Fehlerinjektion.
- **Modell B:** Training mit Fehlerinjektion in der Vorwärtsrechnung.

Beide Modelle starten je Wiederholung mit denselben initialen Gewichten und verwenden dieselbe Reihenfolge der Trainingsbilder, dieselben Hyperparameter und gleich viele Trainingsschritte. Modell B erhält zusätzlich die definierten Injektionen. Datenreihenfolge, Initialisierung und Injektion verwenden getrennte Zufallsgeneratoren.

| Trainingsvariante | Auswertung ohne Injektion | Auswertung mit Injektion |
|---|---|---|
| Modell A: normales Training | Referenzgenauigkeit | Robustheit ohne Fehlertraining |
| Modell B: Fehlertraining | Einfluss auf normale Erkennungsleistung | Robustheit nach Fehlertraining |

Die Bewertung berücksichtigt beide Spalten: Eine bessere Genauigkeit unter Fehlern allein reicht nicht aus, wenn die Genauigkeit ohne Fehler stark sinkt.

### Gestufter Experimentumfang

Alle Angaben sind Startwerte für den Pilotversuch. Die endgültige Konfiguration wird vor der Testauswertung dokumentiert und eingefroren.

| Untersuchung | Geplanter Umfang |
|---|---|
| Trainingswiederholungen | Drei unabhängige Seeds, je Seed ein gepaartes Modell A und B |
| Trainingsdauer | Zunächst fünf Epochen; Checkpoints nach Epoche 1, 3 und 5 |
| Injektionsrate beim Training | Für A: 0; für B zunächst 0,1 |
| Hauptvergleich | Beide Modelle × zwei Schichten × zwei Bitbereiche, bei p = 1 |
| Ergänzung Injektionsrate | p = 0,1; 0,5; 1 an der zweiten Schicht für Exponentenfehler |
| Ergänzung Trainingsdauer | Checkpoints 1, 3, 5 bei derselben festgelegten Fehlerkonfiguration |
| Genauigkeit ohne Injektion | Abschließende Modelle auf dem gesamten Testset |
| Injektionsauswertung | Zunächst mindestens 1.000 Bilder je Konfiguration und Trainingsseed |

Der Hauptvergleich umfasst acht Konfigurationen pro Trainingsseed, somit 24 Konfigurationen und zunächst 24.000 Bilddurchläufe unter Injektion. Die Untersuchungen von Rate und Trainingsdauer ergänzen einzelne gezielte Vergleiche; es wird nicht jede Kombination aller Faktoren durchgerechnet.

Für Modell A und B wird beim Vergleich derselbe vorbereitete Injektionsplan verwendet: identische Bilder, Schichten, Wertekoordinaten und Bitpositionen. Die Zahlenwerte vor dem Eingriff können sich durch das unterschiedliche Training unterscheiden. Injektionen aus dem Training werden bei der Testauswertung nicht wiederverwendet.

Bei seltenen Effekten wird die Stichprobe nach einem vorab festgelegten Budget erweitert. Ein Ergebnis ohne beobachtete Fehlklassifikation wird mit der Zahl der geprüften Injektionen berichtet und nicht als allgemeiner Nachweis von Fehlerfreiheit gewertet.

## 5. Meilensteinplanung

### Meilenstein 1 – Referenzmodell und Machbarkeit, Woche 1–2

**Ziel:** Das CNN lässt sich trainieren; die gewählte PyTorchFI-Anbindung ist grundsätzlich nutzbar.

**Aufgaben:**

- Entwicklungsumgebung einrichten und Paketversionen festhalten.
- CNN-Vorlage übernehmen, an MNIST anpassen und Datenaufteilung einrichten.
- Training, Evaluation und Checkpoints bereitstellen.
- Bereits in Woche 1 eine Inferenzinjektion und einen Trainingsschritt unter Injektion ausprobieren.
- Ein erstes Modell ohne Fehler trainieren und Lernkurven protokollieren.

**Erfolgskriterium:** Das Referenzmodell erreicht als Arbeitsziel mindestens 95 % Validierungsgenauigkeit. Eine Injektion ist nachgewiesen; ein endlicher Trainingsschritt unter Injektion aktualisiert die vorgesehenen Parameter.

**Ergebnis:** Ausführbare Referenzpipeline und dokumentierter Machbarkeitsnachweis.

### Meilenstein 2 – Kontrollierte Fehlerinjektion, Woche 3–4

**Ziel:** Dasselbe definierte Fehlermodell ist für Inferenz und Training einsetzbar.

**Aufgaben:**

- Bitflip-Funktion für Float32 und beide Injektionsschichten fertigstellen.
- Injektionsrate, Koordinatenauswahl und Bitbereich konfigurierbar machen.
- Korrektes Bitmuster und unveränderte Referenzläufe überprüfen.
- Gradientenbehandlung, Modellkopien, Optimiererzustände und Hook-Lebensdauer prüfen.
- Nicht endliche Werte und Ausführungsfehler protokollieren.

**Erfolgskriterium:** Gezielte Injektionen sind reproduzierbar. Mehrere Trainingsschritte funktionieren mit dem vorgesehenen Modell und fortgeführtem Optimiererzustand; deaktivierte Injektion verändert den normalen Ablauf nicht.

**Ergebnis:** Validiertes Fehlerinjektionsmodul einschließlich Trainingsanbindung.

### Meilenstein 3 – Automatisierung und Vergleichsmodelle, Woche 5–6

**Ziel:** Trainingsvarianten und Auswertung lassen sich ohne manuelle Eingriffe ausführen.

**Aufgaben:**

- Gemeinsame Konfigurationsdateien für Training und Experimente bereitstellen.
- Pilotversuche auf Validierungsdaten durchführen und Umfang festlegen.
- Gepaarte Trainingsläufe für Modell A und B mit drei Seeds ausführen.
- Checkpoints nach festgelegten Epochen speichern.
- Wiederverwendbare Injektionspläne und strukturierte Ergebnisprotokolle erzeugen.

**Erfolgskriterium:** Ein automatisierter Ablauf erzeugt beide Trainingsvarianten, ihre Checkpoints und vergleichbare Auswertungsergebnisse. Alle Konfigurationen und Zufalls-Seeds sind nachvollziehbar gespeichert.

**Ergebnis:** Trainierte Vergleichsmodelle und automatisiertes Experimentframework.

### Meilenstein 4 – Messungen und Auswertung, Woche 7–8

**Ziel:** Die Forschungsfragen werden anhand kontrollierter Experimente untersucht.

**Aufgaben:**

- Hauptvergleich für zwei Schichten und zwei Bitbereiche durchführen.
- Ausgewählte Vergleiche von Injektionsrate und Trainingsdauer ergänzen.
- Genauigkeit mit und ohne Injektion sowie neu entstandene Fehler auswerten.
- Ungültige Ausgaben und Trainingsabbrüche gesondert analysieren.
- Unterschiede über die Trainingsseeds und Unsicherheit der Messungen darstellen.

**Erfolgskriterium:** Beide Trainingsvarianten sind unter denselben Testbedingungen verglichen. Messzahlen, Streuung und Einschränkungen erlauben eine nachvollziehbare Antwort auf die Forschungsfragen.

**Ergebnis:** Rohdaten, Auswertungsskripte, Diagramme und erste Interpretation.

### Meilenstein 5 – Dokumentation und Abschluss, Woche 9–10

**Ziel:** Versuchsaufbau, Ergebnisse und Grenzen sind verständlich und reproduzierbar dokumentiert.

**Aufgaben:**

- Architektur, Fehlermodell, Trainingsablauf und Gradientenannahme beschreiben.
- Ergebnisse interpretieren und Forschungsfragen beantworten.
- Aufwand, Fehlerquellen und Übertragbarkeit diskutieren.
- Repository mit Ausführungsanleitung, Konfigurationen und Ergebnisübersicht vervollständigen.
- Abschlussbericht und Präsentation erstellen.

**Erfolgskriterium:** Eine andere Person kann Referenztraining und ein definiertes Injektionsexperiment anhand der Anleitung wiederholen. Schlussfolgerungen werden durch die erhobenen Daten gestützt.

**Ergebnis:** Vollständiges Seminarprojekt mit Code, Messdaten, Bericht und Präsentation.

Dokumentation und Ergebnisprotokollierung erfolgen fortlaufend; die letzten beiden Wochen dienen der Fertigstellung.

## 6. Messgrößen und Auswertungsregeln

| Messgröße | Bedeutung |
|---|---|
| Genauigkeit ohne Fehler | Anteil korrekter Klassifikationen bei deaktivierter Injektion |
| Genauigkeit unter Fehlern | Anteil korrekter Klassifikationen unter der jeweiligen Injektionskonfiguration |
| Genauigkeitsverlust | Genauigkeit ohne Fehler minus Genauigkeit unter Fehlern |
| Vorhersageänderung | Anteil gültiger Vergleichspaare, bei denen sich die Klassifikation ändert |
| Neu entstandene Fehlklassifikation | Ohne Injektion korrekt, mit Injektion gültig, aber falsch |
| Ungültige Ergebnisse | Anteil von Ausgaben mit NaN, Unendlich oder ohne auswertbare Vorhersage |
| Trainingsstabilität | Entwicklung des Loss, ungültige Gradienten und abgebrochene Trainingsläufe |

Für neu entstandene Fehlklassifikationen dient jeweils der fehlerfreie Lauf desselben Checkpoints als Bezug. Der Nenner ist die Zahl seiner ohne Fehler korrekt klassifizierten Fälle und wird mit angegeben. Da sich diese Mengen für A und B unterscheiden können, wird ergänzend auf der gemeinsamen Menge sauber korrekt erkannter Bilder verglichen.

Bereits vorher falsche Vorhersagen und durch die Injektion zufällig korrigierte Vorhersagen werden getrennt ausgewiesen. Eine bloße Änderung der Ausgabe ist daher nicht automatisch eine neu entstandene Fehlklassifikation.

Ungültige oder abgebrochene Inferenzfälle zählen bei der Gesamtgenauigkeit als nicht korrekt und werden zusätzlich separat berichtet. Sie dürfen nicht durch ein argmax auf ungültigen Zahlen scheinbar zu normalen Klassifikationen werden. Bei p kleiner als 1 werden sowohl alle Bilder als auch die tatsächlich injizierten Fälle ausgewertet.

Bei nicht endlichem Trainings-Loss oder ungültigen Gradienten wird der betroffene Lauf vor dem nächsten Optimiererschritt beendet und als instabil protokolliert. Fehler werden weder still verworfen noch durch nachträgliches Begrenzen der Werte verdeckt. Eine gezielte Stabilisierung wäre ein eigenes, klar ausgewiesenes Zusatzexperiment.

Die Darstellung umfasst Mittelwerte und Einzelwerte der drei Trainingsseeds sowie geeignete Unsicherheitsangaben. Mehrere Injektionen an demselben Modell ersetzen keine unabhängigen Trainingswiederholungen.

Pro Experiment werden mindestens Modell- und Checkpoint-ID, Epoche, Trainingsseed, Injektionsseed, Bild-ID, Label, Zielschicht, Koordinaten, Bitposition, Bitmuster vor und nach dem Eingriff, Referenzvorhersage, veränderte Vorhersage und Status gespeichert. Ohne Injektion entfallen die nicht zutreffenden Injektionsfelder.

## 7. Arbeitsteilung und Projektergebnisse

### Arbeitsschwerpunkte

| Schwerpunkt | Hauptaufgaben |
|---|---|
| Person 1 | CNN-Vorlage, MNIST-Anbindung, Training, Checkpoints und Referenzmessungen |
| Person 2 | PyTorchFI-Anbindung, Bitflips, Injektionspläne und Experimentsteuerung |
| Gemeinsam | Machbarkeitstest, Fehlermodell, Versuchsplanung, Auswertung, Bericht und Präsentation |

Beide Teammitglieder sollen die gesamte Pipeline erklären können. Die Schwerpunkte können im Verlauf gewechselt werden.

### Ergebnisse am Projektende

1. Ein selbst trainiertes MNIST-CNN aus Standard-PyTorch-Schichten.
2. Eine dokumentierte Fehlerinjektion in zwei Faltungsschichten mit überprüften Bitflips.
3. Vergleichsmodelle aus normalem Training und Training unter Fehlerinjektion.
4. Ein reproduzierbarer Ablauf für Trainingsläufe und Injektionsexperimente.
5. Messdaten und Diagramme zu Schicht, Bitbereich, Injektionsrate und Trainingsdauer.
6. Eine wissenschaftliche Bewertung der Robustheit und möglicher Nachteile des Fehlertrainings.

Der Schwerpunkt liegt auf Fehlerinjektion und experimenteller Zuverlässigkeitsanalyse. Zusätzliche Vergleiche mit fremden vortrainierten Modellen, anderen Architekturen, Gewichtsfehlern, dem Vorzeichenbit oder weiteren Trainingsinjektionsraten kommen erst nach Abschluss des Kernvergleichs infrage.

Die Experimente simulieren ausgewählte Fehlerwirkungen auf Tensor-Ebene. Aussagen über reale Hardware-Ausfallraten oder die Zuverlässigkeit ganzer Betriebssysteme lassen sich daraus nicht unmittelbar ableiten.

## 8. Grundlagen

- [Lernziele im Projekt-Repository](https://github.com/naisonnn/mnist-fault-injection/blob/main/docs/lernziele.md)
- [Bisherige Projektplanung](https://github.com/naisonnn/mnist-fault-injection/blob/main/docs/projektplanung.md)
- [PyTorchFI: Projekt und Quellcode](https://github.com/pytorchfi/pytorchfi)
- [PyTorchFI: Hook-Anbindung und Modellkopien](https://github.com/pytorchfi/pytorchfi/blob/master/pytorchfi/core.py)
- [PyTorch: Erweiterung von Autograd und benutzerdefinierte Ableitungen](https://docs.pytorch.org/docs/stable/notes/extending.html)
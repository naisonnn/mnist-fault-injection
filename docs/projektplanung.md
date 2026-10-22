# Projektplan: MNIST Fault Injection

**Team:** Vincent Dalchau, Nathan Richter  
**Verwendete Technologien:** Python, PyTorch, PyTorchFI, MNIST

## 1. Projektziel

Ziel des Projekts ist die Untersuchung der Auswirkungen simulierter Hardwarefehler auf das Verhalten und die Klassifikationsgenauigkeit eines neuronalen Netzes.

Dazu wird ein eigenes Convolutional Neural Network (CNN) mit ``PyTorch`` implementiert und auf dem MNIST-Datensatz zur Erkennung handgeschriebener Ziffern trainiert.

Mithilfe des Frameworks ``PyTorchFI`` (= _PyTorch Fault Injection_) sollen während der Inferenz gezielt Bitfehler in Zwischenergebnisse verschiedener Netzwerkschichten injiziert werden.

Anschließend wird untersucht, wie sich diese Fehler auf die Vorhersagen des Modells auswirken und ob bestimmte Schichten oder Bitpositionen besonders fehleranfällig sind.

### Zentrale Forschungsfrage

**Welchen Einfluss haben simulierte Bitfehler in den Aktivierungswerten unterschiedlicher CNN-Schichten auf die Klassifikationsgenauigkeit eines neuronalen Netzes?**

Daraus ergeben sich drei Teilfragen:

1. Wie häufig verändert ein injizierter Bitfehler die Klassifikation?
2. Sind unterschiedliche CNN-Schichten unterschiedlich empfindlich gegenüber Bitfehlern?
3. Welchen Einfluss hat die Position des veränderten Bits innerhalb eines Gleitkommawertes?

## 2. Methodik

Das Projekt umfasst vier technische Komponenten:

- **Convolutional Neural Net (CNN):** Eigenständig implementiertes und trainiertes neuronales Netz zur MNIST-Klassifikation.
- **Fehlerinjektion:** Gezielte Manipulation einzelner Aktivierungswerte mithilfe von ``PyTorchFI`` und einer Bitflip-Funktion.
- **Experimentsteuerung:** Automatisierte Durchführung reproduzierbarer Inferenzläufe mit und ohne Fehlerinjektion.
- **Auswertung:** Vergleich der Ergebnisse und statistische Analyse von Fehlklassifikationen, unveränderten Vorhersagen und Ausführungsfehlern.

Es werden ausschließlich Fehler während der Inferenz untersucht. Das Training selbst erfolgt ohne Fehlerinjektion.

## 3. Meilensteinplanung

### Meilenstein 1 – Projektgrundlage und CNN (Woche 1–2)

**Ziel:** Ein eigenständig entwickeltes, funktionsfähiges neuronales Netz steht zur Verfügung.

Aufgaben:
- Python-Projekt und Entwicklungsumgebung einrichten
- ``PyTorch`` und ``PyTorchFI`` auf Kompatibilität prüfen
- MNIST-Datensatz einbinden
- CNN-Architektur mit zwei Convolutional Layers entwickeln
- Trainings- und Evaluationsskripte implementieren
- Modell trainieren und abspeichern

**Erfolgskriterium:** Das CNN erreicht eine angestrebte Klassifikationsgenauigkeit von mindestens 95 % auf dem MNIST-Testdatensatz.

**Ergebnis:** Trainiertes und reproduzierbar ausführbares CNN.

### Meilenstein 2 – Fehlerinjektion implementieren (Woche 3–4)

**Ziel:** Einzelne Bitfehler können kontrolliert während der Inferenz injiziert werden.

Aufgaben:
- PyTorchFI in das CNN integrieren
- Manipulation einzelner Aktivierungswerte implementieren
- Bitflip-Funktion für Float32-Werte entwickeln
- Injektionspunkte in zwei CNN-Schichten definieren
- Erfolgreiche Fehlerinjektion anhand von Testfällen nachweisen
- Sicherstellen, dass aufeinanderfolgende Experimente unabhängig voneinander bleiben

**Erfolgskriterium:** Ein gezielt ausgewähltes Bit eines Aktivierungswertes kann während eines Inferenzlaufs verändert werden. Die tatsächliche Bitänderung ist nachweisbar und der unveränderte Referenzlauf bleibt reproduzierbar.

**Ergebnis:** Funktionsfähiges Fehlerinjektionsmodul.

### Meilenstein 3 – Experimentautomatisierung (Woche 5–6)

**Ziel:** Fehlerinjektionsexperimente lassen sich automatisiert und reproduzierbar durchführen.

Aufgaben:
- Automatisierte Durchführung von Inferenzläufen entwickeln
- Zufällige Injektionspositionen kontrolliert auswählen
- Unterschiedliche Injektionsschichten untersuchen
- Bitflips im Exponenten und in der Mantisse unterscheiden
- Ergebnisse strukturiert speichern (z. B. als CSV)
- Kontrollläufe ohne Fehlerinjektion durchführen

**Erfolgskriterium:** Eine größere Anzahl von Experimenten kann ohne manuelle Eingriffe durchgeführt werden. Für jeden Lauf werden Injektionsparameter, ursprüngliche Vorhersage, manipulierte Vorhersage und Ausführungsstatus protokolliert.

**Ergebnis:** Automatisiertes Experimentframework mit ersten Messdaten.

### Meilenstein 4 – Experimente und Auswertung (Woche 7–8)

**Ziel:** Die Auswirkungen der injizierten Bitfehler werden quantitativ untersucht.

Aufgaben:
- Festgelegte Experimentkonfigurationen durchführen
- Fehlklassifikationsrate und Vorhersageänderungsrate bestimmen
- Unterschiede zwischen den CNN-Schichten analysieren
- Auswirkungen verschiedener Bitpositionen vergleichen
- Ausführungsfehler und ungültige Berechnungsergebnisse erfassen
- Ergebnisse grafisch aufbereiten

**Erfolgskriterium:** Für jede Untersuchungskonfiguration liegen ausreichend wiederholte Messungen vor, sodass sich die beobachteten Unterschiede nachvollziehbar vergleichen lassen.

**Ergebnis:** Ausgewertete Versuchsdaten, Diagramme und dokumentierte Erkenntnisse.

### Meilenstein 5 – Dokumentation und Abschluss (Woche 9–10)

**Ziel:** Das Projekt ist vollständig dokumentiert und die Forschungsfrage kann anhand der Ergebnisse beantwortet werden.

Aufgaben:
- Versuchsaufbau und Implementierung dokumentieren
- Experimentergebnisse interpretieren
- Forschungsfrage beantworten
- Einschränkungen und Fehlerquellen diskutieren
- GitHub-Repository vervollständigen
- Abschlussbericht und Präsentation vorbereiten

**Erfolgskriterium:** Experimente sind anhand der Dokumentation reproduzierbar und die Forschungsfrage wird durch die erhobenen Messdaten beantwortet.

**Ergebnis:** Abgeschlossenes Projekt mit Quellcode, Experimentdaten, Auswertung und Dokumentation.

## 4. Geplanter Experimentumfang

Um das Projekt innerhalb von zehn Wochen realistisch umzusetzen, wird die Untersuchung zunächst auf folgende Parameter begrenzt:

| Parameter | Festlegung |
|---|---|
| Datensatz | MNIST |
| Netzwerk | Eigenes CNN |
| Fehlerart | Einzelner Bitflip |
| Injektionsobjekt | Aktivierungswert |
| Injektionspunkte | Zwei CNN-Schichten |
| Bitbereiche | Mantisse und Exponent |
| Ausführungsphase | Inferenz |
| Experimentwiederholungen | Angestrebt: 500 je Konfiguration |

Daraus ergeben sich zunächst vier Experimentkonfigurationen mit insgesamt rund 2.000 Fehlerinjektionsläufen.

Die wichtigsten Messgrößen sind:

- Anteil veränderter Klassifikationen gegenüber dem fehlerfreien Referenzlauf
- Anteil neu entstandener Fehlklassifikationen
- Anteil unveränderter Vorhersagen
- Anteil ungültiger Berechnungsergebnisse oder Ausführungsabbrüche

Die Experimente erfolgen mit festgelegten Zufalls-Seeds und einer dokumentierten Auswahl von Testbildern.

## 5. Erwartete Projektergebnisse

Am Ende des Projektseminars sollen folgende Ergebnisse vorliegen:

1. Ein selbst entwickeltes und trainiertes CNN zur MNIST-Klassifikation.
2. Eine funktionierende Integration von PyTorchFI einschließlich gezielter Bitflip-Injektionen.
3. Ein automatisiertes und reproduzierbares Experimentframework.
4. Eine quantitative Auswertung der Auswirkungen verschiedener Bitfehler.
5. Eine wissenschaftliche Bewertung der Fehleranfälligkeit der untersuchten CNN-Schichten.

Das Projekt soll damit einen experimentellen Beitrag zum Verständnis der Zuverlässigkeit neuronaler Netze gegenüber simulierten Speicher- und Berechnungsfehlern liefern.

## 6. Projektabgrenzung

**Nicht Bestandteil des Kernprojekts sind:**

- Entwicklung eines großen Sprachmodells
- Training unter fehlerhaften Bedingungen
- Physische Manipulation von Hardware
- Injektion tatsächlicher Betriebssystem- oder Kernel-Fehler
- Entwicklung komplexer Fehlerkorrekturmechanismen
- Vergleich zahlreicher unterschiedlicher Netzwerkarchitekturen

Die Experimente simulieren die Auswirkungen ausgewählter Hardwarefehler auf der Ebene von PyTorch-Tensoren. Sie ersetzen keine Fehlerinjektion in reale Hardware oder Betriebssystemkomponenten.

Diese Eingrenzung soll eine vollständige Bearbeitung innerhalb des vorgesehenen Zeitrahmens gewährleisten.
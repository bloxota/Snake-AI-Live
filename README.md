# Snake-AI-Live
Ein intelligentes Snake-Spiel, bei dem eine KI live spielt und nach jedem Spiel lernt. Mit Pygame visualisiert, inklusive Score- und Highscore-Anzeige.
README.md Beispiel
# Snake AI Live

Ein intelligentes Snake-Spiel, bei dem eine KI live spielt und online dazulernt. Das Projekt ist **anfängerfreundlich**, zeigt **Score und Highscore**, und die KI kann **trainiert oder geladen** werden, um sofort "schlau" zu spielen.


## Features

- KI spielt live das Snake-Spiel
- Score und Highscore werden korrekt angezeigt
- Das Spiel ist in Pygame visualisiert
- KI kann bereits trainiertes Modell laden oder selbst trainieren
- Normale Geschwindigkeit wie bei menschlichem Spielen
- Modell wird automatisch gespeichert, sodass Fortschritt erhalten bleibt

## Installation

1. Repository klonen:

```bash
git clone https://github.com/DEIN_USERNAME/Snake-AI-Live.git
cd Snake-AI-Live


Virtuelle Umgebung erstellen (optional, empfohlen):

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Mac/Linux


Abhängigkeiten installieren:

pip install pygame stable-baselines3 gymnasium numpy

Nutzung
python snake_ai_smart_live_fixed.py


Die KI spielt direkt live

Score und Highscore werden angezeigt

Beim ersten Start trainiert die KI kurz, danach ist sie schlau

Modell speichern / laden

Das trainierte Modell wird automatisch unter snake_dqn_smart_live.zip gespeichert

Bei Neustart wird das Modell automatisch geladen, wenn es existiert

Lizenz

Dieses Projekt ist unter der MIT-Lizenz verfügbar.

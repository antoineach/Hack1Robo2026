# Essais CPU Qwen

- [Paliers lents](cinema-paliers-lents.json) et [texte brut à lire](cinema-paliers-lents.txt). Un changement de 0,1 tous les 12 tokens après les 20 premiers. Les températures de départ, de pic et de fin valent 0,8, 1,4 et 0,1.
- [Paliers rapides](cinema-paliers-rapides.json). Même amorce et même graine, avec un changement tous les 6 tokens.
- [Rechutes à 0,15 et 0,10](rechutes-profondes.json). Quatre suites sur le cinéma et la serre.
- [Exploration](exploration.json), [validation](validation.json) et [comparaison avec la même amorce](demonstration.json). Les 13 récits et leurs [jugements à l’aveugle par GPT‑5.6 Luna](luna-synthese.json).
- [Rejouer la génération et suivre le texte](evolution-luna.html). Depuis la racine du dépôt, lance `python3 -m http.server 1920 --bind 127.0.0.1` et ouvre `http://127.0.0.1:1920/experiences/evolution-luna.html`.

La trace associe chaque token à une température, une entropie en nats et une fin de caractère Unicode dans la suite générée. L’amorce est conservée dans un champ distinct. Les textes sont repris bruts, y compris les changements de langue et les fins coupées.

## Régénérer sur CPU

```bash
python -m pip install torch 'transformers==5.5.4' accelerate
python experiences/qwen_cpu.py --intervalle 12 --sortie zonos3-rendus/recit.json
python zonos3-mini/preparer_llm.py zonos3-rendus/recit.json --sortie zonos3-rendus/requetes.json
python zonos3-mini/rejouer.py --mesures zonos3-rendus/requetes.json
```

Le script Qwen télécharge environ 1 Go de poids au premier lancement. La dernière commande utilise [le serveur local ZONOS3-Mini](../zonos3-mini/README.md#relancer-le-moteur-cpu).

Pour la courbe linéaire prolongée, remplace `--intervalle 12` par `--rechute 0.15` ou `--rechute 0.10`. Les champs `assistant_prefix` et `seed` des traces indiquent les valeurs à reprendre avec `--amorce` et `--graine` pour la serre.

Dans la première campagne, aucun des 13 récits soumis à Luna n’atteint 3 sur 5 en cohérence. La comparaison avec amorce donne un avantage à la courbe en originalité et en progression sur un tirage. La série à paliers a été générée après ce jugement.

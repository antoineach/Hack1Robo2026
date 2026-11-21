# ZONOS3-Mini · Effets pendant la phrase

## Lire un récit Qwen

Le [récit du cinéma à paliers lents](../experiences/cinema-paliers-lents.json) inclut la température et la fin Unicode de chaque token dans la suite générée. Le script `preparer_llm.py` utilise ces positions pour produire deux requêtes à texte et graine identiques. La prise pilotée associe la montée aux directions de vertige et la rechute à une voix plus soufflée. Le texte parlé contient l’amorce et la suite brute, sans annonce de changement.

Le fichier [qwen-mesures.json](qwen-mesures.json) est prêt pour le serveur décrit plus bas. Depuis la racine du dépôt.

```bash
python3 zonos3-mini/rejouer.py --mesures zonos3-mini/qwen-mesures.json --liste
python3 zonos3-mini/rejouer.py --mesures zonos3-mini/qwen-mesures.json
```

Pour convertir un autre récit.

```bash
python3 zonos3-mini/preparer_llm.py experiences/rechutes-profondes.json \
  --essai cinema-rechute-10 --sortie zonos3-rendus/requetes.json
python3 zonos3-mini/rejouer.py --mesures zonos3-rendus/requetes.json
```

`--suivi rate` permet de préparer une variante à cadence de caractères. Si la synthèse s’arrête avant la fin d’un récit long, augmente le budget `--max-tokens` du convertisseur et `--max` au démarrage du serveur. Le rendu final dépend de l’alignement et de la normalisation du moteur.

La galerie compare un témoin à des prises où le moteur applique des vecteurs de contrôle à des passages précis du texte français. Elle contient six nouvelles prises sur la même phrase, ainsi que les 36 essais de la première série. [Ouvrir la galerie privée et télécharger les WAV](https://ecoute-zonos3-fr.brightcape9.chatgpt.site/#effets-pendant-la-phrase)

| Prise | Passage ciblé |
| --- | --- |
| [Témoin](audio/passage_temoin.wav) | Phrase entière, sans vecteur ajouté |
| [Élan au début](audio/passage_euphorie.wav) | « Quand les lumières du quai se rallument » |
| [Vertige au milieu](audio/passage_vertige.wav) | « je te fais signe près de la porte bleue », suivi de la lecture |
| [Vertige à cadence fixe](audio/passage_vertige_cadence.wav) | Même texte et mêmes vecteurs, placement à cadence de caractères |
| [Souffle à la fin](audio/passage_souffle.wav) | « et tu viens me rejoindre » |
| [Deux bascules](audio/passage_contraste.wav) | Élan à l’ouverture, souffle à la fin |

## Reprendre dans une autre session

- `mesures.json` contient les requêtes envoyées, les modes de placement réellement utilisés par le serveur, les transcriptions automatiques et le SHA-256 de chaque WAV. Les transcriptions sont indicatives.
- `audio/` contient les prises d’origine en mono à 44,1 kHz. `bilan.json` résume la campagne et `configuration.json` décrit le serveur et ses directions latentes.
- `rejouer.py` relit les requêtes et trouve la voix Gaspard sur le serveur courant. Les nouveaux WAV vont par défaut dans `zonos3-rendus/`. Le dossier peut être cloné ou ouvert directement par une autre session Codex.
- `latent_cases.py` et `temporal_cases.py` contiennent les recettes qui ont servi à monter les deux séries. Pour adapter une prise, copie `mesures.json`, modifie la requête voulue et passe le nouveau chemin avec `--mesures`.

Pour écouter la galerie sur ta machine depuis ce dossier, lance `python3 -m http.server 1920 --bind 127.0.0.1`, puis ouvre `http://127.0.0.1:1920/#effets-pendant-la-phrase`.

## Relancer le moteur CPU

Prévois CMake, un compilateur C/C++, Rust avec Cargo, le CLI Hugging Face `hf`, Python 3 et environ 1,2 Go pour les poids. Depuis ce dossier, récupère les versions utilisées pour cette campagne. [Moteur C++/ggml de Zyphra](https://github.com/Zyphra/ZONOS3-Mini/tree/a50e14117d99181ba7f28047b451ba03ba1a09a9/servers/ggml)

```bash
git clone https://github.com/Zyphra/ZONOS3-Mini.git moteur
git -C moteur checkout a50e14117d99181ba7f28047b451ba03ba1a09a9
git -C moteur submodule update --init servers/ggml/ggml third_party/zyphra-textnorm
cmake -S moteur/servers/ggml -B moteur/build -DCMAKE_BUILD_TYPE=Release
cmake --build moteur/build --parallel 2 --target zonos3-server

hf download Zyphra/ZONOS3-Mini --revision 2921b8821772b0bd10a04456863d21abf9d2f283 \
  --include 'gguf/backbone-ptq.gguf' 'gguf/zac-keep2.gguf' 'gguf/steer.gguf' 'gguf/voices/French/*' \
  --local-dir moteur/poids

OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OMP_WAIT_POLICY=PASSIVE ZAC_THREADS=2 \
  moteur/build/zonos3-server moteur/poids/gguf/backbone-ptq.gguf \
  --cpu --host 127.0.0.1 --port 1919 --batch 2 --max 720 --zac-threads 1 \
  --steer moteur/poids/gguf/steer.gguf \
  --tts-default-voices-dir moteur/poids/gguf/voices \
  --laci-heads fr_fr=moteur/assets/laci_heads/zonos3_mini_heads_fr.json
```

Dans un deuxième terminal, depuis le même dossier.

```bash
python3 rejouer.py --liste
python3 rejouer.py --essai passage_temoin --essai passage_vertige --essai passage_vertige_cadence
```

Les bornes `start` et `end` de `control_spans` sont des positions Unicode dans le texte brut, avec une fin exclue. Chaque passage accepte des directions internes `attnres_source`. `reading` demande le suivi de la lecture avec les têtes d’alignement françaises, tandis que `rate` place le contrôle sur une cadence de caractères. La valeur `span_follow` stockée dans les mesures vient de la réponse du serveur. L’écoute garde le dernier mot sur le caractère perçu de chaque prise.

# Hack1Robo · Température par token

Fais varier l’échantillonnage d’un Gemma 4 au fil d’un récit. Le même modèle produit le début nominal, la montée, le pic et la rechute. Le notebook donne le texte, la température et l’entropie à chaque token, puis lance une comparaison à l’aveugle.

## Démarrer dans le navigateur

1. [Ouvre le notebook sur Google Colab](https://colab.research.google.com/github/antoineach/Hack1Robo2026/blob/main/Hack1Robo.ipynb).
2. Pour le tirage exact, choisis **Exécution → Modifier le type d’exécution → GPU**, puis exécute les cellules de la section **Décodage exact**. Le GPU gratuit dépend des quotas et de la disponibilité de Colab. Les poids sont téléchargés dans la session cloud.
3. Fais d’abord le test d’un récit, puis lance la comparaison. La configuration courte du notebook utilise une graine et un prompt par lot. Augmente le nombre de prompts et de graines pour interpréter les écarts. Télécharge le JSON avant la fin de la session. [FAQ Colab](https://research.google.com/colaboratory/faq.html)

Le décodeur utilise **Gemma 4 E2B Instruct**. Le juge et les trois rôles d’agents peuvent utiliser le même modèle, sans clé. Pour un juge indépendant, règle le secret Colab `GEMINI_API_KEY` ou `OPENROUTER_API_KEY` et sélectionne le fournisseur dans la cellule de configuration. Les rôles « associations », « cohérence » et « progression » proposent chacun un profil à partir des notes précédentes. Tu peux leur faire partager le même LLM ou leur donner un callable par rôle. Les notes et les justifications du juge sont dans le fichier d’expérience. [Fiche du modèle](https://huggingface.co/google/gemma-4-E2B-it)

La section **API hébergée** fonctionne dans une session sans GPU. Elle contient une démo de trois blocs, une sonde des probabilités et un catalogue des modèles gratuits qui déclarent cette capacité. Les secrets sont lus depuis Colab. Le notebook affiche les réponses de test sans afficher les clés.

## Choix du fournisseur

| Service | Usage |
|---|---|
| Colab gratuit | Tirage exact sur les logits du modèle, avec une session GPU temporaire. |
| OpenRouter | Gemma 4 26B A4B `:free` pour la démo par blocs ou le juge. Lors de la vérification, son endpoint gratuit déclarait `temperature` mais aucun `logprobs`. Trois autres modèles gratuits déclaraient `top_logprobs`, dont `liquid/lfm-3-1.7b:free`. Le catalogue et les quotas évoluent. |
| Gemini API | Gemma 4 26B A4B sur le palier gratuit pour les blocs, ou un Gemini Flash-Lite pris en charge pour le jugement. La sonde vérifie ce qu’un modèle donné renvoie réellement. |
| OpenCode | Sa documentation réserve désormais les modèles de chat gratuits au client OpenCode. Les appels externes à son API d’inférence reçoivent HTTP 403. Son API publique ne documente ni distribution complète ni fonction de tirage personnalisée. |

[Catalogue OpenRouter](https://openrouter.ai/docs/api/api-reference/models/get-models) · [Fournisseur du Gemma 4 gratuit](https://openrouter.ai/api/v1/models/google/gemma-4-26b-a4b-it:free/endpoints) · [Gemma dans la Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) · [Tarifs Gemini](https://ai.google.dev/gemini-api/docs/pricing#gemma-4) · [API OpenCode](https://opencode.ai/docs/console/api/inference/)

La continuation hébergée demande au modèle d’écrire la suite à partir du texte déjà produit. La température est fixe dans chaque appel et la longueur des blocs varie. L’expérience par token se déroule dans le moteur Transformers, qui conserve son cache et la séquence exacte des IDs. Colab sert ici de notebook interactif.

## Ce que permettent les probabilités

Avec le **vecteur complet** `p = softmax(z / T0)`, les logits relatifs valent `T0 * log(p)` à une constante additive près. Cette constante s’annule lors d’un nouveau tirage. On peut donc calculer `softmax(log(p) * T0 / T1)` à la température `T1`, si les probabilités sont celles de la softmax à `T0` sans autre filtre. Des `top_logprobs` donnent seulement quelques candidats. La sonde repondère les candidats reçus et renormalise à l’intérieur de cette liste. Une troncature de vocabulaire, un filtre de tirage ou une probabilité arrondie à zéro retire de l’information. Une réponse en streaming transmet les tokens après leur sélection.

## Expérience et classement

- La référence garde `T = 0,8`. La courbe monte de `0,8` à `1,4` entre les tokens 20 et 55, descend à `0,35` au token 95, puis reste à ce niveau. Les indices débutent à zéro après le prompt.
- **Softmax** applique la courbe. **Mélange** ajoute au pic une composante plus chaude parmi les meilleurs candidats. **Entmax 1,5** remplace complètement la softmax au pic dans la variante prédéfinie. Les variantes reviennent à la softmax à la rechute. [Entmax](https://aclanthology.org/P19-1146/)
- Le juge ne reçoit ni la courbe ni le nom de la variante. Il note la cohérence, l’originalité et la progression sur cinq points. Le seuil par défaut est une moyenne de cohérence de 3. Les profils admissibles sont classés selon la moyenne des deux autres axes.
- La meilleure variante admissible passe sur des prompts réservés, avec les deux contrôles **température constante** et **style demandé dans le prompt**. Compare les textes et les notes de validation. Le gagnant de l’exploration peut perdre sur ce lot. Un juge local qui partage le modèle du générateur peut partager ses biais.
- Le tirage exact utilise le vocabulaire complet, avec une longueur minimale commune pour laisser la rechute apparaître avant que le modèle puisse terminer. Le contrôle nominal suit les mêmes réglages. La génération et les notes peuvent varier selon l’environnement et le modèle de jugement.

`hack1robo.py` contient le décodeur, les clients API, le juge et la boucle d’expérience. Le notebook importe ce fichier depuis le dépôt.

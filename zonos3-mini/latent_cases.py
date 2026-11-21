"""Paired latent steering experiments. The numbers are vector coefficients."""
PHRASE = "Le train arrive dans cinq minutes. Je suis encore sur le quai. Tu me vois, près de la porte bleue ?"
TEXTE_B = "J’ai laissé mon manteau sur la chaise. La lumière du couloir est toujours allumée."

SEDATION = {'delivery.tempo': -2, 'mood.arousal': -4, 'expressive.expressive': -4, 'emotions.serene': 3}
EUPHORIA = {'delivery.tempo': 1.2, 'mood.arousal': 4, 'mood.valence': 4, 'emotions.happy': 3}
VERTIGO = {'delivery.tempo': -0.8, 'emotions.confused': 3, 'disfluency.disfluent': 6,
           'expressive.expressive': 5}
DISTANCE = {'mood.arousal': -3, 'expressive.expressive': -6, 'delivery.whisper': 1.2,
            'room.clean': -2}
AGITATION = {'delivery.tempo': 1.8, 'mood.arousal': 4, 'mood.valence': -3, 'disfluency.disfluent': 7,
             'expressive.expressive': 5}

def sweep(family, name, controls, target):
    for i, scale in enumerate((0.45, 1, 1.45), 1):
        yield (f'{name}_{i}', family, PHRASE,
               {'controls': {direction: round(weight * scale, 3) for direction, weight in controls.items()}},
               target)

def span(text, part, controls):
    start = text.index(part)
    return {'start': start, 'end': start + len(part), 'controls': controls}

AXES = [
    ('ralenti', 'delivery.tempo', -2, 'Allonger le débit.'),
    ('accelere', 'delivery.tempo', 2, 'Accélérer le débit.'),
    ('hesitation', 'disfluency.disfluent', 8, 'Faire apparaître hésitations et blocages.'),
    ('souffle', 'delivery.whisper', 2, 'Augmenter le souffle et réduire le voisement.'),
    ('excitation', 'mood.arousal', 5, 'Augmenter l’énergie vocale.'),
    ('egal', 'expressive.expressive', -8, 'Réduire les excursions de hauteur.'),
    ('timbre_grave', 'pitch.pitch', -2, 'Abaisser la hauteur via l’embedding de voix.'),
    ('piece_lointaine', 'room.clean', -3, 'Dégrader la netteté via l’embedding de voix.'),
]

CASES = [
    ('temoin', 'Témoins', PHRASE, {}, 'Référence sans intervention latente.'),
    ('temoin_repeat', 'Témoins', PHRASE, {}, 'Même graine, comparaison bit à bit.'),
    ('temoin_seed', 'Témoins', PHRASE, {'seed': 43}, 'Variation due à une autre graine.'),
    *((name, 'Contrôles isolés', PHRASE, {'controls': {direction: weight}}, target)
      for name, direction, weight, target in AXES),
    *sweep('Somnolence', 'somnolence', SEDATION, 'Débit ralenti, voix posée, faible expressivité.'),
    *sweep('Euphorie', 'euphorie', EUPHORIA, 'Élan rapide, intonation positive, énergie accrue.'),
    *sweep('Vertige', 'vertige', VERTIGO, 'Débit hésitant, confusion, intonation plus instable.'),
    *sweep('Voix distante', 'distance', DISTANCE, 'Voix soufflée et détachée, avec une pièce moins nette.'),
    *sweep('Agitation', 'agitation', AGITATION, 'Accentuer la tension et les hésitations à débit élevé.'),
    ('bascule', 'Effets pendant la phrase', PHRASE,
     {'control_spans': [span(PHRASE, 'Je suis encore sur le quai.', VERTIGO)], 'control_span_follow': 'reading', 'control_span_rate': 1.05},
     'Concentrer le vertige sur la deuxième phrase.'),
    ('alternance', 'Effets pendant la phrase', PHRASE,
     {'control_spans': [span(PHRASE, 'Le train arrive dans cinq minutes.', EUPHORIA),
                        span(PHRASE, 'Tu me vois, près de la porte bleue ?', SEDATION)],
      'control_span_follow': 'reading', 'control_span_rate': 1.05},
     'Commencer en euphorie puis terminer en somnolence.'),
    ('retombee', 'Effets pendant la phrase', PHRASE,
     {'controls': {'delivery.tempo': EUPHORIA['delivery.tempo']},
      'control_spans': [
          span(PHRASE, 'Le train arrive', {k: v for k, v in EUPHORIA.items() if k != 'delivery.tempo'}),
          span(PHRASE, 'dans cinq minutes.', {k: v * 0.45 for k, v in EUPHORIA.items() if k != 'delivery.tempo'}),
      ], 'control_span_rate': 1.05},
     'Atténuer l’euphorie par paliers pendant la première phrase, avec un réglage de débit constant.'),
    ('vertige_cfg', 'Effets pendant la phrase', PHRASE,
     {'controls': VERTIGO, 'controls_cfg_scale': 1.5},
     'Amplifier les effets du vertige par guidance conditionnelle.'),
    ('agitation_cfg', 'Effets pendant la phrase', PHRASE,
     {'controls': AGITATION, 'controls_cfg_scale': 1.5},
     'Amplifier la tension avec une deuxième passe sans intervention.'),
    ('somnolence_grave', 'Effets pendant la phrase', PHRASE,
     {'controls': {**SEDATION, 'pitch.pitch': -2, 'texture.soft': 3}},
     'Ajouter un timbre grave et doux à la somnolence.'),
    ('autre_texte', 'Reproductibilité', TEXTE_B, {}, 'Témoin sur une autre phrase française.'),
    ('autre_vertige', 'Reproductibilité', TEXTE_B, {'controls': VERTIGO}, 'Tester le vertige sur cette autre phrase.'),
    ('autre_somnolence', 'Reproductibilité', TEXTE_B, {'controls': SEDATION}, 'Tester la somnolence sur cette autre phrase.'),
    ('autre_euphorie', 'Reproductibilité', TEXTE_B, {'controls': EUPHORIA}, 'Tester l’euphorie sur cette autre phrase.'),
]

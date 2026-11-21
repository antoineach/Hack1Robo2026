"""Create a matched French listening series for control changes within one sentence."""
from latent_cases import span

TEXT = "Quand les lumières du quai se rallument, je te fais signe près de la porte bleue et tu viens me rejoindre."
FIRST = "Quand les lumières du quai se rallument"
MIDDLE = "je te fais signe près de la porte bleue"
LAST = "et tu viens me rejoindre"
VERTIGO = {'emotions.confused': 2.2, 'disfluency.disfluent': 3.5, 'expressive.expressive': 4}
BRIGHT = {'mood.valence': 3, 'mood.arousal': 3, 'emotions.happy': 2}
WHISPER = {'delivery.whisper': 2.2, 'mood.arousal': -2}

def options(*spans, follow='reading'):
    return {'control_spans': [span(TEXT, words, controls) for words, controls in spans],
            'control_span_follow': follow, 'control_span_rate': 1.05}

FAMILY = 'Effets pendant la phrase'
CASES = [
    ('passage_temoin', FAMILY, TEXT, {}, 'Phrase de référence pour les cinq prises suivantes.'),
    ('passage_euphorie', FAMILY, TEXT, options((FIRST, BRIGHT)), 'Donner un élan à l’ouverture, puis retirer le contrôle.'),
    ('passage_vertige', FAMILY, TEXT, options((MIDDLE, VERTIGO)), 'Concentrer le vertige au milieu en suivant la lecture.'),
    ('passage_vertige_cadence', FAMILY, TEXT, options((MIDDLE, VERTIGO), follow='rate'),
     'Reprendre les mêmes vecteurs en plaçant le passage selon une cadence de caractères.'),
    ('passage_souffle', FAMILY, TEXT, options((LAST, WHISPER)), 'Rendre la fin plus soufflée.'),
    ('passage_contraste', FAMILY, TEXT, options((FIRST, BRIGHT), (LAST, WHISPER)),
     'Enchaîner une ouverture lumineuse et une fin murmurée.'),
]

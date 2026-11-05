"""Telugu text frontend.

Tokens are Unicode code points of the Telugu block (U+0C00-U+0C7F) + space + punctuation, i.e. a
character-level model (Telugu spelling is close to phonetic, so no G2P is needed). Numbers, times
and ordinals are verbalised into Telugu words. `romanize_tokens` gives an approximate Latin
transliteration used ONLY to drive the multilingual CTC aligner during preprocessing.
"""
import re
import unicodedata

PAD, SPACE = "_", " "
PUNCT = list(",.?!-")
TELUGU = [chr(c) for c in range(0x0C00, 0x0C80)]
SYMBOLS = [PAD, SPACE] + PUNCT + TELUGU
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
N_SYMBOLS = len(SYMBOLS)

# ------------------------------------------------------------------------------ numbers
_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది",
         "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది",
         "పంతొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}
_DIGIT_TE = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")


def number_to_telugu(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0:
            return "వంద" if h == 1 else _ONES[h] + " వందలు"
        return ("వంద" if h == 1 else _ONES[h] + " వందల") + " " + number_to_telugu(r)
    if n < 100000:
        th, r = divmod(n, 1000)
        if r == 0:
            return "వెయ్యి" if th == 1 else number_to_telugu(th) + " వేలు"
        return ("వెయ్యి" if th == 1 else number_to_telugu(th) + " వేల") + " " + number_to_telugu(r)
    return " ".join(_ONES[int(c)] for c in str(n))          # very large: digit by digit


def _ordinal(n):
    w = number_to_telugu(n)
    return (w[:-1] if w[-1] in "ుిై" else w) + "వ"


def normalize(text):
    t = unicodedata.normalize("NFC", text).translate(_DIGIT_TE)
    t = t.replace("‌", "").replace("‍", "")           # ZWNJ / ZWJ
    t = re.sub(r"(\d{1,2}):(\d{2})", lambda m: number_to_telugu(int(m.group(1))) + " " +
               (number_to_telugu(int(m.group(2))) if m.group(2) != "00" else ""), t)
    t = re.sub(r"(\d+)\s*వ\b", lambda m: _ordinal(int(m.group(1))), t)
    t = re.sub(r"\d+", lambda m: number_to_telugu(int(m.group(0))), t)
    t = t.replace(";", ",").replace(":", ",").replace("–", "-").replace("—", "-").replace("…", ".")
    t = "".join(ch for ch in t if ch in SYM2ID or ch == " ")
    t = re.sub(r"\s+([.,?!])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def text_to_tokens(text):
    return [c for c in normalize(text) if c in SYM2ID]


def tokens_to_ids(tokens):
    return [SYM2ID[t] for t in tokens if t in SYM2ID]


def text_to_sequence(text):
    return tokens_to_ids(text_to_tokens(text))


# ------------------------------------------------------------------------------ romanisation
_V = dict(zip("అఆఇఈఉఊఋౠఎఏఐఒఓఔ", "a a i i u u ru ru e e ai o o au".split()))
_C = dict(zip("కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహౘౙ",
              "k kh g gh n ch ch j jh n t th d dh n t th d dh n p ph b bh m y r r l l v sh sh s h ch j".split()))
_SIGN = dict(zip("ాిీుూృౄెేైొోౌ", "a i i u u ru ru e e ai o o au".split()))
_VIRAMA, _OTHER = "్", {"ం": "m", "ః": "h", "ఁ": "n"}


def romanize_tokens(tokens):
    """Latin letters (a-z) aligned to each token; '' for tokens without their own sound."""
    out = []
    for i, ch in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if ch in _C:
            out.append(_C[ch] + ("" if nxt in _SIGN or nxt == _VIRAMA else "a"))
        elif ch in _V:
            out.append(_V[ch])
        elif ch in _SIGN:
            out.append(_SIGN[ch])
        elif ch in _OTHER:
            out.append(_OTHER[ch])
        else:
            out.append("")
    return out


def is_separator(tok):
    return tok == SPACE or tok in PUNCT

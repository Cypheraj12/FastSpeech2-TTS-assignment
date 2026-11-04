"""English text frontend: normalisation (numbers, times, dates, abbreviations) + g2p_en phonemes."""
import re


PAD, SPACE = "_", " "
PUNCT = list(",.?!;:'")
ARPA = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW "
        "V W Y Z ZH").split()
ARPA_STRESS = [p + str(s) if p[0] in "AEIOU" else p for p in ARPA for s in ((0, 1, 2) if p[0] in "AEIOU" else (None,))]
SYMBOLS = [PAD, SPACE] + PUNCT + ARPA_STRESS
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
N_SYMBOLS = len(SYMBOLS)

_MONTHS = "january february march april may june july august september october november december".split()
_ABBR = [(r"\bmr\.", "mister"), (r"\bmrs\.", "missus"), (r"\bdr\.", "doctor"), (r"\bst\.", "saint"),
         (r"\bvs\.?", "versus"), (r"\betc\.", "et cetera"), (r"&", " and ")]
_g2p = None

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD_IRR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}


def num2words(n, to="cardinal"):
    """Small dependency-free English number speller (0 .. 999,999,999)."""
    def below_1000(x):
        parts = []
        if x >= 100:
            parts.append(_ONES[x // 100] + " hundred")
            x %= 100
        if x >= 20:
            parts.append(_TENS[x // 10] + ("-" + _ONES[x % 10] if x % 10 else ""))
        elif x > 0 or not parts:
            parts.append(_ONES[x])
        return " ".join(parts)
    if n == 0:
        words = "zero"
    else:
        chunks, parts = [], []
        for name, size in (("million", 10 ** 6), ("thousand", 1000)):
            if n >= size:
                parts.append(below_1000(n // size) + " " + name)
                n %= size
        if n:
            parts.append(below_1000(n))
        words = " ".join(parts)
    words = words.replace("-", " ")
    if to == "ordinal":
        head, _, last = words.rpartition(" ")
        if last in _ORD_IRR:
            last = _ORD_IRR[last]
        elif last.endswith("y"):
            last = last[:-1] + "ieth"
        else:
            last += "th"
        words = (head + " " + last).strip()
    return words


def _spell_letters(s):
    return " ".join(s.upper())


def _year(n):
    if 2000 <= n <= 2009:
        return num2words(n)
    if 1100 <= n <= 2099 and n % 100 != 0:
        return f"{num2words(n // 100)} {num2words(n % 100)}" if n % 100 >= 10 else \
            f"{num2words(n // 100)} oh {num2words(n % 100)}"
    return num2words(n)


def normalize(text):
    t = text.strip()
    def clock(m):
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "")
        hs = num2words(h)
        ms = "" if mi == 0 else (f"oh {num2words(mi)}" if mi < 10 else num2words(mi))
        if mi == 0 and ap == "":
            ms = "o'clock"
        return " ".join(x for x in [hs, ms, _spell_letters(ap.replace('.', ''))] if x)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", clock, t)
    mon = "|".join(m.capitalize() for m in _MONTHS) + "|" + "|".join(_MONTHS)
    t = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({mon})\b", lambda m: f"{num2words(int(m.group(1)), to='ordinal')} {m.group(2)}", t)
    t = re.sub(rf"\b({mon})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lambda m: f"{m.group(1)} {num2words(int(m.group(2)), to='ordinal')}", t)
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), t)
    t = re.sub(r"\b(1\d{3}|20\d{2})\b", lambda m: _year(int(m.group(1))), t)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: f"{num2words(int(m.group(1)))} point {' '.join(num2words(int(c)) for c in m.group(2))}", t)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group(0))), t)
    t = re.sub(r"\b([AP])\.?[Mm]\.?\b", lambda m: f"{m.group(1)} M", t)
    for pat, rep in _ABBR:
        t = re.sub(pat, rep, t, flags=re.I)
    t = t.lower().replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "").replace('"', "")
    t = t.replace(";", ",").replace(":", ",").replace("-", " ")
    t = "".join(ch for ch in t if ch in SYM2ID or ch.isalnum() or ch == " ")
    t = re.sub(r"\s+([,.?!])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def text_to_words(text):
    norm = normalize(text)
    return [w for w in re.split(r"([,.?!\s])", norm) if w and not w.isspace()]


def word_to_phones(word):
    global _g2p
    if word in PUNCT or word in (PAD, SPACE):
        return [word]
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    phones = _g2p(word)
    clean = []
    for p in phones:
        p = p.strip()
        if not p or p == " ":
            continue
        if p in SYM2ID:
            clean.append(p)
        elif p.rstrip("012") in SYM2ID:
            clean.append(p.rstrip("012"))
    return clean or [SPACE]


def text_to_tokens(text):
    words = text_to_words(text)
    tokens = []
    for i, w in enumerate(words):
        if i:
            tokens.append(SPACE)
        tokens.extend(word_to_phones(w))
    return tokens


def tokens_to_ids(tokens):
    return [SYM2ID[t] for t in tokens if t in SYM2ID]


def text_to_sequence(text):
    return tokens_to_ids(text_to_tokens(text))

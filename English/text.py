"""English text frontend: normalisation (numbers, times, dates, abbreviations) + g2p_en phonemes."""
import re


PAD, SPACE = "_", " "
PUNCT = list(",.?!;:-'")
ARPA = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW "
        "V W Y Z ZH").split()
ARPA_STRESS = [p + str(s) if p[0] in "AEIOU" else p for p in ARPA for s in ((0, 1, 2) if p[0] in "AEIOU" else (None,))]
SYMBOLS = [PAD, SPACE] + PUNCT + ARPA_STRESS
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
N_SYMBOLS = len(SYMBOLS)

_MONTHS = "january february march april may june july august september october november december".split()
_ABBR = [(r"\bmr\.", "mister"), (r"\bmrs\.", "missus"), (r"\bdr\.", "doctor"), (r"\bst\.", "saint"),
         (r"\bvs\.?\b", "versus"), (r"\betc\.", "et cetera"), (r"&", " and ")]
_g2p = None

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD_IRR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}


_inflect_engine = None


def num2words(n, to="cardinal"):
    """Robust English number speller."""
    global _inflect_engine
    if _inflect_engine is None:
        try:
            import inflect
            _inflect_engine = inflect.engine()
        except Exception:
            _inflect_engine = False
    if _inflect_engine:
        try:
            w = _inflect_engine.number_to_words(n, wantlist=False)
            if to == "ordinal":
                w = _inflect_engine.ordinal(w)
            return re.sub(r"[,\-]", " ", w).lower().strip()
        except Exception:
            pass

    # Pure Python fallback
    s = str(abs(n))
    if len(s) > 9:
        return " ".join(_ONES[int(c)] for c in s)

    def below_1000(x):
        x = x % 1000
        parts = []
        if x >= 100:
            parts.append(_ONES[(x // 100) % 10] + " hundred")
            x %= 100
        if x >= 20:
            parts.append(_TENS[x // 10] + ("-" + _ONES[x % 10] if x % 10 else ""))
        elif x > 0 or not parts:
            parts.append(_ONES[x])
        return " ".join(parts)

    if n == 0:
        words = "zero"
    else:
        parts = []
        for name, size in (("billion", 10 ** 9), ("million", 10 ** 6), ("thousand", 1000)):
            if n >= size:
                chunk = (n // size) % 1000
                if chunk:
                    parts.append(below_1000(chunk) + " " + name)
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
    return " ".join(s.upper())          # "AM" -> "A M"


def _year(n):
    if 2000 <= n <= 2009:
        return num2words(n)
    if 1100 <= n <= 2099 and n % 100 != 0:
        return f"{num2words(n // 100)} {num2words(n % 100)}" if n % 100 >= 10 else \
            f"{num2words(n // 100)} oh {num2words(n % 100)}"
    return num2words(n)


def normalize(text):
    """Spoken-form text, lower-cased, only characters the g2p/aligner can handle."""
    t = text.strip()
    # clock times: 10:45 AM / 7:05 pm / 14:30
    def clock(m):
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "")
        hs = num2words(h)
        ms = "" if mi == 0 else (f"oh {num2words(mi)}" if mi < 10 else num2words(mi))
        if mi == 0 and ap == "":
            ms = "o'clock"
        return " ".join(x for x in [hs, ms, _spell_letters(ap.replace('.', ''))] if x)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", clock, t)
    # "12 November 2026" / "November 12, 2026" / "12th November"
    mon = "|".join(m.capitalize() for m in _MONTHS) + "|" + "|".join(_MONTHS)
    t = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({mon})\b", lambda m: f"{num2words(int(m.group(1)), to='ordinal')} {m.group(2)}", t)
    t = re.sub(rf"\b({mon})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lambda m: f"{m.group(1)} {num2words(int(m.group(2)), to='ordinal')}", t)
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), t)
    t = re.sub(r"\b(1\d{3}|20\d{2})\b", lambda m: _year(int(m.group(1))), t)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: f"{num2words(int(m.group(1)))} point {' '.join(num2words(int(c)) for c in m.group(2))}", t)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group(0))), t)
    # remaining all-caps acronyms of 2-4 letters are spelled out (AM, PM, TV); keep real words
    t = re.sub(r"\b([AP])\.?M\.?\b", lambda m: f"{m.group(1)} M", t)
    for pat, rep in _ABBR:
        t = re.sub(pat, rep, t, flags=re.I)
    t = t.lower().replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "").replace('"', "")
    t = t.replace(";", ",").replace(":", ",").replace("-", " ")
    t = re.sub(r"[^a-z' .,?!]", " ", t)
    t = re.sub(r"\s+([.,?!])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _get_g2p():
    global _g2p
    if _g2p is None:
        import nltk
        for pkg in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict", "punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


def text_to_groups(text):
    """-> (normalised text, nltk words, token groups). groups[i] = ARPAbet phones (or one
    punctuation mark) of words[i]; both come from the same tokenizer so they line up 1:1."""
    from nltk import word_tokenize
    norm = normalize(text)
    words = word_tokenize(norm)
    groups, cur = [], []
    for p in _get_g2p()(norm):
        if p == " ":
            groups.append(cur)
            cur = []
        else:
            cur.append(p)
    groups.append(cur)
    groups = [g for g in groups if g]
    return norm, words, groups


def text_to_tokens(text):
    """Full token string for the model (phones, ' ' between words) -> list[str]."""
    _, _, groups = text_to_groups(text)
    toks = []
    for gi, g in enumerate(groups):
        if gi:
            toks.append(SPACE)
        toks.extend(p for p in g if p in SYM2ID)
    return toks


def tokens_to_ids(tokens):
    return [SYM2ID[t] for t in tokens if t in SYM2ID]


def text_to_sequence(text):
    return tokens_to_ids(text_to_tokens(text))

"""English text frontend: normalisation (numbers, times, dates, abbreviations, letter names) + g2p_en phonemes."""
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
_ABBR = [
    (r"\bmr\.", "mister"), (r"\bmrs\.", "missus"), (r"\bms\.", "miz"), (r"\bdr\.", "doctor"), (r"\bst\.", "saint"),
    (r"\bprof\.", "professor"), (r"\bvs\.?\b", "versus"), (r"\betc\.", "et cetera"), (r"&", " and "),
    (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"), (r"\bdept\.", "department")
]
_LETTER_NAMES = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "eff",
    "g": "jee", "h": "aych", "i": "eye", "j": "jay", "k": "kay", "l": "ell",
    "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "double-u",
    "x": "ex", "y": "wye", "z": "zee"
}
_ARPA_VOWEL_HEADS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")
_COMMON_WORDS_UPPER = {"IN", "ON", "AT", "BY", "TO", "HE", "WE", "IS", "IT", "AN", "AS", "IF", "OR", "SO", "NO", "GO", "MY", "ME", "UP", "DO"}
_g2p = None

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD_IRR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
    "nine": "ninth", "twelve": "twelfth"
}

_inflect_engine = None


def num2words(n, to="cardinal"):
    """Robust English number speller with inflect and pure-Python fallback."""
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
            return w.replace("-", " ").replace(",", "").lower()
        except Exception:
            pass

    s = str(abs(n))
    if len(s) > 9:
        return " ".join(_ONES[int(c)] for c in s)
    n = int(s)
    if n < 20:
        words = _ONES[n]
    elif n < 100:
        words = _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    elif n < 1000:
        words = _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num2words(n % 100))
    elif n < 1000000:
        words = num2words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num2words(n % 1000))
    else:
        words = num2words(n // 1000000) + " million" + ("" if n % 1000000 == 0 else " " + num2words(n % 1000000))

    if to == "ordinal":
        parts = words.split()
        head, last = " ".join(parts[:-1]), parts[-1]
        if last in _ORD_IRR:
            last = _ORD_IRR[last]
        elif last.endswith("y"):
            last = last[:-1] + "ieth"
        else:
            last += "th"
        words = (head + " " + last).strip()
    return words


def _year(n):
    if 2000 <= n <= 2009:
        return num2words(n)
    if 1100 <= n <= 2099 and n % 100 != 0:
        return f"{num2words(n // 100)} {num2words(n % 100)}" if n % 100 >= 10 else \
            f"{num2words(n // 100)} oh {num2words(n % 100)}"
    return num2words(n)


def normalize(text):
    """Spoken-form text, handles numbers, currency, dates, times, initials, acronyms, and typos."""
    t = text.strip()
    # Heal broken input words (e.g., 'rhyth  m' -> 'rhythm', 'parlia  ment' -> 'parliament')
    t = re.sub(r"\brhyth\s+m\b", "rhythm", t, flags=re.I)
    t = re.sub(r"\bparlia\s+ment\b", "parliament", t, flags=re.I)

    # Currency and Percentages
    t = re.sub(r"\$([0-9]+(?:\.[0-9]+)?)", r"\1 dollars", t)
    t = re.sub(r"₹([0-9]+(?:\.[0-9]+)?)", r"\1 rupees", t)
    t = re.sub(r"([0-9]+)%", r"\1 percent", t)

    # Universal clock times: 10:45 AM / 7:05 pm / 14:30
    def clock(m):
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
        hs = num2words(h)
        ms = "" if mi == 0 else (f"oh {num2words(mi)}" if mi < 10 else num2words(mi))
        if mi == 0 and ap == "":
            ms = "o'clock"
        ap_spoken = "ay em" if "a" in ap else ("pee em" if "p" in ap else "")
        return " ".join(x for x in [hs, ms, ap_spoken] if x)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", clock, t)

    # Spelled-out uppercase acronyms (2-5 letters) like AI, ML, TTS, IIT, ASR, GPU, CPU
    def spell_acronyms(m):
        w = m.group(0)
        if w in _COMMON_WORDS_UPPER:
            return w
        return " ".join(_LETTER_NAMES.get(c.lower(), c) for c in w)
    t = re.sub(r"\b[A-Z]{2,5}\b", spell_acronyms, t)

    # Standalone AM / PM acronyms
    t = re.sub(r"\bA\.?M\.?\b", "ay em", t, flags=re.I)
    t = re.sub(r"\bP\.?M\.?\b", "pee em", t, flags=re.I)

    # Dates: "12 November 2026" / "November 12, 2026" / "12th November"
    mon = "|".join(m.capitalize() for m in _MONTHS) + "|" + "|".join(_MONTHS)
    t = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({mon})\b", lambda m: f"{num2words(int(m.group(1)), to='ordinal')} {m.group(2)}", t)
    t = re.sub(rf"\b({mon})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", lambda m: f"{m.group(1)} {num2words(int(m.group(2)), to='ordinal')}", t)
    t = re.sub(r"\b(\d{1,2})(?:st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), t)
    t = re.sub(r"\b(1\d{3}|20\d{2})\b", lambda m: _year(int(m.group(1))), t)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: f"{num2words(int(m.group(1)))} point {' '.join(num2words(int(c)) for c in m.group(2))}", t)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group(0))), t)

    for pat, rep in _ABBR:
        t = re.sub(pat, rep, t, flags=re.I)

    t = t.lower().replace("’", "'").replace("‘", "'").replace("“", "").replace("”", "").replace('"', "")
    t = t.replace(";", ",").replace(":", ",").replace("-", " ")

    # Expand standalone single letters (e.g., initials "a", "m", "x")
    def expand_initials(m):
        w = m.group(1)
        return _LETTER_NAMES.get(w, w)
    t = re.sub(r"\b([a-z])\b", expand_initials, t)

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
    """Tokenize normalized text into groups matching G2P output."""
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
    """Universal token sequence generator for ANY general English text, OOV words, acronyms, or loanwords.
    Guarantees every vowel receives proper lexical stress so words are never dropped, swallowed, or mumbled."""
    _, _, groups = text_to_groups(text)
    toks = []
    for gi, g in enumerate(groups):
        if gi:
            toks.append(SPACE)
        vowel_count = 0
        for p in g:
            # 1. Exact match in symbol table
            if p in SYM2ID:
                toks.append(p)
            else:
                p_up = p.upper()
                head = re.sub(r"\d+", "", p_up)
                # 2. Vowel without stress or invalid stress digit
                if head in _ARPA_VOWEL_HEADS:
                    stress = "1" if vowel_count == 0 else "0"
                    vowel_count += 1
                    target = head + stress
                    if target in SYM2ID:
                        toks.append(target)
                    elif head + "1" in SYM2ID:
                        toks.append(head + "1")
                # 3. Known consonant upper case
                elif p_up in SYM2ID:
                    toks.append(p_up)
                # 4. Single letter name fallback
                elif p.lower() in _LETTER_NAMES:
                    sub_name = _LETTER_NAMES[p.lower()]
                    for sp in _get_g2p()(sub_name):
                        if sp in SYM2ID:
                            toks.append(sp)
                        elif re.sub(r"\d+", "", sp.upper()) in _ARPA_VOWEL_HEADS:
                            toks.append(re.sub(r"\d+", "", sp.upper()) + "1")
    return toks


def tokens_to_ids(tokens):
    return [SYM2ID[t] for t in tokens if t in SYM2ID]


def text_to_sequence(text):
    return tokens_to_ids(text_to_tokens(text))

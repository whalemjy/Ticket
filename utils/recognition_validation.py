"""Fallback validation helpers for OCR operation-ticket results."""

import re


NON_ENTRY_STATUS_TEXTS = {"已执行", "未执行", "作废"}


def normalized_text(text):
    """Normalize OCR text for comparisons while retaining its content."""
    return re.sub(r"\s+", "", str(text or ""))


def sequence_value(text, *, allow_wrappers=False):
    """Return a sequence number, tolerating common punctuation wrappers."""
    normalized = normalized_text(text)
    if re.fullmatch(r"\d{1,3}", normalized):
        return int(normalized)
    if allow_wrappers:
        match = re.fullmatch(
            r"[.:：。．、,，;；()（）\[\]【】号序]*([0-9]{1,3})"
            r"[.:：。．、,，;；()（）\[\]【】号序]*",
            normalized,
        )
        if match is not None:
            return int(match.group(1))
    return None


def is_non_entry_status_text(text):
    normalized = normalized_text(text).strip(
        "()（）[]【】.:：。．、,，;；!?！？√✓✔∨"
    )
    return normalized in NON_ENTRY_STATUS_TEXTS


def split_merged_sequence_entry(record, sequence_anchor):
    """Split a sequence prefix only when its box crosses the sequence column."""
    normalized = normalized_text(record.get("text"))
    match = re.fullmatch(r"(\d{1,3})(\D.*)", normalized)
    if match is None:
        return None
    crosses = record["left"] <= sequence_anchor["center_x"] <= record["right"]
    extends = record["right"] > sequence_anchor["right"]
    if not crosses or not extends:
        return None
    return int(match.group(1)), match.group(2)


def filter_continuous_recoveries(sequence_records, recovered):
    """Keep recovered values only when each primary-sequence gap is complete."""
    recovered = sorted(recovered, key=lambda item: item[1]["center_y"])
    if not recovered:
        return []
    existing = sorted(sequence_records, key=lambda item: item[1]["center_y"])
    if not existing:
        values = [sequence for sequence, _, _ in recovered]
        if values[0] < 1 or any(
            current != previous + 1
            for previous, current in zip(values, values[1:])
        ):
            return []
        return recovered

    accepted = []
    first_sequence, first_record, _ = existing[0]
    before = [item for item in recovered if item[1]["center_y"] < first_record["center_y"]]
    before_values = [sequence for sequence, _, _ in before]
    expected_before = list(range(first_sequence - len(before_values), first_sequence))
    if expected_before and expected_before[0] >= 1 and before_values == expected_before:
        accepted.extend(before)

    for previous, current in zip(existing, existing[1:]):
        previous_sequence, previous_record, _ = previous
        current_sequence, current_record, _ = current
        between = [
            item for item in recovered
            if previous_record["center_y"] < item[1]["center_y"] < current_record["center_y"]
        ]
        between_values = [sequence for sequence, _, _ in between]
        if between_values == list(range(previous_sequence + 1, current_sequence)):
            accepted.extend(between)

    last_sequence, last_record, _ = existing[-1]
    after = [item for item in recovered if item[1]["center_y"] > last_record["center_y"]]
    after_values = [sequence for sequence, _, _ in after]
    expected_after = list(range(last_sequence + 1, last_sequence + 1 + len(after_values)))
    if after_values == expected_after:
        accepted.extend(after)
    return accepted

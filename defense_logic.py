import json
import re
from pathlib import Path

from validate_mission import (
    ValidationResult,
    merge_results,
    normalize_text,
    validate_busbar_transfer,
    validate_grounding_prerequisites,
    validate_hot_standby_to_cold_standby,
    validate_main_transformer_transition,
    validate_sequence_control_ticket,
)


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _has_direction(text: str, source: str, target: str) -> bool:
    return bool(re.search(rf"由[^，,。；;]*{source}[^，,。；;]*(?:转|至|到|切至|变为)[^，,。；;]*{target}", text))


def _classify_clause(clause: str) -> str:
    text = normalize_text(clause)
    if text.startswith("核对"):
        return "audit_only"
    if re.search(r"主变|变压器", text) and (
        _has_direction(text, r"运行", r"热备(?:用)?")
        or _has_direction(text, r"热备(?:用)?", r"运行")
    ):
        return "main_transformer_running_hot_standby"
    if _has_direction(text, r"热备(?:用)?", r"冷备(?:用)?"):
        return "hot_standby_to_cold_standby"
    if _has_direction(text, r"冷备(?:用)?", r"热备(?:用)?"):
        return "cold_standby_to_hot_standby"
    if _has_direction(text, r"冷备(?:用)?", r"(?:开关及线路)?检修"):
        return "cold_standby_to_maintenance"
    if _has_direction(text, r"(?:开关及线路)?检修", r"冷备(?:用)?"):
        return "maintenance_to_cold_standby"
    if _has_direction(text, r"运行", r"热备(?:用)?"):
        return "running_to_hot_standby"
    if _has_direction(text, r"热备(?:用)?", r"运行"):
        return "hot_standby_to_running"
    if re.search(r"母线.*(?:倒至|倒换|切换|转移)|(?:倒至|倒换|切换|转移).*母线", text):
        return "busbar_transfer"
    if re.search(r"^(?:拉开|断开|分闸).*(?:开关|断路器|刀闸)", text):
        return "switch_open"
    return "others"


def judge_type(mission: str) -> list[dict[str, str]]:
    """Split a mission and preserve state-transition direction."""
    clauses = [
        item.strip()
        for item in re.split(r"[，,、;；。]+", str(mission or ""))
        if item.strip()
    ]
    return [
        {"submission": clause, "type": _classify_clause(clause)}
        for clause in clauses
    ]


def validate_ticket(data: dict) -> ValidationResult:
    mission = str(data.get("mission", ""))
    entries = data.get("entries", {})
    types = {item["type"] for item in judge_type(mission)}
    results: list[ValidationResult] = []

    if "顺控" in normalize_text(mission):
        return merge_results("ticket", [validate_sequence_control_ticket(entries, mission)])

    # Grounding safety is action-driven and applies regardless of mission title.
    results.append(validate_grounding_prerequisites(entries))
    if "hot_standby_to_cold_standby" in types:
        results.append(validate_hot_standby_to_cold_standby(entries))
    if "main_transformer_running_hot_standby" in types:
        results.append(validate_main_transformer_transition(entries, mission))
    if "busbar_transfer" in types:
        results.append(validate_busbar_transfer(entries, mission))

    return merge_results("ticket", results)


def process_ticket_details(src: str | Path) -> ValidationResult:
    return validate_ticket(read_json(src))


def process_ticket(src: str | Path) -> bool:
    """Return True only when the ticket passes without manual-review findings."""
    return process_ticket_details(src).passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a station operation ticket")
    parser.add_argument("ticket", type=Path, help="Path to a command JSON file")
    args = parser.parse_args()
    result = process_ticket_details(args.ticket)
    print(json.dumps({
        "status": result.status,
        "findings": [finding.__dict__ for finding in result.findings],
    }, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.passed else 1)

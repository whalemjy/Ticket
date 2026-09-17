"""Sequence-aware interlock checks for OCR operation-ticket entries."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
import unicodedata


CHECK_WORDS = r"检查|核对|确认|验明|查验"
OPEN_WORDS = r"拉开|断开|分闸"
CLOSE_WORDS = r"合上|合闸"
COUPLER_WORDS = r"母联|母线分段|分段|桥"
STATE_WORDS = r"开关及线路检修|热备(?:用)?|冷备(?:用)?|运行|检修"


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    message: str
    step: str | None = None
    entry: str | None = None
    severity: str = "error"


@dataclass(frozen=True)
class ValidationResult:
    rule: str
    findings: tuple[ValidationFinding, ...] = ()

    @property
    def status(self) -> str:
        if any(item.severity == "error" for item in self.findings):
            return "blocked"
        if self.findings:
            return "review"
        return "pass"

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def __bool__(self) -> bool:
        return self.passed


def merge_results(rule: str, results: Iterable[ValidationResult]) -> ValidationResult:
    findings: list[ValidationFinding] = []
    seen: set[tuple] = set()
    for result in results:
        for finding in result.findings:
            key = (finding.code, finding.step, finding.entry, finding.severity)
            if key not in seen:
                findings.append(finding)
                seen.add(key)
    return ValidationResult(rule, tuple(findings))


def normalize_text(value: object) -> str:
    """Normalize harmless OCR/typographic variance without changing device IDs."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", "", text)
    # OCR occasionally leaves one marker immediately before the operation verb.
    text = re.sub(
        rf"^[A-Za-z0-9]{{1,2}}(?=(?:{CHECK_WORDS}|{OPEN_WORDS}|{CLOSE_WORDS}|投入|停用|将|选择|装设|拆除))",
        "",
        text,
    )
    return text


def _all_ordered_entries(entries: Mapping | Iterable | str) -> list[tuple[str, str, str]]:
    if isinstance(entries, str):
        raw_items = [("1", entries)] if entries.strip() else []
    elif isinstance(entries, Mapping):
        keys = list(entries)
        if keys and all(str(key).isdigit() for key in keys):
            keys.sort(key=lambda key: int(str(key)))
        raw_items = [(str(key), entries[key]) for key in keys]
    else:
        raw_items = [(str(index), value) for index, value in enumerate(entries, start=1)]
    return [
        (step, str(raw).strip(), normalize_text(raw))
        for step, raw in raw_items
        if str(raw).strip()
    ]


def _ordered_entries(entries: Mapping | Iterable | str) -> list[tuple[str, str, str]]:
    # "核对..." is an audit record, not an operation to validate and not
    # evidence that can satisfy a prerequisite for a later operation. The
    # sequence-control validator deliberately uses _all_ordered_entries because
    # its template explicitly requires a "核对当前运行方式" field.
    return [
        item for item in _all_ordered_entries(entries)
        if not _is_audit_only(item[2])
    ]


def _is_confirmation(text: str) -> bool:
    return bool(re.match(rf"^(?:{CHECK_WORDS})", text))


def _is_audit_only(text: str) -> bool:
    """Return whether an entry must be excluded from all interlock checks."""
    # Accept the common sequence-number residue emitted by OCR (``1.`` / ``1、``)
    # before the actual entry text.
    return bool(re.match(r"^(?:[（(【\[]?\d{1,3}[）)】\].、．.:：-]?)?核对", text))


def _is_power_or_secondary_device(text: str) -> bool:
    return bool(re.search(r"电源|空气开关|压板|把手|切换开关|二次插头|保护装置", text))


def _canonical_number(value: str) -> str:
    stripped = value.lstrip("0")
    return stripped or "0"


def _device_parts(text: str) -> list[tuple[str, str | None]]:
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?P<base>0*\d{1,4})(?P<suffix>-[A-Za-z]?\d+)?"
        r"(?=(?:中性点)?(?:接地刀闸|地刀|刀闸|隔离开关|开关手车|断路器|开关|间隔))",
        re.IGNORECASE,
    )
    result = []
    for match in pattern.finditer(text):
        base = _canonical_number(match.group("base"))
        suffix = match.group("suffix")
        result.append((base, suffix.upper() if suffix else None))
    return result


def _device_base(text: str) -> str | None:
    parts = _device_parts(text)
    return parts[-1][0] if parts else None


def _branch_number(text: str) -> str | None:
    parts = _device_parts(text)
    if not parts or not parts[-1][1]:
        return None
    match = re.search(r"(\d+)$", parts[-1][1])
    return _canonical_number(match.group(1)) if match else None


def _transformer_ids(text: str) -> set[str]:
    ids = {
        _canonical_number(item)
        for item in re.findall(r"#?\s*(\d+)\s*号?主变", text)
    }
    pair = re.search(r"#?(\d+)[、,，]#?(\d+)主变", text)
    if pair:
        ids.update(_canonical_number(item) for item in pair.groups())
    return ids


def _breaker_open_confirmation(text: str) -> bool:
    return (
        _is_confirmation(text)
        and bool(re.search(r"开关|断路器", text))
        and not bool(re.search(r"手车|空气开关|电源", text))
        and bool(re.search(r"分闸|分位|拉开位置|断开位置|确已拉开", text))
    )


def _disconnector_operation(text: str) -> bool:
    if _is_confirmation(text) or _is_audit_only(text) or _is_power_or_secondary_device(text):
        return False
    knife = bool(re.search(OPEN_WORDS + "|" + CLOSE_WORDS, text)) and bool(
        re.search(r"刀闸|隔离开关", text)
    ) and not bool(re.search(r"接地刀闸|地刀", text))
    handcart = bool(re.search(r"开关手车|刀闸手车", text)) and bool(
        re.search(r"拉至|推至|摇至|移至|由.+(?:转|至)", text)
    )
    return knife or handcart


def _grounding_close(text: str) -> bool:
    return (
        not _is_confirmation(text)
        and not _is_power_or_secondary_device(text)
        and bool(re.search(CLOSE_WORDS, text))
        and bool(re.search(r"接地刀闸|地刀", text))
    )


def _ground_wire_install(text: str) -> bool:
    return (
        not _is_confirmation(text)
        and bool(re.search(r"装设|装上", text))
        and bool(re.search(r"接地线|接地一组|接地$", text))
    )


def _breaker_operation(text: str) -> bool:
    return (
        not _is_confirmation(text)
        and not _is_audit_only(text)
        and not _is_power_or_secondary_device(text)
        and bool(re.search(OPEN_WORDS + "|" + CLOSE_WORDS, text))
        and bool(re.search(r"开关|断路器", text))
        and not bool(re.search(r"刀闸|地刀", text))
    )


def _neutral_closed_confirmation(text: str) -> bool:
    return (
        _is_confirmation(text)
        and "中性点" in text
        and bool(re.search(r"接地刀闸|地刀", text))
        and bool(re.search(r"合闸|合位|合上位置|确已合好|确已合上", text))
    )


def _neutral_open_operation(text: str) -> bool:
    return (
        not _is_confirmation(text)
        and not _is_power_or_secondary_device(text)
        and "中性点" in text
        and bool(re.search(r"接地刀闸|地刀", text))
        and bool(re.search(OPEN_WORDS, text))
    )


def _is_indirect_voltage_test(text: str) -> bool:
    # Missing right brackets and trailing OCR characters are intentionally tolerated.
    return "间接验电" in text and _is_confirmation(text)


def _is_direct_voltage_test(text: str) -> bool:
    return (
        "间接验电" not in text
        and bool(re.search(r"验明|验电", text))
        and bool(re.search(r"无电压|无电|不带电", text))
    )


def _voltage_test_matches(test_text: str, target_text: str) -> bool | None:
    target_base = _device_base(target_text)
    test_base = _device_base(test_text)
    if not target_base or not test_base:
        return None
    if target_base != test_base:
        return False
    target_branch = _branch_number(target_text)
    test_branch = _branch_number(test_text)
    if not target_branch or not test_branch:
        return True
    return target_branch == test_branch


def _canonical_state(state: str) -> str:
    state = normalize_text(state)
    if state == "热备":
        return "热备用"
    if state == "冷备":
        return "冷备用"
    return state


def _extract_transition(text: str) -> tuple[str, str, str] | None:
    match = re.search(
        rf"由(?P<source>{STATE_WORDS})(?:转为?|至|到|变为)(?P<target>{STATE_WORDS})",
        normalize_text(text),
    )
    if not match:
        return None
    return (
        match.string[:match.start()],
        _canonical_state(match.group("source")),
        _canonical_state(match.group("target")),
    )


def _canonical_sequence_device(value: str) -> str:
    text = normalize_text(value)
    text = re.sub(r"^.*?顺控(?:主机)?(?:[:：])?", "", text)
    text = re.sub(r"^在?顺控主机", "", text)
    text = re.sub(r"^\d+(?:\.\d+)?kV", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:开关|断路器|间隔)$", "", text)
    return text.strip("，,。；;:：")


def _sequence_device_from_entry(text: str, kind: str) -> str | None:
    patterns = {
        "new": r"新建(?P<device>.+?)(?:间隔)?操作任务",
        "current": r"核对(?P<device>.+?)当前运行方式为",
        "target": r"选择(?P<device>.+?)目标运行方式为",
        "generated": r"生成(?P<device>.+?)由(?:" + STATE_WORDS + r")",
    }
    match = re.search(patterns[kind], text)
    if not match:
        return None
    return _canonical_sequence_device(match.group("device"))


def _sequence_name_matches_mission(device: str, mission_subject: str) -> bool:
    subject = _canonical_sequence_device(mission_subject)
    if not subject:
        return False
    if device == subject or device.startswith(subject):
        return True
    # A compound mission can execute one interval through sequence control and
    # continue with related manual operations. Ignore its list separators when
    # checking that the selected interval belongs to the mission subject.
    compact_subject = re.sub(r"[及与、,，]", "", subject)
    return device in compact_subject


def validate_sequence_control_ticket(entries, mission: str) -> ValidationResult:
    """Validate the setup fields of a one-key sequence-control operation."""
    findings: list[ValidationFinding] = []
    mission_text = normalize_text(mission)
    mission_transition = _extract_transition(mission_text)
    if mission_transition is None:
        return ValidationResult("sequence_control", (ValidationFinding(
            "SEQUENCE_MISSION_TRANSITION_MISSING",
            "一键顺控操作任务中未识别到明确的源运行方式和目标运行方式。",
        ),))

    mission_subject, expected_source, expected_target = mission_transition
    mission_subject = re.sub(r"^.*?顺控(?:[:：])?", "", mission_subject)
    records: dict[str, tuple[int, str, str, str | None]] = {}
    ordered = _all_ordered_entries(entries)

    for index, (step, raw, text) in enumerate(ordered):
        kind = None
        state = None
        if "新建" in text and "操作任务" in text:
            kind = "new"
        elif "当前运行方式为" in text:
            kind = "current"
            match = re.search(rf"当前运行方式为(?P<state>{STATE_WORDS})", text)
            state = _canonical_state(match.group("state")) if match else None
        elif "目标运行方式为" in text:
            kind = "target"
            match = re.search(rf"目标运行方式为(?P<state>{STATE_WORDS})", text)
            state = _canonical_state(match.group("state")) if match else None
        elif "生成" in text and "操作票" in text:
            kind = "generated"
        if kind and kind not in records:
            records[kind] = (index, step, raw, state)

    required = {
        "current": ("SEQUENCE_CURRENT_MODE_MISSING", "未找到一键顺控的当前运行方式核对项。"),
        "target": ("SEQUENCE_TARGET_MODE_MISSING", "未找到一键顺控的目标运行方式选择项。"),
        "generated": ("SEQUENCE_GENERATED_TASK_MISSING", "未找到一键顺控生成操作票的任务项。"),
    }
    for kind, (code, message) in required.items():
        if kind not in records:
            findings.append(ValidationFinding(code, message))

    devices: list[tuple[str, str, str]] = []
    for kind in ("new", "current", "target", "generated"):
        if kind not in records:
            continue
        _, step, raw, _ = records[kind]
        device = _sequence_device_from_entry(normalize_text(raw), kind)
        if device:
            devices.append((step, raw, device))
        else:
            findings.append(ValidationFinding(
                "SEQUENCE_DEVICE_NAME_MISSING",
                "无法从一键顺控操作项目中提取设备双重名称。",
                step,
                raw,
            ))

    if devices:
        reference_device = devices[0][2]
        mission_device = _canonical_sequence_device(mission_subject)
        core_steps = {step for step, _, _ in devices}
        for step, raw, device in devices:
            same_as_reference = device == reference_device
            matches_mission = _sequence_name_matches_mission(device, mission_subject)
            if not same_as_reference or not matches_mission:
                expected_device = reference_device if not same_as_reference else mission_device
                findings.append(ValidationFinding(
                    "SEQUENCE_DEVICE_NAME_MISMATCH",
                    f"一键顺控设备双重名称应与操作任务一致，期望 {expected_device}，实际 {device}。",
                    step,
                    raw,
                ))

        block_start = min(record[0] for record in records.values())
        block_end = len(ordered) - 1
        for index, (_, _, text) in enumerate(ordered[block_start:], start=block_start):
            if re.search(r"(?:顺控主机)?执行正确", text):
                block_end = index
                break
        for _, step, raw in (
            (index, step, raw)
            for index, (step, raw, _) in enumerate(ordered)
            if block_start <= index <= block_end and step not in core_steps
        ):
            text = normalize_text(raw)
            mentions_primary_device = bool(
                re.search(r"\d.*(?:开关|断路器|刀闸|间隔)|(?:开关|断路器|刀闸|间隔).*\d", text)
            )
            if mentions_primary_device and reference_device not in text:
                findings.append(ValidationFinding(
                    "SEQUENCE_DEVICE_NAME_MISMATCH",
                    f"一键顺控区段中的设备双重名称应为 {reference_device}。",
                    step,
                    raw,
                ))

    if "current" in records:
        _, step, raw, actual_source = records["current"]
        if actual_source != expected_source:
            findings.append(ValidationFinding(
                "SEQUENCE_CURRENT_MODE_MISMATCH",
                f"当前运行方式应为 {expected_source}，实际识别为 {actual_source or '无法识别'}。",
                step,
                raw,
            ))
    if "target" in records:
        _, step, raw, actual_target = records["target"]
        if actual_target != expected_target:
            findings.append(ValidationFinding(
                "SEQUENCE_TARGET_MODE_MISMATCH",
                f"目标运行方式应为 {expected_target}，实际识别为 {actual_target or '无法识别'}。",
                step,
                raw,
            ))
    if "generated" in records:
        _, step, raw, _ = records["generated"]
        generated_transition = _extract_transition(raw)
        if generated_transition is None or generated_transition[1:] != (expected_source, expected_target):
            actual = "无法识别" if generated_transition is None else f"{generated_transition[1]}转{generated_transition[2]}"
            findings.append(ValidationFinding(
                "SEQUENCE_GENERATED_TASK_MISMATCH",
                f"生成操作票的任务应为 {expected_source}转{expected_target}，实际为 {actual}。",
                step,
                raw,
            ))

    if all(kind in records for kind in required):
        positions = [records[kind][0] for kind in ("current", "target", "generated")]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            findings.append(ValidationFinding(
                "SEQUENCE_SETUP_ORDER_INVALID",
                "一键顺控应依次核对当前运行方式、选择目标运行方式、生成操作票。",
            ))

    return ValidationResult("sequence_control", tuple(findings))


def validate_hot_standby_to_cold_standby(entries) -> ValidationResult:
    findings: list[ValidationFinding] = []
    open_breakers: set[str] = set()
    for step, raw, text in _ordered_entries(entries):
        if _breaker_open_confirmation(text):
            base = _device_base(text)
            if base:
                open_breakers.add(base)
        if not _disconnector_operation(text):
            continue
        base = _device_base(text)
        if not base:
            findings.append(ValidationFinding(
                "DEVICE_UNCERTAIN",
                "无法可靠提取刀闸或手车对应的断路器编号，请人工复核设备名称。",
                step,
                raw,
                "review",
            ))
        elif base not in open_breakers:
            findings.append(ValidationFinding(
                "BREAKER_NOT_CONFIRMED_OPEN",
                f"操作设备 {base} 的刀闸或手车前，未找到同编号断路器分位确认。",
                step,
                raw,
            ))
    return ValidationResult("hot_standby_to_cold_standby", tuple(findings))


def validate_grounding_prerequisites(entries, minimum_indirect_tests: int = 2) -> ValidationResult:
    findings: list[ValidationFinding] = []
    ordered = _ordered_entries(entries)
    last_ground_by_base: dict[str, int] = {}
    for index, (step, raw, text) in enumerate(ordered):
        if not (_grounding_close(text) or _ground_wire_install(text)):
            continue
        base = _device_base(text)
        if not base:
            findings.append(ValidationFinding(
                "GROUNDING_LOCATION_UNCERTAIN",
                "无法可靠提取接地操作位置，请人工核对验电位置与接地位置是否一致。",
                step,
                raw,
                "review",
            ))
            continue
        start = last_ground_by_base.get(base, -1) + 1
        prior = ordered[start:index]
        direct_count = 0
        indirect_count = 0
        uncertain_count = 0
        for _, _, prior_text in prior:
            if not (_is_direct_voltage_test(prior_text) or _is_indirect_voltage_test(prior_text)):
                continue
            matched = _voltage_test_matches(prior_text, text)
            if matched is None:
                uncertain_count += 1
            elif matched and _is_direct_voltage_test(prior_text):
                direct_count += 1
            elif matched:
                indirect_count += 1
        if direct_count < 1 and indirect_count < minimum_indirect_tests:
            if uncertain_count:
                findings.append(ValidationFinding(
                    "VOLTAGE_TEST_LOCATION_UNCERTAIN",
                    "发现验电记录，但无法可靠关联到本次接地位置，请人工复核设备名称。",
                    step,
                    raw,
                    "review",
                ))
            else:
                findings.append(ValidationFinding(
                    "INSUFFICIENT_VOLTAGE_TEST",
                    f"接地操作前需至少 1 条同位置直接验电，或至少 {minimum_indirect_tests} 条同位置间接验电。",
                    step,
                    raw,
                ))
        last_ground_by_base[base] = index
    return ValidationResult("grounding_prerequisites", tuple(findings))


def validate_main_transformer_transition(entries, mission: str = "") -> ValidationResult:
    findings: list[ValidationFinding] = []
    mission_transformers = _transformer_ids(normalize_text(mission))
    confirmed: set[str] = set()
    planned_main_breaker = False
    direct_main_breaker = False

    for step, raw, text in _ordered_entries(entries):
        transformer_ids = _transformer_ids(text)
        if _neutral_closed_confirmation(text):
            confirmed.update(transformer_ids)
        if _neutral_open_operation(text):
            confirmed.difference_update(transformer_ids)

        if _is_confirmation(text) and re.search(r"操作任务正确", text) and re.search(
            rf"(?:{OPEN_WORDS}|{CLOSE_WORDS}).*(?:主变).*(?:开关|断路器)", text
        ):
            planned_main_breaker = True

        if not _breaker_operation(text):
            continue
        involved = set(transformer_ids)
        if not involved and len(mission_transformers) == 1:
            involved = set(mission_transformers)
        if not involved:
            continue
        direct_main_breaker = True
        missing = involved - confirmed
        if missing:
            names = "、".join(f"#{item}主变" for item in sorted(missing))
            findings.append(ValidationFinding(
                "NEUTRAL_GROUND_NOT_CONFIRMED",
                f"分合 {names} 关联开关前，未确认同一主变中性点接地刀闸在合位。",
                step,
                raw,
            ))

    if planned_main_breaker and not direct_main_breaker:
        findings.append(ValidationFinding(
            "SEQUENCE_CONTROL_REVIEW",
            "票据通过顺控任务执行主变开关操作，需由顺控明细或实时设备状态确认中性点接地条件。",
            severity="review",
        ))
    return ValidationResult("main_transformer_transition", tuple(findings))


def _coupler_key(text: str) -> str | None:
    if not re.search(COUPLER_WORDS, text):
        return None
    return _device_base(text) or "generic"


def _bus_transfer_action(text: str, mission_is_transfer: bool) -> bool:
    if _is_confirmation(text) or _is_audit_only(text) or re.search(COUPLER_WORDS, text):
        return False
    explicit = bool(
        re.search(r"母线.*(?:倒至|倒换|切换|转移).*(?:刀闸|操作)", text)
        or re.search(r"执行.*母线.*(?:倒至|倒换|切换|转移)", text)
    )
    return explicit or (mission_is_transfer and _disconnector_operation(text))


def validate_busbar_transfer(entries, mission: str = "") -> ValidationResult:
    findings: list[ValidationFinding] = []
    mission_text = normalize_text(mission)
    mission_is_transfer = bool(re.search(r"母线.*(?:倒至|倒换|切换|转移)|(?:倒至|倒换|切换|转移).*母线", mission_text))
    breaker_closed: set[str] = set()
    knife_closed: set[str] = set()
    current_present: set[str] = set()
    interconnection_plate_in = False
    control_power_open = False

    for step, raw, text in _ordered_entries(entries):
        if _bus_transfer_action(text, mission_is_transfer):
            shared_devices = breaker_closed & knife_closed
            reliable = bool(current_present or shared_devices)
            missing = []
            if not reliable:
                missing.append("母联有电流，或同一母联开关和相关刀闸均在合位")
            if not interconnection_plate_in:
                missing.append("互联压板已投入")
            if not control_power_open:
                missing.append("母联控制电源已拉开")
            if missing:
                findings.append(ValidationFinding(
                    "BUSBAR_TRANSFER_PREREQUISITE_MISSING",
                    "母线刀闸操作前缺少条件：" + "；".join(missing) + "。",
                    step,
                    raw,
                ))

        key = _coupler_key(text)
        if key:
            confirmed = _is_confirmation(text)
            closed = bool(re.search(r"合闸|合位|合上位置|确已合好|确已合上", text))
            if confirmed and closed and re.search(r"开关|断路器", text) and not re.search(r"刀闸|手车", text):
                breaker_closed.add(key)
            if confirmed and closed and re.search(r"刀闸|隔离开关|刀闸手车", text):
                knife_closed.add(key)
            if confirmed and re.search(r"有电流|电流.*(?:正常|通过|不为零|大于0)", text):
                current_present.add(key)
            if not confirmed and re.search(r"拉开|断开|切断|停用", text) and "控制电源" in text:
                control_power_open = True
        if not _is_confirmation(text) and "投入" in text and re.search(r"互联.*压板|母联互联压板", text):
            interconnection_plate_in = True
    return ValidationResult("busbar_transfer", tuple(findings))


# Boolean compatibility wrappers used by the existing dispatcher.
def running_hot_standby_transition(entries) -> bool:
    return validate_main_transformer_transition(entries).passed


def hot_standby_cold_standby_transition(entries) -> bool:
    return validate_hot_standby_to_cold_standby(entries).passed


def cold_standby_to_maintenance(entries) -> bool:
    return validate_grounding_prerequisites(entries).passed


def main_transformer_running_to_hot_standby(entries) -> bool:
    return validate_main_transformer_transition(entries).passed


def busbar_transfer(entries) -> bool:
    return validate_busbar_transfer(entries).passed


def switch_open(entries) -> bool:
    return any(_breaker_operation(text) for _, _, text in _ordered_entries(entries))


def bus_restoration(entries) -> bool:
    return bool(_ordered_entries(entries))


def bus_operation(entries) -> bool:
    return bool(_ordered_entries(entries))


def others(entries) -> bool:
    return bool(_ordered_entries(entries))

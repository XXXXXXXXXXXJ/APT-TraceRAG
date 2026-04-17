import json
import os
from pathlib import Path

INPUT_DIR = Path("./sample_reports")
OUTPUT_DIR = Path("./R_behaviour")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Field trimming helpers ----------

def extract_tls(tls_list):
    if not isinstance(tls_list, list):
        return []

    extracted = []
    for item in tls_list:
        extracted.append({
            "ja3": item.get("ja3"),
            "ja3s": item.get("ja3s"),
            "version": item.get("version"),
            "issuer_cn": item.get("issuer", {}).get("CN"),
            "subject_cn": item.get("subject", {}).get("CN"),
        })
    return extracted


def extract_ids_alerts(alerts):
    if not isinstance(alerts, list):
        return []

    extracted = []
    for alert in alerts:
        severity = alert.get("alert_severity")
        if severity in {"low", "medium"}:
            continue

        extracted.append({
            "rule_id": alert.get("rule_id"),
            "rule_category": alert.get("rule_category"),
            "rule_msg": alert.get("rule_msg"),
            "rule_source": alert.get("rule_source"),
            "alert_severity": alert.get("alert_severity"),
        })
    return extracted

def extract_signature_matches(matches):
    """
    Keep only signatures with severity == IMPACT_SEVERITY_HIGH.
    """
    if not isinstance(matches, list):
        return []

    filtered = []
    for item in matches:
        if item.get("severity") == "IMPACT_SEVERITY_HIGH":
            filtered.append(item)

    return filtered

# ---------- Main extraction logic ----------

def extract_minimal_fields(data):
    """
    Safely extract a minimal field set from VT JSON.
    """
    fb = data.get("data", {})
    if not isinstance(fb, dict):
        return {}

    cleaned = {}

    # Direct fields
    simple_fields = [
        "command_executions",
        "processes_injected",
        "processes_killed",
        "invokes",
        "services_opened",
        "mutexes_created",
        "ja3_digests",
        "dns_lookups",
    ]

    for field in simple_fields:
        value = fb.get(field)
        if value:
            cleaned[field] = value

    # Trim TLS
    tls = extract_tls(fb.get("tls"))
    if tls:
        cleaned["tls"] = tls

    # Trim ids_alerts
    alerts = extract_ids_alerts(fb.get("ids_alerts"))
    if alerts:
        cleaned["ids_alerts"] = alerts
    
    sigs = extract_signature_matches(fb.get("signature_matches"))
    if sigs:
        cleaned["signature_matches"] = sigs

    return cleaned


# ---------- Batch processing ----------

def process_files():
    for json_file in INPUT_DIR.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cleaned = extract_minimal_fields(data)

            if not cleaned:
                continue

            output_path = OUTPUT_DIR / json_file.name
            with open(output_path, "w", encoding="utf-8") as out:
                json.dump(cleaned, out, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[!] Failed to process {json_file}: {e}")


if __name__ == "__main__":
    process_files()

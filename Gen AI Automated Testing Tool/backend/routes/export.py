"""
Export Route
GET /api/export/{document_id}?format=csv|json|jira
"""
import csv
import io
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from routes.generate import _results

router = APIRouter()


@router.get("/{document_id}")
def export(
    document_id: str,
    format: str = Query(default="csv", regex="^(csv|json|jira)$"),
):
    cases = _results.get(document_id, [])
    if not cases:
        raise HTTPException(status_code=404, detail="No test cases found for this document.")

    if format == "csv":
        return _export_csv(cases, document_id)
    elif format == "json":
        return _export_json(cases, document_id)
    elif format == "jira":
        return _export_jira(cases, document_id)


# ── Exporters ─────────────────────────────────────────────────────────────────

def _export_csv(cases, document_id):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Test ID", "Title", "Type", "Priority",
        "Preconditions", "Steps", "Expected Result",
        "Linked Requirement", "Tags", "Status",
    ])

    for tc in cases:
        steps_text = " | ".join(
            f"{s.step_number}. {s.action} → {s.expected}" for s in tc.steps
        )
        writer.writerow([
            tc.id, tc.title, tc.test_type.value, tc.priority.value,
            "; ".join(tc.preconditions),
            steps_text,
            tc.expected_result,
            tc.linked_requirement,
            ", ".join(tc.tags),
            tc.status,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=test_cases_{document_id[:8]}.csv"},
    )


def _export_json(cases, document_id):
    data = [tc.model_dump() for tc in cases]
    return StreamingResponse(
        iter([json.dumps(data, indent=2)]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=test_cases_{document_id[:8]}.json"},
    )


def _export_jira(cases, document_id):
    """
    Jira bulk import CSV format (compatible with Jira's CSV importer).
    Each test case becomes a Jira issue of type "Test".
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Summary", "Issue Type", "Priority", "Description", "Labels", "Custom Field (Linked Req)"])

    priority_map = {
        "P0 - Critical": "Highest",
        "P1 - High":     "High",
        "P2 - Medium":   "Medium",
        "P3 - Low":      "Low",
    }

    for tc in cases:
        steps_html = "<br/>".join(
            f"<b>Step {s.step_number}:</b> {s.action} — <i>Expected: {s.expected}</i>"
            for s in tc.steps
        )
        description = (
            f"<b>Preconditions:</b> {'; '.join(tc.preconditions)}<br/>"
            f"<b>Steps:</b><br/>{steps_html}<br/>"
            f"<b>Expected Result:</b> {tc.expected_result}"
        )
        writer.writerow([
            f"[{tc.id}] {tc.title}",
            "Test",
            priority_map.get(tc.priority.value, "Medium"),
            description,
            " ".join(tc.tags),
            tc.linked_requirement,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=jira_import_{document_id[:8]}.csv"},
    )

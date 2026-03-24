import React, { useState } from "react";
import {
  reviewTestCase, exportTestCases,
  type TestCase, type GenerateResponse,
} from "../utils/api";
import toast from "react-hot-toast";

const PRIORITY_COLOR: Record<string, string> = {
  "P0 - Critical": "#ef4444",
  "P1 - High":     "#f97316",
  "P2 - Medium":   "#f59e0b",
  "P3 - Low":      "#10b981",
};

const TYPE_COLOR: Record<string, string> = {
  functional:    "#6366f1",
  edge:          "#f59e0b",
  negative:      "#ef4444",
  performance:   "#10b981",
  security:      "#8b5cf6",
  accessibility: "#06b6d4",
  regression:    "#f97316",
  uat:           "#84cc16",
};

const STATUS_STYLE: Record<string, React.CSSProperties> = {
  draft:    { background: "#f1f5f9", color: "#64748b" },
  approved: { background: "#dcfce7", color: "#166534" },
  rejected: { background: "#fee2e2", color: "#991b1b" },
};

interface Props {
  results: GenerateResponse;
  onBack: () => void;
}

export default function ResultsPage({ results, onBack }: Props) {
  const [cases,      setCases]     = useState<TestCase[]>(results.test_cases);
  const [expanded,   setExpanded]  = useState<string | null>(null);
  const [filter,     setFilter]    = useState<string>("all");

  const approve = async (tc: TestCase) => {
    await reviewTestCase(tc.id, "approve");
    setCases(prev => prev.map(c => c.id === tc.id ? { ...c, status: "approved" } : c));
    toast.success("Approved");
  };

  const reject = async (tc: TestCase) => {
    await reviewTestCase(tc.id, "reject");
    setCases(prev => prev.map(c => c.id === tc.id ? { ...c, status: "rejected" } : c));
    toast("Rejected", { icon: "✕" });
  };

  const filtered = filter === "all" ? cases : cases.filter(c => c.test_type === filter);

  const counts = cases.reduce((acc, c) => {
    acc[c.status] = (acc[c.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const uniqueTypes = [...new Set(cases.map(c => c.test_type))];

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem 1rem" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
        <button onClick={onBack} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--color-text-secondary)", fontSize: 14 }}>
          ← Back
        </button>
        <div>
          <h1 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 28, margin: 0 }}>
            {results.total} test cases generated
          </h1>
          <p style={{ color: "var(--color-text-secondary)", fontSize: 13, marginTop: 2 }}>
            {counts.approved || 0} approved · {counts.rejected || 0} rejected · {counts.draft || 0} pending review
          </p>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {(["csv", "json", "jira"] as const).map(fmt => (
            <button
              key={fmt}
              onClick={() => exportTestCases(results.document_id, fmt)}
              style={{
                padding: "7px 14px", borderRadius: 8,
                border: "1.5px solid var(--color-border-secondary)",
                background: "transparent",
                color: "var(--color-text-secondary)",
                cursor: "pointer", fontSize: 13, fontWeight: 500,
              }}
            >
              Export {fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Coverage gaps */}
      {results.coverage_gaps?.length > 0 && (
        <div style={{
          background: "#fffbeb", border: "1px solid #fde68a",
          borderRadius: 8, padding: "12px 16px", marginBottom: 20,
        }}>
          <p style={{ fontWeight: 500, color: "#92400e", marginBottom: 6, fontSize: 13 }}>
            ⚠ Coverage gaps detected
          </p>
          <ul style={{ margin: 0, paddingLeft: 18, color: "#92400e", fontSize: 12 }}>
            {results.coverage_gaps.map((g, i) => <li key={i}>{g}</li>)}
          </ul>
        </div>
      )}

      {/* Filter chips */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        {["all", ...uniqueTypes].map(t => (
          <button
            key={t}
            onClick={() => setFilter(t)}
            style={{
              padding: "5px 12px", borderRadius: 16, fontSize: 12, fontWeight: 500, cursor: "pointer",
              border: `1.5px solid ${filter === t ? (TYPE_COLOR[t] || "#6366f1") : "var(--color-border-secondary)"}`,
              background: filter === t ? `${TYPE_COLOR[t] || "#6366f1"}18` : "transparent",
              color: filter === t ? (TYPE_COLOR[t] || "#6366f1") : "var(--color-text-secondary)",
            }}
          >
            {t === "all" ? `All (${cases.length})` : `${t} (${cases.filter(c => c.test_type === t).length})`}
          </button>
        ))}
      </div>

      {/* Test case cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {filtered.map(tc => (
          <div
            key={tc.id}
            style={{
              border: "1px solid var(--color-border-tertiary)",
              borderRadius: 10,
              background: "var(--color-background-secondary)",
              overflow: "hidden",
            }}
          >
            {/* Card header */}
            <div
              style={{ padding: "12px 16px", cursor: "pointer", display: "flex", alignItems: "center", gap: 12 }}
              onClick={() => setExpanded(expanded === tc.id ? null : tc.id)}
            >
              <code style={{ fontSize: 11, color: "var(--color-text-tertiary)", minWidth: 90 }}>{tc.id}</code>

              <span style={{ flex: 1, fontWeight: 500, fontSize: 14, color: "var(--color-text-primary)" }}>
                {tc.title}
              </span>

              <span style={{
                padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 600,
                color: "#fff", background: TYPE_COLOR[tc.test_type] || "#6366f1",
              }}>
                {tc.test_type}
              </span>

              <span style={{
                padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 600,
                color: PRIORITY_COLOR[tc.priority], border: `1px solid ${PRIORITY_COLOR[tc.priority]}`,
              }}>
                {tc.priority.split(" - ")[0]}
              </span>

              <span style={{
                padding: "2px 10px", borderRadius: 12, fontSize: 11,
                ...STATUS_STYLE[tc.status],
              }}>
                {tc.status}
              </span>

              <span style={{ color: "var(--color-text-tertiary)", fontSize: 12 }}>
                {expanded === tc.id ? "▲" : "▼"}
              </span>
            </div>

            {/* Expanded detail */}
            {expanded === tc.id && (
              <div style={{ borderTop: "1px solid var(--color-border-tertiary)", padding: "16px" }}>
                {tc.preconditions.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4 }}>
                      PRECONDITIONS
                    </p>
                    <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: "var(--color-text-primary)" }}>
                      {tc.preconditions.map((p, i) => <li key={i}>{p}</li>)}
                    </ul>
                  </div>
                )}

                <div style={{ marginBottom: 12 }}>
                  <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4 }}>
                    STEPS
                  </p>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                    <thead>
                      <tr style={{ background: "var(--color-background-tertiary)" }}>
                        <th style={{ padding: "6px 10px", textAlign: "left", fontWeight: 500, color: "var(--color-text-secondary)", width: 40 }}>#</th>
                        <th style={{ padding: "6px 10px", textAlign: "left", fontWeight: 500, color: "var(--color-text-secondary)" }}>Action</th>
                        <th style={{ padding: "6px 10px", textAlign: "left", fontWeight: 500, color: "var(--color-text-secondary)" }}>Expected</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tc.steps.map(s => (
                        <tr key={s.step_number} style={{ borderTop: "1px solid var(--color-border-tertiary)" }}>
                          <td style={{ padding: "6px 10px", color: "var(--color-text-tertiary)" }}>{s.step_number}</td>
                          <td style={{ padding: "6px 10px", color: "var(--color-text-primary)" }}>{s.action}</td>
                          <td style={{ padding: "6px 10px", color: "var(--color-text-secondary)" }}>{s.expected}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div style={{ marginBottom: 12 }}>
                  <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", marginBottom: 4 }}>EXPECTED RESULT</p>
                  <p style={{ fontSize: 13, color: "var(--color-text-primary)" }}>{tc.expected_result}</p>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  {tc.linked_requirement && (
                    <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>
                      🔗 {tc.linked_requirement}
                    </span>
                  )}
                  <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
                    <button
                      onClick={() => approve(tc)}
                      disabled={tc.status === "approved"}
                      style={{
                        padding: "6px 14px", borderRadius: 7, border: "none", cursor: "pointer",
                        background: tc.status === "approved" ? "#dcfce7" : "#16a34a",
                        color: tc.status === "approved" ? "#166534" : "#fff",
                        fontSize: 13, fontWeight: 500,
                      }}
                    >
                      {tc.status === "approved" ? "✓ Approved" : "Approve"}
                    </button>
                    <button
                      onClick={() => reject(tc)}
                      disabled={tc.status === "rejected"}
                      style={{
                        padding: "6px 14px", borderRadius: 7,
                        border: "1.5px solid var(--color-border-secondary)",
                        background: "transparent", cursor: "pointer",
                        color: "var(--color-text-secondary)", fontSize: 13,
                      }}
                    >
                      Reject
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

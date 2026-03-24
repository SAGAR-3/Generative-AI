import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { uploadBRD, generateTestCases, type BRDDocument, type TestType, type GenerateResponse } from "../utils/api";
import toast from "react-hot-toast";

const ALL_TEST_TYPES: TestType[] = [
  "functional", "edge", "negative", "performance",
  "security", "accessibility", "regression", "uat",
];

const TYPE_META: Record<TestType, { label: string; color: string }> = {
  functional:    { label: "Functional",     color: "#6366f1" },
  edge:          { label: "Edge cases",     color: "#f59e0b" },
  negative:      { label: "Negative",       color: "#ef4444" },
  performance:   { label: "Performance",    color: "#10b981" },
  security:      { label: "Security",       color: "#8b5cf6" },
  accessibility: { label: "Accessibility",  color: "#06b6d4" },
  regression:    { label: "Regression",     color: "#f97316" },
  uat:           { label: "UAT",            color: "#84cc16" },
};

interface Props {
  onResults: (res: GenerateResponse) => void;
}

export default function UploadPage({ onResults }: Props) {
  const [doc,         setDoc]         = useState<BRDDocument | null>(null);
  const [selected,    setSelected]    = useState<Set<TestType>>(new Set(ALL_TEST_TYPES));
  const [maxCases,    setMaxCases]    = useState(50);
  const [uploading,   setUploading]   = useState(false);
  const [generating,  setGenerating]  = useState(false);

  const onDrop = useCallback(async (files: File[]) => {
    if (!files[0]) return;
    setUploading(true);
    try {
      const uploaded = await uploadBRD(files[0]);
      setDoc(uploaded);
      toast.success(`"${uploaded.filename}" uploaded — ${uploaded.chunks.length} chunks extracted.`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [],
              "text/plain": [], "text/markdown": [], "text/html": [] },
    multiple: false,
  });

  const toggleType = (t: TestType) => {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(t) ? next.delete(t) : next.add(t);
      return next;
    });
  };

  const handleGenerate = async () => {
    if (!doc) return;
    if (selected.size === 0) { toast.error("Select at least one test type."); return; }
    setGenerating(true);
    try {
      const results = await generateTestCases(doc.id, [...selected], maxCases);
      toast.success(`Generated ${results.total} test cases!`);
      onResults(results);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Generation failed. Check your API key.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: "2rem 1rem" }}>
      <h1 style={{ fontFamily: "'DM Serif Display', serif", fontSize: 36, marginBottom: 4, color: "var(--color-text-primary)" }}>
        BRD → Test Cases
      </h1>
      <p style={{ color: "var(--color-text-secondary)", marginBottom: 32 }}>
        Upload your Business Requirement Document and let the AI generate comprehensive test cases.
      </p>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        style={{
          border: `2px dashed ${isDragActive ? "#6366f1" : "var(--color-border-primary)"}`,
          borderRadius: 12,
          padding: "48px 24px",
          textAlign: "center",
          cursor: "pointer",
          background: isDragActive ? "rgba(99,102,241,0.05)" : "var(--color-background-secondary)",
          transition: "all 0.2s",
          marginBottom: 28,
        }}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <p style={{ color: "var(--color-text-secondary)" }}>Uploading &amp; parsing…</p>
        ) : doc ? (
          <div>
            <div style={{ fontSize: 28, marginBottom: 8 }}>✓</div>
            <p style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>{doc.filename}</p>
            <p style={{ color: "var(--color-text-secondary)", fontSize: 13 }}>
              {doc.chunks.length} chunks extracted · click to replace
            </p>
          </div>
        ) : (
          <>
            <p style={{ fontSize: 32, marginBottom: 8 }}>📄</p>
            <p style={{ fontWeight: 500, color: "var(--color-text-primary)" }}>
              Drop your BRD here, or click to browse
            </p>
            <p style={{ color: "var(--color-text-secondary)", fontSize: 13, marginTop: 4 }}>
              PDF · DOCX · TXT · Markdown · HTML
            </p>
          </>
        )}
      </div>

      {/* Test type selector */}
      <h3 style={{ fontSize: 14, fontWeight: 500, marginBottom: 12, color: "var(--color-text-primary)" }}>
        Test types to generate
      </h3>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 28 }}>
        {ALL_TEST_TYPES.map(t => {
          const meta  = TYPE_META[t];
          const active = selected.has(t);
          return (
            <button
              key={t}
              onClick={() => toggleType(t)}
              style={{
                padding: "6px 14px",
                borderRadius: 20,
                border: `1.5px solid ${active ? meta.color : "var(--color-border-secondary)"}`,
                background: active ? `${meta.color}18` : "transparent",
                color: active ? meta.color : "var(--color-text-secondary)",
                cursor: "pointer",
                fontSize: 13,
                fontWeight: active ? 500 : 400,
                transition: "all 0.15s",
              }}
            >
              {meta.label}
            </button>
          );
        })}
      </div>

      {/* Max cases slider */}
      <div style={{ marginBottom: 32 }}>
        <label style={{ fontSize: 14, fontWeight: 500, color: "var(--color-text-primary)" }}>
          Max test cases: <strong>{maxCases}</strong>
        </label>
        <input
          type="range" min={10} max={100} step={5}
          value={maxCases}
          onChange={e => setMaxCases(+e.target.value)}
          style={{ display: "block", width: "100%", marginTop: 8 }}
        />
      </div>

      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={!doc || generating}
        style={{
          width: "100%",
          padding: "14px 24px",
          borderRadius: 10,
          border: "none",
          background: !doc || generating ? "var(--color-border-secondary)" : "#6366f1",
          color: !doc || generating ? "var(--color-text-tertiary)" : "#fff",
          fontSize: 16,
          fontWeight: 500,
          cursor: !doc || generating ? "not-allowed" : "pointer",
          transition: "background 0.2s",
        }}
      >
        {generating ? "Generating test cases…" : "Generate test cases"}
      </button>
    </div>
  );
}

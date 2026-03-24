import React, { useState } from "react";
import { Toaster } from "react-hot-toast";
import UploadPage from "./pages/UploadPage";
import ResultsPage from "./pages/ResultsPage";
import { type GenerateResponse } from "./utils/api";

export default function App() {
  const [results, setResults] = useState<GenerateResponse | null>(null);

  return (
    <>
      <Toaster position="top-right" />
      <header style={{
        borderBottom: "1px solid var(--color-border-tertiary)",
        padding: "14px 24px",
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}>
        <span style={{ fontSize: 20 }}>🧪</span>
        <span style={{
          fontFamily: "'DM Serif Display', serif",
          fontSize: 18,
          color: "var(--color-text-primary)",
          fontWeight: 400,
        }}>
          GenAI Test Generator
        </span>
        <span style={{
          fontSize: 11, padding: "2px 8px", borderRadius: 12,
          background: "#6366f118", color: "#6366f1", fontWeight: 500,
        }}>
          BETA
        </span>
      </header>

      <main>
        {results
          ? <ResultsPage results={results} onBack={() => setResults(null)} />
          : <UploadPage onResults={setResults} />
        }
      </main>
    </>
  );
}

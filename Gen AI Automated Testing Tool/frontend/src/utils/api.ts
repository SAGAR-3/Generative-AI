/**
 * API Client — all backend interactions centralised here.
 */
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000/api",
  timeout: 120_000,   // 2 min — LLM calls can be slow
});

// ── Types ─────────────────────────────────────────────────────────────────────

export type TestType =
  | "functional" | "edge" | "negative"
  | "performance" | "security" | "accessibility"
  | "regression" | "uat";

export type Priority = "P0 - Critical" | "P1 - High" | "P2 - Medium" | "P3 - Low";

export interface TestStep {
  step_number: number;
  action:      string;
  expected:    string;
}

export interface TestCase {
  id:                 string;
  title:              string;
  test_type:          TestType;
  priority:           Priority;
  preconditions:      string[];
  steps:              TestStep[];
  expected_result:    string;
  linked_requirement: string;
  tags:               string[];
  status:             "draft" | "approved" | "rejected";
}

export interface BRDDocument {
  id:       string;
  filename: string;
  content:  string;
  chunks:   string[];
}

export interface GenerateResponse {
  document_id:   string;
  test_cases:    TestCase[];
  total:         number;
  coverage_gaps: string[];
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const uploadBRD = async (file: File): Promise<BRDDocument> => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<BRDDocument>("/upload/", form);
  return data;
};

export const generateTestCases = async (
  documentId: string,
  testTypes: TestType[],
  maxCases = 50,
): Promise<GenerateResponse> => {
  const { data } = await api.post<GenerateResponse>("/generate/", {
    document_id: documentId,
    test_types:  testTypes,
    max_cases:   maxCases,
    model:       "claude-sonnet-4-20250514",
  });
  return data;
};

export const reviewTestCase = async (
  testCaseId: string,
  action: "approve" | "reject" | "edit",
  editedCase?: TestCase,
  feedback?: string,
): Promise<void> => {
  await api.post("/generate/review", {
    test_case_id: testCaseId,
    action,
    edited_case:  editedCase,
    feedback,
  });
};

export const exportTestCases = (
  documentId: string,
  format: "csv" | "json" | "jira",
): void => {
  const base = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
  window.open(`${base}/export/${documentId}?format=${format}`, "_blank");
};

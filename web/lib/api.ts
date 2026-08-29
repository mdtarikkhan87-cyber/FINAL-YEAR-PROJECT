/**
 * Typed client for the FastAPI service in `api/`.
 *
 * Every prediction response carries a `model_quality` block and an
 * `input_warnings` list. Both are surfaced in the UI rather than hidden: these
 * models are trained on synthetic data and are weak in measured ways, and a
 * polished interface that quietly drops the caveats would misrepresent them.
 */

/**
 * Where the browser sends API calls.
 *
 * Unset (containers, production): "/api" — same origin, proxied server-side by
 * app/api/[...path]/route.ts to API_INTERNAL_URL, which is read at request
 * time and so can change without rebuilding the image.
 *
 * Set (npm run dev, via .env.local): that URL is called directly, which is
 * simpler locally and exercises the API's CORS configuration.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export type WageUnit = "Year" | "Hour" | "Month" | "Week" | "Bi-Weekly";

export interface CaseInput {
  filing_date: string;
  soc_code: string;
  worksite_state: string;
  prevailing_wage: number;
  prevailing_wage_unit: WageUnit;
  offered_wage: number;
  offered_wage_unit: WageUnit;
  job_title?: string | null;
  employer_num_employees?: number | null;
  employer_year_established?: number | null;
  naics_code?: string | null;
  employer_filing_volume?: number;
  skill_level?: string | null;
  job_education?: string | null;
  job_experience_months?: number | null;
  job_alt_occupation?: boolean;
  job_foreign_lang_req?: boolean;
  job_combo_occupation?: boolean;
  job_req_normal?: boolean;
  layoff_past_six_months?: boolean;
  professional_occupation?: boolean;
  refile?: boolean;
  class_of_admission?: string | null;
  citizenship?: string | null;
  has_attorney?: boolean;
}

export interface ModelQuality {
  trained_on: string;
  test_metric: string;
  baseline_metric: string;
  caveat: string;
}

export interface ProcessingTimeResponse {
  prediction_id: string;
  predicted_days: number;
  predicted_months: number;
  likely_range_days: [number, number];
  filing_date: string;
  estimated_decision_date: string;
  model_quality: ModelQuality;
  input_warnings: string[];
  explain_url: string;
}

export interface OutcomeProbability {
  outcome: string;
  probability: number;
}

export interface OutcomeResponse {
  prediction_id: string;
  predicted_outcome: string;
  confidence: number;
  probabilities: OutcomeProbability[];
  model_quality: ModelQuality;
  input_warnings: string[];
  explain_url: string;
}

export interface ShapContribution {
  feature: string;
  value: string;
  shap_value: number;
  direction: "increases" | "decreases";
}

export interface ExplanationResponse {
  prediction_id: string;
  task: "regression" | "classification";
  created_at: string;
  prediction: number;
  explained_class: string | null;
  base_value: number;
  units: string;
  contributions: ShapContribution[];
  reconstruction_check: number;
  note: string;
}

export interface FieldOption {
  value: string;
  label: string;
}

export interface OptionsResponse {
  soc_codes: FieldOption[];
  worksite_states: FieldOption[];
  skill_levels: FieldOption[];
  education_levels: FieldOption[];
  classes_of_admission: FieldOption[];
  citizenships: FieldOption[];
  wage_units: FieldOption[];
  outcome_classes: string[];
}

export interface HealthResponse {
  status: string;
  models_loaded: string[];
  stored_predictions: number;
  store_capacity: number;
  warning: string;
}

/** A validation failure from FastAPI, unpacked into readable lines. */
export class ApiError extends Error {
  readonly status: number;
  readonly details: string[];

  constructor(message: string, status: number, details: string[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

function unpackDetail(payload: unknown): string[] {
  if (!payload || typeof payload !== "object") return [];
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return [detail];
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      if (typeof d === "string") return d;
      const loc = Array.isArray(d?.loc)
        ? d.loc.filter((p: unknown) => p !== "body").join(".")
        : "";
      const msg = String(d?.msg ?? "invalid value").replace(/^Value error, /, "");
      return loc ? `${loc}: ${msg}` : msg;
    });
  }
  return [];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      `Cannot reach the prediction API at ${API_BASE}.`,
      0,
      ["Start it with:  uvicorn api.main:app --port 8000"],
    );
  }

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const details = unpackDetail(body);
    throw new ApiError(
      response.status === 422
        ? "Some of the case details need fixing."
        : `Request failed (HTTP ${response.status}).`,
      response.status,
      details,
    );
  }
  return body as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  options: () => request<OptionsResponse>("/meta/options"),
  processingTime: (payload: CaseInput) =>
    request<ProcessingTimeResponse>("/predict/processing-time", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  outcome: (payload: CaseInput) =>
    request<OutcomeResponse>("/predict/outcome", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  explain: (predictionId: string) =>
    request<ExplanationResponse>(`/explain/${predictionId}`),
};

/** Human-readable labels for the model's raw feature names. */
export const FEATURE_LABELS: Record<string, string> = {
  employer_size_bucket: "Employer size band",
  employer_num_employees: "Employer headcount",
  employer_filing_volume: "Employer filing volume",
  employer_age_years: "Employer age",
  naics_sector: "Industry sector",
  soc_code: "Occupation code",
  soc_major_group: "Occupation group",
  skill_level: "Wage skill level",
  worksite_state: "Worksite state",
  worksite_region: "Worksite region",
  prevailing_wage_annual: "Prevailing wage (annual)",
  offered_wage_annual: "Offered wage (annual)",
  wage_ratio: "Offered ÷ prevailing wage",
  wage_premium: "Wage premium",
  offered_below_prevailing: "Offered below prevailing",
  job_education: "Education required",
  job_experience_months: "Experience required",
  requires_experience: "Experience required?",
  job_alt_occupation: "Alternate occupation accepted",
  job_foreign_lang_req: "Foreign language required",
  job_combo_occupation: "Combination occupation",
  job_req_normal: "Requirements are normal",
  layoff_past_six_months: "Layoffs in past 6 months",
  professional_occupation: "Professional occupation",
  refile: "Refiled case",
  filing_fiscal_year: "Filing fiscal year",
  filing_fiscal_quarter: "Filing quarter",
  filing_month: "Filing month",
  has_attorney: "Represented by attorney",
  class_of_admission: "Current visa status",
  citizenship: "Country of citizenship",
};

export function featureLabel(name: string): string {
  return FEATURE_LABELS[name] ?? name.replace(/_/g, " ");
}

export const OUTCOME_LABELS: Record<string, string> = {
  certified: "Certified",
  denied: "Denied",
  withdrawn: "Withdrawn",
  certified_expired: "Certified-Expired",
};

export const OUTCOME_COLORS: Record<string, string> = {
  certified: "#15803d",
  denied: "#b91c1c",
  withdrawn: "#b45309",
  certified_expired: "#4338ca",
};

"use client";

import { useEffect, useMemo, useState } from "react";

import {
  api,
  ApiError,
  type CaseInput,
  type FieldOption,
  type OptionsResponse,
  type WageUnit,
} from "@/lib/api";

const FALLBACK_UNITS: FieldOption[] = [
  { value: "Year", label: "Year" },
  { value: "Hour", label: "Hour" },
  { value: "Month", label: "Month" },
  { value: "Week", label: "Week" },
  { value: "Bi-Weekly", label: "Bi-Weekly" },
];

const PRESETS: Record<string, Partial<FormState>> = {
  "Software engineer, large tech": {
    soc_code: "15-1252",
    worksite_state: "CA",
    prevailing_wage: "145000",
    offered_wage: "162000",
    employer_num_employees: "4200",
    employer_year_established: "2009",
    naics_code: "541511",
    skill_level: "LEVEL III",
    job_education: "MASTER'S",
    job_experience_months: "36",
    class_of_admission: "H-1B",
    citizenship: "INDIA",
  },
  "Nurse, small employer": {
    soc_code: "29-1141",
    worksite_state: "TX",
    prevailing_wage: "36.5",
    prevailing_wage_unit: "Hour",
    offered_wage: "39",
    offered_wage_unit: "Hour",
    employer_num_employees: "42",
    employer_year_established: "1998",
    naics_code: "622110",
    skill_level: "LEVEL II",
    job_education: "BACHELOR'S",
    job_experience_months: "24",
    class_of_admission: "H-1B",
    citizenship: "PHILIPPINES",
  },
  "Below prevailing wage (risk case)": {
    soc_code: "15-1252",
    worksite_state: "NJ",
    prevailing_wage: "150000",
    offered_wage: "138000",
    employer_num_employees: "25",
    employer_year_established: "2019",
    naics_code: "541512",
    skill_level: "LEVEL II",
    job_education: "BACHELOR'S",
    job_experience_months: "24",
    class_of_admission: "H-1B",
    citizenship: "INDIA",
    layoff_past_six_months: true,
    job_foreign_lang_req: true,
  },
};

export interface FormState {
  filing_date: string;
  soc_code: string;
  worksite_state: string;
  prevailing_wage: string;
  prevailing_wage_unit: WageUnit;
  offered_wage: string;
  offered_wage_unit: WageUnit;
  employer_num_employees: string;
  employer_year_established: string;
  naics_code: string;
  skill_level: string;
  job_education: string;
  job_experience_months: string;
  class_of_admission: string;
  citizenship: string;
  job_alt_occupation: boolean;
  job_foreign_lang_req: boolean;
  layoff_past_six_months: boolean;
  refile: boolean;
  has_attorney: boolean;
}

const INITIAL: FormState = {
  filing_date: "2024-03-15",
  soc_code: "15-1252",
  worksite_state: "CA",
  prevailing_wage: "145000",
  prevailing_wage_unit: "Year",
  offered_wage: "162000",
  offered_wage_unit: "Year",
  employer_num_employees: "4200",
  employer_year_established: "2009",
  naics_code: "541511",
  skill_level: "LEVEL III",
  job_education: "MASTER'S",
  job_experience_months: "36",
  class_of_admission: "H-1B",
  citizenship: "INDIA",
  job_alt_occupation: false,
  job_foreign_lang_req: false,
  layoff_past_six_months: false,
  refile: false,
  has_attorney: true,
};

function num(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export function toPayload(form: FormState): CaseInput {
  return {
    filing_date: form.filing_date,
    soc_code: form.soc_code,
    worksite_state: form.worksite_state,
    prevailing_wage: num(form.prevailing_wage) ?? 0,
    prevailing_wage_unit: form.prevailing_wage_unit,
    offered_wage: num(form.offered_wage) ?? 0,
    offered_wage_unit: form.offered_wage_unit,
    employer_num_employees: num(form.employer_num_employees),
    employer_year_established: num(form.employer_year_established),
    naics_code: form.naics_code.trim() || null,
    skill_level: form.skill_level ? titleCase(form.skill_level) : null,
    job_education: form.job_education ? titleCase(form.job_education) : null,
    job_experience_months: num(form.job_experience_months),
    class_of_admission: form.class_of_admission || null,
    citizenship: form.citizenship || null,
    job_alt_occupation: form.job_alt_occupation,
    job_foreign_lang_req: form.job_foreign_lang_req,
    layoff_past_six_months: form.layoff_past_six_months,
    refile: form.refile,
    has_attorney: form.has_attorney,
  };
}

/** The API's enums are title-cased ("Level III"); the model's levels are upper. */
function titleCase(value: string): string {
  return value
    .toLowerCase()
    .split(" ")
    .map((w) => (w === "i" || w === "ii" || w === "iii" || w === "iv"
      ? w.toUpperCase()
      : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}

function Select({
  label,
  value,
  onChange,
  options,
  hint,
  allowEmpty,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: FieldOption[];
  hint?: string;
  allowEmpty?: boolean;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <select
        className="input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {allowEmpty && <option value="">Not specified</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      {hint && <span className="mt-1 block text-xs text-ink-500">{hint}</span>}
    </label>
  );
}

function Text({
  label,
  value,
  onChange,
  type = "text",
  placeholder,
  hint,
  inputMode,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
  hint?: string;
  inputMode?: "numeric" | "decimal";
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <input
        className="input"
        type={type}
        value={value}
        placeholder={placeholder}
        inputMode={inputMode}
        onChange={(e) => onChange(e.target.value)}
      />
      {hint && <span className="mt-1 block text-xs text-ink-500">{hint}</span>}
    </label>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-ink-200 px-3 py-2.5 transition hover:border-ink-300 hover:bg-ink-50">
      <input
        type="checkbox"
        className="h-4 w-4 rounded border-ink-300 text-accent-600 focus:ring-accent-500/30"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="text-[13px] leading-snug text-ink-700">{label}</span>
    </label>
  );
}

export default function CaseForm({
  onSubmit,
  loading,
  error,
}: {
  onSubmit: (payload: CaseInput) => void;
  loading: boolean;
  error: ApiError | null;
}) {
  const [form, setForm] = useState<FormState>(INITIAL);
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [optionsError, setOptionsError] = useState(false);

  useEffect(() => {
    api
      .options()
      .then(setOptions)
      .catch(() => setOptionsError(true));
  }, []);

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const units = options?.wage_units ?? FALLBACK_UNITS;

  const annualised = useMemo(() => {
    const factor: Record<string, number> = {
      Year: 1, Hour: 2080, Month: 12, Week: 52, "Bi-Weekly": 26,
    };
    const pw = (num(form.prevailing_wage) ?? 0) * (factor[form.prevailing_wage_unit] ?? 1);
    const ow = (num(form.offered_wage) ?? 0) * (factor[form.offered_wage_unit] ?? 1);
    if (!pw || !ow) return null;
    return { pw, ow, ratio: ow / pw };
  }, [form.prevailing_wage, form.prevailing_wage_unit, form.offered_wage, form.offered_wage_unit]);

  return (
    <form
      className="card p-6"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(toPayload(form));
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-ink-950">Case details</h2>
        <div className="flex flex-wrap gap-1.5">
          {Object.keys(PRESETS).map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => setForm((f) => ({ ...INITIAL, ...f, ...PRESETS[name] }))}
              className="rounded-md border border-ink-200 px-2.5 py-1 text-[11px] font-medium text-ink-600 transition hover:border-accent-400 hover:text-accent-700"
            >
              {name}
            </button>
          ))}
        </div>
      </div>

      {optionsError && (
        <p className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          Could not load dropdown options from the API — using a reduced set. Some
          values may not be recognised by the model.
        </p>
      )}

      <div className="mt-5 space-y-5">
        <fieldset>
          <legend className="mb-3 text-xs font-semibold uppercase tracking-wide text-ink-500">
            Filing
          </legend>
          <div className="grid gap-4 sm:grid-cols-2">
            <Text
              label="Filing date"
              type="date"
              value={form.filing_date}
              onChange={(v) => set("filing_date", v)}
            />
            <Select
              label="Worksite state"
              value={form.worksite_state}
              onChange={(v) => set("worksite_state", v)}
              options={options?.worksite_states ?? [{ value: "CA", label: "CA" }]}
            />
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-3 text-xs font-semibold uppercase tracking-wide text-ink-500">
            Occupation
          </legend>
          <div className="grid gap-4">
            <Select
              label="SOC occupation code"
              value={form.soc_code}
              onChange={(v) => set("soc_code", v)}
              options={options?.soc_codes ?? [{ value: "15-1252", label: "15-1252" }]}
            />
            <div className="grid gap-4 sm:grid-cols-2">
              <Select
                label="Wage skill level"
                value={form.skill_level}
                onChange={(v) => set("skill_level", v)}
                options={options?.skill_levels ?? []}
                allowEmpty
              />
              <Select
                label="Education required"
                value={form.job_education}
                onChange={(v) => set("job_education", v)}
                options={options?.education_levels ?? []}
                allowEmpty
              />
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-3 text-xs font-semibold uppercase tracking-wide text-ink-500">
            Wages
          </legend>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="grid grid-cols-[1fr_110px] gap-2">
              <Text
                label="Prevailing wage"
                value={form.prevailing_wage}
                onChange={(v) => set("prevailing_wage", v)}
                inputMode="decimal"
              />
              <Select
                label="Unit"
                value={form.prevailing_wage_unit}
                onChange={(v) => set("prevailing_wage_unit", v as WageUnit)}
                options={units}
              />
            </div>
            <div className="grid grid-cols-[1fr_110px] gap-2">
              <Text
                label="Offered wage"
                value={form.offered_wage}
                onChange={(v) => set("offered_wage", v)}
                inputMode="decimal"
              />
              <Select
                label="Unit"
                value={form.offered_wage_unit}
                onChange={(v) => set("offered_wage_unit", v as WageUnit)}
                options={units}
              />
            </div>
          </div>
          {annualised && (
            <p
              className={`mt-2.5 text-xs ${
                annualised.ratio < 1 ? "text-red-700" : "text-ink-500"
              }`}
            >
              Annualised: prevailing ${Math.round(annualised.pw).toLocaleString()} ·
              offered ${Math.round(annualised.ow).toLocaleString()} · ratio{" "}
              {annualised.ratio.toFixed(3)}
              {annualised.ratio < 1 && " — below prevailing wage, a strong denial signal"}
            </p>
          )}
        </fieldset>

        <fieldset>
          <legend className="mb-3 text-xs font-semibold uppercase tracking-wide text-ink-500">
            Employer
          </legend>
          <div className="grid gap-4 sm:grid-cols-3">
            <Text
              label="Number of employees"
              value={form.employer_num_employees}
              onChange={(v) => set("employer_num_employees", v)}
              inputMode="numeric"
            />
            <Text
              label="Year established"
              value={form.employer_year_established}
              onChange={(v) => set("employer_year_established", v)}
              inputMode="numeric"
            />
            <Text
              label="NAICS code"
              value={form.naics_code}
              onChange={(v) => set("naics_code", v)}
              placeholder="541511"
              inputMode="numeric"
            />
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-3 text-xs font-semibold uppercase tracking-wide text-ink-500">
            Foreign worker
          </legend>
          <div className="grid gap-4 sm:grid-cols-3">
            <Select
              label="Country of citizenship"
              value={form.citizenship}
              onChange={(v) => set("citizenship", v)}
              options={options?.citizenships ?? []}
              allowEmpty
            />
            <Select
              label="Current visa status"
              value={form.class_of_admission}
              onChange={(v) => set("class_of_admission", v)}
              options={options?.classes_of_admission ?? []}
              allowEmpty
            />
            <Text
              label="Experience required (months)"
              value={form.job_experience_months}
              onChange={(v) => set("job_experience_months", v)}
              inputMode="numeric"
            />
          </div>
        </fieldset>

        <fieldset>
          <legend className="mb-3 text-xs font-semibold uppercase tracking-wide text-ink-500">
            Case characteristics
          </legend>
          <div className="grid gap-2 sm:grid-cols-2">
            <Toggle
              label="Alternate occupation accepted"
              checked={form.job_alt_occupation}
              onChange={(v) => set("job_alt_occupation", v)}
            />
            <Toggle
              label="Foreign language required"
              checked={form.job_foreign_lang_req}
              onChange={(v) => set("job_foreign_lang_req", v)}
            />
            <Toggle
              label="Layoffs in the past 6 months"
              checked={form.layoff_past_six_months}
              onChange={(v) => set("layoff_past_six_months", v)}
            />
            <Toggle
              label="Refiled case"
              checked={form.refile}
              onChange={(v) => set("refile", v)}
            />
            <Toggle
              label="Represented by an attorney"
              checked={form.has_attorney}
              onChange={(v) => set("has_attorney", v)}
            />
          </div>
        </fieldset>
      </div>

      {error && (
        <div className="mt-5 rounded-lg border border-red-300 bg-red-50 px-4 py-3">
          <p className="text-sm font-semibold text-red-900">{error.message}</p>
          {error.details.length > 0 && (
            <ul className="mt-1.5 list-inside list-disc space-y-0.5 text-xs text-red-800">
              {error.details.map((d) => (
                <li key={d}>{d}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? "Predicting…" : "Predict this case"}
          {!loading && <span aria-hidden>→</span>}
        </button>
        <button
          type="button"
          className="btn-ghost"
          onClick={() => setForm(INITIAL)}
          disabled={loading}
        >
          Reset
        </button>
      </div>
    </form>
  );
}

"use client";

import { useState } from "react";

import CaseForm from "@/components/CaseForm";
import OutcomeChart from "@/components/OutcomeChart";
import ShapChart from "@/components/ShapChart";
import SyntheticNotice from "@/components/SyntheticNotice";
import {
  api,
  ApiError,
  OUTCOME_LABELS,
  type CaseInput,
  type ExplanationResponse,
  type ModelQuality,
  type OutcomeResponse,
  type ProcessingTimeResponse,
} from "@/lib/api";

interface Results {
  time: ProcessingTimeResponse;
  outcome: OutcomeResponse;
  timeExplanation: ExplanationResponse;
  outcomeExplanation: ExplanationResponse;
}

function QualityNote({ quality, label }: { quality: ModelQuality; label: string }) {
  return (
    <details className="mt-4 rounded-lg border border-ink-200 bg-ink-50 px-4 py-3">
      <summary className="cursor-pointer text-xs font-semibold text-ink-700">
        How good is this {label} model?
      </summary>
      <dl className="mt-3 space-y-2 text-xs leading-relaxed text-ink-600">
        <div>
          <dt className="font-medium text-ink-800">Trained on</dt>
          <dd>{quality.trained_on}</dd>
        </div>
        <div>
          <dt className="font-medium text-ink-800">Test performance</dt>
          <dd className="font-mono">{quality.test_metric}</dd>
        </div>
        <div>
          <dt className="font-medium text-ink-800">Baseline</dt>
          <dd className="font-mono">{quality.baseline_metric}</dd>
        </div>
        <div>
          <dt className="font-medium text-ink-800">Caveat</dt>
          <dd>{quality.caveat}</dd>
        </div>
      </dl>
    </details>
  );
}

function Warnings({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3">
      <p className="text-xs font-semibold text-amber-900">
        Inputs affecting this prediction
      </p>
      <ul className="mt-1.5 list-inside list-disc space-y-1 text-xs leading-relaxed text-amber-900">
        {items.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="card flex h-full min-h-[420px] flex-col items-center justify-center p-10 text-center">
      <span
        aria-hidden
        className="grid h-12 w-12 place-items-center rounded-xl bg-accent-50 text-xl text-accent-600"
      >
        ◷
      </span>
      <h3 className="mt-4 text-base font-semibold text-ink-950">
        No prediction yet
      </h3>
      <p className="mt-2 max-w-sm text-sm leading-relaxed text-ink-600">
        Fill in the case details and submit. You will get a processing-time
        estimate, outcome probabilities, and the SHAP attribution showing which
        factors drove each result.
      </p>
    </div>
  );
}

export default function PredictPage() {
  const [results, setResults] = useState<Results | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [shapTab, setShapTab] = useState<"time" | "outcome">("time");

  async function handleSubmit(payload: CaseInput) {
    setLoading(true);
    setError(null);
    try {
      const [time, outcome] = await Promise.all([
        api.processingTime(payload),
        api.outcome(payload),
      ]);
      const [timeExplanation, outcomeExplanation] = await Promise.all([
        api.explain(time.prediction_id),
        api.explain(outcome.prediction_id),
      ]);
      setResults({ time, outcome, timeExplanation, outcomeExplanation });
    } catch (e) {
      setError(
        e instanceof ApiError
          ? e
          : new ApiError("Something went wrong.", 0, [String(e)]),
      );
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  const explanation =
    shapTab === "time" ? results?.timeExplanation : results?.outcomeExplanation;

  return (
    <div className="bg-ink-50 pb-20">
      <div className="border-b border-ink-200 bg-white">
        <div className="container-page py-10">
          <h1 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
            Predict a case
          </h1>
          <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ink-600">
            Enter the details of a PERM filing. The models return a timeline
            estimate, outcome probabilities, and a SHAP attribution for each
            prediction.
          </p>
          <div className="mt-5 max-w-3xl">
            <SyntheticNotice compact />
          </div>
        </div>
      </div>

      <div className="container-page grid gap-6 pt-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:grid-cols-[480px_minmax(0,1fr)]">
        <div>
          <CaseForm onSubmit={handleSubmit} loading={loading} error={error} />
        </div>

        <div className="space-y-6">
          {!results && !loading && <EmptyState />}

          {loading && (
            <div className="card flex min-h-[420px] items-center justify-center p-10">
              <div className="text-center">
                <div
                  className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-ink-200 border-t-accent-600"
                  aria-hidden
                />
                <p className="mt-4 text-sm text-ink-600">
                  Running both models and computing SHAP…
                </p>
              </div>
            </div>
          )}

          {results && !loading && (
            <>
              {/* Processing time */}
              <section className="card animate-fade-up p-6">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-500">
                      Predicted processing time
                    </h2>
                    <p className="mt-3 text-4xl font-semibold tracking-tight text-ink-950">
                      {Math.round(results.time.predicted_days)}
                      <span className="ml-2 text-lg font-medium text-ink-500">
                        days
                      </span>
                    </p>
                    <p className="mt-1 text-sm text-ink-600">
                      about {results.time.predicted_months} months
                    </p>
                  </div>
                  <div className="shrink-0 rounded-lg bg-ink-50 px-4 py-3 text-right">
                    <p className="text-xs font-medium text-ink-500">
                      Estimated decision
                    </p>
                    <p className="mt-1 font-mono text-sm font-semibold text-ink-900">
                      {results.time.estimated_decision_date}
                    </p>
                  </div>
                </div>

                <div className="mt-5 rounded-lg border border-ink-200 px-4 py-3">
                  <p className="text-xs font-medium text-ink-500">
                    Likely range (± the model&apos;s average error)
                  </p>
                  <p className="mt-1 text-sm font-semibold text-ink-900">
                    {Math.round(results.time.likely_range_days[0])} –{" "}
                    {Math.round(results.time.likely_range_days[1])} days
                  </p>
                  <p className="mt-1.5 text-xs leading-relaxed text-ink-500">
                    An empirical error band from test-set MAE, not a calibrated
                    prediction interval.
                  </p>
                </div>

                <Warnings items={results.time.input_warnings} />
                <QualityNote
                  quality={results.time.model_quality}
                  label="processing-time"
                />
              </section>

              {/* Outcome */}
              <section className="card animate-fade-up p-6">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-500">
                    Predicted outcome
                  </h2>
                  <span className="pill bg-ink-100 text-ink-700">
                    most likely:{" "}
                    {OUTCOME_LABELS[results.outcome.predicted_outcome] ??
                      results.outcome.predicted_outcome}{" "}
                    · {(results.outcome.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-4">
                  <OutcomeChart result={results.outcome} />
                </div>
                <Warnings items={results.outcome.input_warnings} />
                <QualityNote
                  quality={results.outcome.model_quality}
                  label="outcome"
                />
              </section>

              {/* SHAP */}
              <section className="card animate-fade-up p-6">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-500">
                    Why this prediction
                  </h2>
                  <div
                    className="inline-flex rounded-lg border border-ink-200 p-0.5"
                    role="tablist"
                  >
                    {(
                      [
                        ["time", "Processing time"],
                        ["outcome", "Outcome"],
                      ] as const
                    ).map(([key, label]) => (
                      <button
                        key={key}
                        role="tab"
                        aria-selected={shapTab === key}
                        onClick={() => setShapTab(key)}
                        className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                          shapTab === key
                            ? "bg-accent-600 text-white"
                            : "text-ink-600 hover:bg-ink-50"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="mt-4">
                  {explanation && <ShapChart explanation={explanation} />}
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

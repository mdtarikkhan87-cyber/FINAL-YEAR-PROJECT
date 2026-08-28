import Link from "next/link";

const PILLARS = [
  {
    icon: "◎",
    title: "Explainable",
    body: "Every prediction returns exact TreeSHAP attributions — which factors moved the estimate, by how much, and in which direction. Additivity is asserted, not assumed.",
  },
  {
    icon: "⚖",
    title: "Fairness-audited",
    body: "Group metrics across employer size, occupation category, and worksite region using fairlearn — with an explicit guard against a constant classifier scoring as perfectly fair.",
  },
  {
    icon: "◷",
    title: "Temporally validated",
    body: "Five expanding-window folds — train on years up to N, test on N+1 — instead of one fixed split, so performance reads as a series rather than a lucky point estimate.",
  },
];

const STAGES = [
  { name: "DOL disclosure files", detail: "Quarterly OFLC extracts" },
  { name: "Schema harmonisation", detail: "SOC crosswalk, wage units, dates" },
  { name: "Temporal split", detail: "Decision fiscal year" },
  { name: "XGBoost models", detail: "Regression + classification" },
  { name: "SHAP + fairlearn", detail: "Explanation and audit" },
];

const STATS = [
  { k: "Two targets", v: "Processing time and case outcome" },
  { k: "50,000 cases", v: "FY2018–FY2024 synthetic sample" },
  { k: "31 features", v: "Pre-decision information only" },
  { k: "5 folds", v: "Expanding-window validation" },
];

export default function HomePage() {
  return (
    <>
      <section className="relative overflow-hidden bg-ink-950">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            background:
              "radial-gradient(60% 55% at 18% 8%, rgba(52,94,248,0.32) 0%, transparent 62%), radial-gradient(50% 50% at 88% 22%, rgba(28,40,140,0.42) 0%, transparent 60%)",
          }}
        />
        <div className="container-page relative py-20 sm:py-28">
          <div className="max-w-3xl animate-fade-up">
            <span className="pill border border-white/15 bg-white/10 text-accent-100">
              Final-year research project
            </span>
            <h1 className="mt-6 text-4xl font-semibold leading-[1.1] tracking-tight text-white sm:text-5xl lg:text-6xl">
              How long will your PERM case take —
              <span className="text-accent-300"> and how will it end?</span>
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-relaxed text-ink-300">
              Employers, applicants, and immigration attorneys have no reliable way
              to estimate PERM labour-certification timelines or outcomes. This
              project predicts both — and refuses to stop at a single accuracy
              score.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link href="/predict" className="btn-primary px-6 py-3">
                Try a prediction <span aria-hidden>→</span>
              </Link>
              <Link
                href="/about"
                className="inline-flex items-center gap-2 rounded-lg border border-white/20 px-5 py-3 text-sm font-medium text-white transition hover:bg-white/10"
              >
                Read the methodology
              </Link>
            </div>
            <p className="mt-8 max-w-xl text-xs leading-relaxed text-ink-400">
              Trained on synthetic data generated for this project. Demonstrates the
              pipeline end to end; not valid for real cases.
            </p>
          </div>
        </div>
      </section>

      <section className="border-b border-ink-200 bg-white">
        <div className="container-page grid gap-10 py-16 lg:grid-cols-[1.1fr_1fr] lg:gap-16">
          <div>
            <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
              The problem
            </h2>
            <div className="mt-5 space-y-4 text-[15px] leading-relaxed text-ink-600">
              <p>
                PERM is the first major step toward an employment-based green card.
                A case can take a few months or well over a year, and it ends as
                certified, denied, withdrawn, or certified-expired. Public
                processing-time pages report coarse averages, not case-level
                estimates.
              </p>
              <p>
                The cost is concrete: employers cannot plan headcount, applicants
                cannot plan their lives, and attorneys cannot triage which cases
                need intervention.
              </p>
            </div>
          </div>
          <dl className="grid grid-cols-2 gap-4 self-start">
            {STATS.map((s) => (
              <div key={s.k} className="card p-5">
                <dt className="text-sm font-semibold text-ink-950">{s.k}</dt>
                <dd className="mt-1 text-[13px] leading-relaxed text-ink-600">
                  {s.v}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <section className="bg-ink-50">
        <div className="container-page py-16">
          <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
            Beyond a single accuracy score
          </h2>
          <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-ink-600">
            One number is not a defensible answer for a system that touches
            immigration decisions. Three additional pillars carry the work.
          </p>
          <div className="mt-9 grid gap-5 md:grid-cols-3">
            {PILLARS.map((p) => (
              <article key={p.title} className="card p-6">
                <span
                  aria-hidden
                  className="grid h-10 w-10 place-items-center rounded-lg bg-accent-50 text-lg text-accent-700"
                >
                  {p.icon}
                </span>
                <h3 className="mt-4 text-base font-semibold text-ink-950">
                  {p.title}
                </h3>
                <p className="mt-2 text-[14px] leading-relaxed text-ink-600">
                  {p.body}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-white">
        <div className="container-page py-16">
          <h2 className="text-2xl font-semibold tracking-tight text-ink-950 sm:text-3xl">
            The pipeline
          </h2>
          <ol className="mt-8 grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
            {STAGES.map((stage, i) => (
              <li key={stage.name} className="card p-5">
                <span className="font-mono text-xs font-semibold text-accent-600">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <p className="mt-2 text-sm font-semibold leading-snug text-ink-950">
                  {stage.name}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-ink-500">
                  {stage.detail}
                </p>
              </li>
            ))}
          </ol>
          <div className="mt-10 rounded-xl border border-ink-200 bg-ink-50 p-6">
            <p className="text-sm leading-relaxed text-ink-700">
              <strong className="font-semibold text-ink-950">
                Why temporal validation matters here.
              </strong>{" "}
              PERM backlogs shift year to year — median processing time in the
              sample runs from 164 days in FY2018 to 417 in FY2023. A random
              train/test split lets the model see the future and reports a score it
              will never reproduce. Every split in this project is chronological.
            </p>
          </div>
        </div>
      </section>

      <section className="bg-ink-950">
        <div className="container-page flex flex-col items-start gap-6 py-14 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">
              See it work on a case
            </h2>
            <p className="mt-2 max-w-xl text-sm leading-relaxed text-ink-300">
              Enter the details of a hypothetical filing and get a timeline
              estimate, outcome probabilities, and the SHAP attribution behind
              them.
            </p>
          </div>
          <Link href="/predict" className="btn-primary shrink-0 px-6 py-3">
            Open the predictor <span aria-hidden>→</span>
          </Link>
        </div>
      </section>
    </>
  );
}

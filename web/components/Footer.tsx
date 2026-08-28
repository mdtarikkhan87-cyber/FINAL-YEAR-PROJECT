export default function Footer() {
  return (
    <footer className="mt-24 border-t border-ink-200 bg-ink-50">
      <div className="container-page py-10">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-md">
            <p className="text-sm font-semibold text-ink-900">
              PERM Processing Time &amp; Outcome
            </p>
            <p className="mt-2 text-sm leading-relaxed text-ink-600">
              An explainable, fairness-audited machine learning system for U.S.
              PERM labour certification. Final-year research project.
            </p>
          </div>
          <div className="text-sm text-ink-600">
            <p className="font-medium text-ink-800">Data source</p>
            <a
              className="mt-1 block text-accent-700 underline-offset-2 hover:underline"
              href="https://www.dol.gov/agencies/eta/foreign-labor/performance"
              target="_blank"
              rel="noreferrer noopener"
            >
              DOL OFLC Performance Data
            </a>
            <p className="mt-3 max-w-xs text-xs leading-relaxed text-ink-500">
              Models here are fitted to synthetic data. Not legal advice and not a
              prediction about any real case.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}

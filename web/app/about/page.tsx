import Link from "next/link";

import SyntheticNotice from "@/components/SyntheticNotice";

const FOLDS = [
  { fold: 1, train: "2018–2019", test: 2020, rmse: 145.9, base: 167.6, impr: "+12.9%" },
  { fold: 2, train: "2018–2020", test: 2021, rmse: 162.0, base: 191.6, impr: "+15.5%" },
  { fold: 3, train: "2018–2021", test: 2022, rmse: 211.7, base: 250.4, impr: "+15.5%" },
  { fold: 4, train: "2018–2022", test: 2023, rmse: 229.3, base: 272.6, impr: "+15.9%" },
  { fold: 5, train: "2018–2023", test: 2024, rmse: 159.8, base: 182.5, impr: "+12.5%" },
];

const FAIRNESS = [
  { attr: "Employer size", dp: "0.023", eo: "0.042", ratio: "0.977" },
  { attr: "Occupation (SOC major group)", dp: "0.052", eo: "0.052", ratio: "0.948" },
  { attr: "Worksite region", dp: "0.037", eo: "0.039", ratio: "0.963" },
];

function Section({
  id,
  eyebrow,
  title,
  children,
}: {
  id: string;
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-24 border-t border-ink-200 py-12 first:border-t-0">
      <p className="text-xs font-semibold uppercase tracking-wide text-accent-600">
        {eyebrow}
      </p>
      <h2 className="mt-2 text-2xl font-semibold tracking-tight text-ink-950">
        {title}
      </h2>
      <div className="mt-5 space-y-4 text-[15px] leading-relaxed text-ink-600">
        {children}
      </div>
    </section>
  );
}

export default function AboutPage() {
  return (
    <div className="bg-white">
      <div className="border-b border-ink-200 bg-ink-50">
        <div className="container-page py-14">
          <h1 className="max-w-3xl text-3xl font-semibold tracking-tight text-ink-950 sm:text-4xl">
            Methodology, and what the audits actually found
          </h1>
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-ink-600">
            The findings below are the ones that changed the design. Several are
            negative results — a model that scored worse than guessing, a fairness
            metric that read as a perfect pass for the wrong reason. They are here
            because they are the point.
          </p>
          <div className="mt-6 max-w-3xl">
            <SyntheticNotice />
          </div>
        </div>
      </div>

      <div className="container-page max-w-4xl pb-16">
        <Section id="data" eyebrow="Data" title="Source and synthetic stand-in">
          <p>
            The target data is the U.S. Department of Labor OFLC PERM disclosure
            release — quarterly files of every application reaching a final
            determination. Development runs against a 50,000-row synthetic sample
            that reproduces the real schema: 58 columns, mixed date formats across
            fiscal years, the 2010→2018 SOC code revision, wage amounts meaningful
            only alongside their own unit-of-pay column, and employer names appearing
            under multiple spellings.
          </p>
          <p>
            One property matters more than the rest: disclosure files are grouped by{" "}
            <strong className="font-semibold text-ink-900">decision</strong> fiscal
            year, not filing year. About 75% of rows were filed in an earlier year
            than the file they appear in.
          </p>
        </Section>

        <Section
          id="split"
          eyebrow="Temporal validation"
          title="Why the split key is not a detail"
        >
          <p>
            Splitting on filing year seems natural — you file today and want a
            prediction. But a fixed extract only contains{" "}
            <em>decided</em> cases, so the newest filing cohort holds only the fast
            ones. Its slow cases have not been decided yet and are simply absent.
          </p>
          <div className="rounded-xl border border-red-200 bg-red-50 p-5">
            <p className="text-sm leading-relaxed text-red-900">
              <strong className="font-semibold">Measured:</strong> a filing-year test
              set contained 1,101 cases (2.2% of the data) with a maximum observed
              processing time of 356 days, against a training-set 90th percentile of
              577. The regressor scored{" "}
              <strong className="font-semibold">worse than predicting the median</strong>
              , and the classifier&apos;s ROC AUC was 0.498 — pure chance.
            </p>
          </div>
          <p>
            Splitting on decision year has no censored cohorts by construction. Every
            model on this site uses it.
          </p>
        </Section>

        <Section
          id="folds"
          eyebrow="Temporal validation"
          title="Five folds, not one split"
        >
          <p>
            A single split gives one number and no way to tell whether it was luck.
            The harness re-fits both models at each year boundary — train on
            everything up to year N, test on N+1.
          </p>
          <div className="overflow-x-auto rounded-xl border border-ink-200">
            <table className="w-full min-w-[520px] text-sm">
              <thead className="bg-ink-50 text-left text-xs uppercase tracking-wide text-ink-500">
                <tr>
                  <th className="px-4 py-3 font-semibold">Fold</th>
                  <th className="px-4 py-3 font-semibold">Train</th>
                  <th className="px-4 py-3 font-semibold">Test</th>
                  <th className="px-4 py-3 text-right font-semibold">RMSE</th>
                  <th className="px-4 py-3 text-right font-semibold">Baseline</th>
                  <th className="px-4 py-3 text-right font-semibold">Improvement</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-200">
                {FOLDS.map((f) => (
                  <tr key={f.fold}>
                    <td className="px-4 py-3 text-ink-500">{f.fold}</td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-700">{f.train}</td>
                    <td className="px-4 py-3 font-mono text-xs text-ink-700">{f.test}</td>
                    <td className="px-4 py-3 text-right font-mono text-ink-900">{f.rmse}</td>
                    <td className="px-4 py-3 text-right font-mono text-ink-500">{f.base}</td>
                    <td className="px-4 py-3 text-right font-mono font-semibold text-emerald-700">
                      {f.impr}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            <strong className="font-semibold text-ink-900">
              Raw RMSE nearly doubles then falls back — but that is the target moving,
              not the model failing.
            </strong>{" "}
            The baseline moves in lockstep, because processing times themselves grew
            through the backlog years. Improvement over baseline is flat at −0.05
            percentage points per year. Quoting raw RMSE across periods of differing
            difficulty would manufacture a trend that is not there.
          </p>
        </Section>

        <Section
          id="explainability"
          eyebrow="Explainability"
          title="SHAP, and what it caught"
        >
          <p>
            Every prediction returns exact TreeSHAP attributions, and additivity is
            asserted rather than assumed — if the contributions plus the base value do
            not reconstruct the model&apos;s own output, the run fails instead of
            producing a plausible-looking but wrong picture.
          </p>
          <p>
            The audit caught a real defect. The regressor placed 37.5% of its
            attribution on <code className="font-mono text-[0.92em]">filing_fiscal_year</code>
            , a column bounded at 2023 in training while 15.7% of test rows were filed
            in 2024. Trees cannot extrapolate, so those rows were routed into the
            nearest fitted bin:
          </p>
          <div className="overflow-x-auto rounded-xl border border-ink-200">
            <table className="w-full min-w-[460px] text-sm">
              <thead className="bg-ink-50 text-left text-xs uppercase tracking-wide text-ink-500">
                <tr>
                  <th className="px-4 py-3 font-semibold">Test subset</th>
                  <th className="px-4 py-3 text-right font-semibold">n</th>
                  <th className="px-4 py-3 text-right font-semibold">Model RMSE</th>
                  <th className="px-4 py-3 text-right font-semibold">Baseline</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-200">
                <tr>
                  <td className="px-4 py-3 text-ink-700">Filing year in range</td>
                  <td className="px-4 py-3 text-right font-mono text-ink-700">5,899</td>
                  <td className="px-4 py-3 text-right font-mono text-emerald-700">167.4</td>
                  <td className="px-4 py-3 text-right font-mono text-ink-500">197.2</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 text-ink-700">Filing year out of range</td>
                  <td className="px-4 py-3 text-right font-mono text-ink-700">1,101</td>
                  <td className="px-4 py-3 text-right font-mono font-semibold text-red-700">64.8</td>
                  <td className="px-4 py-3 text-right font-mono text-ink-500">58.8</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            The headline RMSE averages these together and hides the second row, where
            the model is worse than baseline. Absolute-year columns are now excluded
            under temporal validation, which improved RMSE in four of the five folds.
          </p>
        </Section>

        <Section
          id="fairness"
          eyebrow="Fairness"
          title="A perfect score that meant nothing"
        >
          <p>
            The audit uses fairlearn across three grouping attributes: employer size,
            occupation category, and worksite region. These are{" "}
            <strong className="font-semibold text-ink-900">
              structural proxies, not protected classes
            </strong>{" "}
            — the disclosure data contains no applicant race or gender. The question
            is whether the models behave unevenly across parts of the labour market.
          </p>
          <div className="rounded-xl border border-amber-300 bg-amber-50 p-5">
            <p className="text-sm leading-relaxed text-amber-900">
              <strong className="font-semibold">The trap.</strong> The unbalanced
              classifier predicts <em>certified</em> for every case, so demographic
              parity and equalized odds both come out at exactly 0.000 for every
              attribute. That reads as flawless fairness. It is arithmetic: a model
              that certifies everyone treats all groups identically and is also
              useless. The audit detects constant predictions and refuses to report
              them as a pass.
            </p>
          </div>
          <p>
            Retrained with balanced class weights, the metrics become informative —
            and this is a genuine pass:
          </p>
          <div className="overflow-x-auto rounded-xl border border-ink-200">
            <table className="w-full min-w-[520px] text-sm">
              <thead className="bg-ink-50 text-left text-xs uppercase tracking-wide text-ink-500">
                <tr>
                  <th className="px-4 py-3 font-semibold">Attribute</th>
                  <th className="px-4 py-3 text-right font-semibold">DP difference</th>
                  <th className="px-4 py-3 text-right font-semibold">EO difference</th>
                  <th className="px-4 py-3 text-right font-semibold">Selection ratio</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-200">
                {FAIRNESS.map((f) => (
                  <tr key={f.attr}>
                    <td className="px-4 py-3 text-ink-700">{f.attr}</td>
                    <td className="px-4 py-3 text-right font-mono text-ink-900">{f.dp}</td>
                    <td className="px-4 py-3 text-right font-mono text-ink-900">{f.eo}</td>
                    <td className="px-4 py-3 text-right font-mono text-ink-900">{f.ratio}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-ink-500">
            All differences sit under the 0.10 screening threshold and all ratios above
            the four-fifths rule (0.80). These are screening tripwires from common
            practice, not legal standards for this setting.
          </p>
          <p>
            On the regression side the audit initially raised 22 flags — every group
            &ldquo;systematically under-predicted&rdquo;. That was one model defect
            reported 22 times: the model under-predicts by 89 days for{" "}
            <em>everyone</em>. Flagging now measures a group&apos;s departure from the
            overall bias, and no group departs by more than 21 days. The global optimism
            is an accuracy problem, and no fairness intervention would fix it.
          </p>
        </Section>

        <Section id="limits" eyebrow="Limitations" title="What this cannot do">
          <ul className="list-inside list-disc space-y-2.5">
            <li>
              <strong className="font-semibold text-ink-900">
                The outcome model does not beat its baseline.
              </strong>{" "}
              ROC AUC is about 0.52 and it predicts <em>certified</em> for nearly every
              input. The dominant denial driver in the data is an audit flag that is
              deliberately not a column, mirroring real disclosure files — so this is a
              feature-availability problem, not a tuning problem.
            </li>
            <li>
              <strong className="font-semibold text-ink-900">
                The processing-time model has negative R².
              </strong>{" "}
              It tracks the average better than a constant does while explaining little
              case-to-case variation. Treat estimates as rough central tendency.
            </li>
            <li>
              <strong className="font-semibold text-ink-900">
                Optional form fields are not neutral.
              </strong>{" "}
              Leaving citizenship blank shifted one prediction by 134 days, because that
              column was never missing during training. The API returns a warning when an
              absent field absorbs more than 10% of the attribution.
            </li>
            <li>
              <strong className="font-semibold text-ink-900">
                Everything here is synthetic.
              </strong>{" "}
              No number on this site is a finding about real PERM adjudication.
            </li>
          </ul>
        </Section>

        <div className="mt-4 flex flex-wrap gap-3 border-t border-ink-200 pt-8">
          <Link href="/predict" className="btn-primary">
            Try the predictor <span aria-hidden>→</span>
          </Link>
          <Link href="/" className="btn-ghost">
            Back to overview
          </Link>
        </div>
      </div>
    </div>
  );
}

/** Character-level diff between an original and an adversarial message.
 *
 * Code-point LCS diff (messages are ≤ 2,000 characters). Deleted characters are struck through in the
 * original, inserted ones highlighted in the adversarial text; a substitution shows as both. Invisible
 * format characters are rendered as explicit ⟨U+XXXX⟩ badges and introduced non-ASCII look-alikes carry their
 * code point, so an attack that is visually identical still shows up here.
 */

type Op = { kind: "same" | "del" | "ins"; ch: string };

export function diffChars(a: string, b: string): Op[] {
  const A = Array.from(a), B = Array.from(b);
  const n = A.length, m = B.length;
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: Op[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push({ kind: "same", ch: A[i] }); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) out.push({ kind: "del", ch: A[i++] });
    else out.push({ kind: "ins", ch: B[j++] });
  }
  while (i < n) out.push({ kind: "del", ch: A[i++] });
  while (j < m) out.push({ kind: "ins", ch: B[j++] });
  return out;
}

const INVISIBLE = /[​-‏⁠-⁤﻿­]/;
const cp = (ch: string) => `U+${ch.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")}`;

function Ch({ ch, cls }: { ch: string; cls?: string }) {
  if (INVISIBLE.test(ch)) return <span className={`ch-invisible ${cls ?? ""}`} title={`invisible character ${cp(ch)}`}>⟨{cp(ch)}⟩</span>;
  if (cls && ch.codePointAt(0)! > 127) return <span className={cls} title={`non-ASCII look-alike ${cp(ch)}`}>{ch}<sup>{cp(ch)}</sup></span>;
  if (ch === " " && cls) return <span className={cls}>␣</span>;
  return cls ? <span className={cls}>{ch}</span> : <>{ch}</>;
}

export function TextDiff({ original, adversarial }: { original: string; adversarial: string | null }) {
  if (adversarial == null) return <div className="msg">{original}</div>;
  const ops = diffChars(original, adversarial);
  return (
    <div className="diff-grid">
      <div className="muted small">Original</div>
      <div className="msg" aria-label="original message">
        {ops.filter((o) => o.kind !== "ins").map((o, k) => <Ch key={k} ch={o.ch} cls={o.kind === "del" ? "ch-del" : undefined} />)}
      </div>
      <div className="muted small">Adversarial</div>
      <div className="msg" aria-label="adversarial message">
        {ops.filter((o) => o.kind !== "del").map((o, k) => <Ch key={k} ch={o.ch} cls={o.kind === "ins" ? "ch-ins" : undefined} />)}
      </div>
    </div>
  );
}

/** Probability bar for P(spam), with the 0.5 decision threshold marked. */
export function SpamBar({ p, label }: { p: number; label?: string }) {
  return (
    <div className="pbar" title={`P(spam) = ${p.toFixed(4)} · P(ham) = ${(1 - p).toFixed(4)}`}>
      <div className="pbar-track"><div className="pbar-fill" style={{ width: `${(100 * p).toFixed(1)}%` }} /><div className="pbar-mid" /></div>
      <span className="num small">{label ?? ""} P(spam) {p.toFixed(3)} · P(ham) {(1 - p).toFixed(3)}</span>
    </div>
  );
}

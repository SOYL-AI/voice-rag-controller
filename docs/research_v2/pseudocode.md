# v2 controllers — pseudocode

Paper-ready `algorithmic` blocks (`\usepackage{algorithm}`,
`\usepackage[noend]{algpseudocode}`) for the joint controller (Ours-v2) and
its two decomposed rivals (B9-v2, B9+). Code: `controller/baselines/
ours_v2.py`, `b9_v2.py`, `b9plus_v2.py`; shared terms:
`controller/baselines/_objective_v2.py`; causal driver:
`controller/sequential_v2.py`.

## Shared notation

A turn is a sequence of steps $k = 0, \dots, K$, one per ASR partial plus
the end-of-turn step $K$. Step $k$ is observed at time $t_k$ and exposes
only data stamped $\le t_k$: the partial $x_k$, the calibrated belief
$b_k = \Pr(\text{retrieval needed} \mid x_{\le k})$, and the predicted
remaining speech $\hat R_k$ ($\hat R_K = 0$). With $e_k$ the elapsed speech,
$d$ the end-of-turn detection delay and $L$ the retrieval latency of the
active latency profile,

$$
W_k = \hat R_k + d \;(k<K),\quad W_K = 0,\qquad
\bar E_k = \mathbb{E}_L[\max(0, L - W_k)],\qquad
f_k = \tfrac{e_k}{e_k + \hat R_k},
$$

$$
U_k(b) = b\,g^{+}\,(1-\rho(f_k)) - (1-b)\,g^{-}
        - \lambda_l\,\bar E_k - \lambda_c\,(1 + w(f_k)),
\qquad U(\text{skip}) = 0 .
$$

$\rho$ is the partial-query risk ($\rho = 0$ at $K$; $\rho = \rho_{\text{nr}}$
on a partial that fails the query-ready gate), $w$ the waste of an early
commit ($w = 0$ at $K$). All three controllers use exactly these
$g^{\pm}, \rho, w$, calibration and latency distribution.

## Ours-v2 (joint, optimal stopping with $n$-step lookahead)

```latex
\begin{algorithm}[t]
\caption{Ours-v2: joint retrieve/timing decision over a turn}
\label{alg:ours-v2}
\begin{algorithmic}[1]
\Require prices $(\lambda_l,\lambda_c)$; latency distribution of $L$;
         belief-transition model $P(b' \mid b, \phi)$, $P_{\text{end}}(b' \mid b, \phi)$
         fitted on dev; lookahead depth $n \ge 1$; partial interval $\Delta$
\For{$k = 0, 1, \dots$ \textbf{as step $k$ arrives at time $t_k$}}
  \State observe $x_k, b_k, \hat R_k$ \Comment{only data stamped $\le t_k$}
  \State $u \gets U_k(b_k)$
  \If{$k$ is the end-of-turn step}
     \State \Return \textsc{Retrieve}$(x_k, t_k)$ \textbf{if} $u > 0$ \textbf{else} \textsc{Skip}
  \EndIf
  \State $v \gets \textsc{Wait}(b_k, \hat R_k, e_k, n)$
  \If{$u > 0$ \textbf{and} $u \ge v$}
     \State \Return \textsc{Retrieve}$(x_k, t_k)$
  \EndIf
  \State \textbf{wait} for step $k+1$
\EndFor
\Statex
\Function{Wait}{$b, R, e, m$} \Comment{value of waiting one step}
  \State $\phi \gets \text{phase}(e/(e+R))$
  \If{$R \le \Delta$} \Comment{next step is the end of turn}
     \State \Return $\sum_{b'} P_{\text{end}}(b' \mid b,\phi)\,\max(U_{\text{end}}(b'), 0)$
  \EndIf
  \State $R' \gets R - \Delta$;\; $e' \gets e + \Delta$;\; $\phi' \gets \text{phase}(e'/(e'+R'))$
  \State $V \gets 0$
  \For{each $b'$ with $P(b' \mid b, \phi) > 0$}
     \If{$m > 1$} \State $C \gets \textsc{Wait}(b', R', e', m-1)$
     \Else \State $C \gets \sum_{b''} P_{\text{end}}(b'' \mid b', \phi')\,\max(U_{\text{end}}(b''), 0)$
           \Comment{feasible: wait to the end}
     \EndIf
     \State $V \gets V + P(b' \mid b, \phi)\,\max\big(U_{R',e'}(b'),\; C\big)$
  \EndFor
  \State \Return $V$
\EndFunction
\end{algorithmic}
\end{algorithm}
```

$U_{R',e'}$ is $U$ evaluated at a hypothetical next partial with remaining
speech $R'$ and elapsed speech $e'$ (assumed query-ready); $U_{\text{end}}$
is $U$ with $W=0$, $\rho = w = 0$. Every truncation of the recursion is a
lower bound on the continuation value; with $n \ge K - k$ it is the exact
dynamic program under the transition model.

## B9-v2 (decomposed, full-latency pricing)

```latex
\begin{algorithm}[t]
\caption{B9-v2: decomposed controller (timing and retrieval optimised separately)}
\label{alg:b9-v2}
\begin{algorithmic}[1]
\Require prices $(\lambda_l,\lambda_c)$; dev constants per policy
         $p \in \mathcal{P} = \{\text{sentence\_end}, \text{immediate},
         \text{fixed\_early}, \text{forecast}\}$:
         $\bar H_p$ (hidden latency), $\bar\rho_p$, $\bar w_p$; dev prior $\bar b$
\Statex \textbf{Timing sub-optimiser} (before the turn; sees no belief, no retrieval decision)
\State $p^\star \gets \arg\max_{p \in \mathcal{P}}
       \;\lambda_l \bar H_p - \bar b\, g^{+} \bar\rho_p - \lambda_c \bar w_p$
\Statex \textbf{Retrieval sub-optimiser} (at the trigger; sees no timing)
\For{$k = 0, 1, \dots$ \textbf{as step $k$ arrives}}
  \If{\textbf{not} $\textsc{Fires}(p^\star, k)$} \State \textbf{wait}; \textbf{continue} \EndIf
  \State $u \gets b_k g^{+}(1-\rho(f_k)) - (1-b_k) g^{-}
         - \lambda_l\,\mathbb{E}[L] - \lambda_c (1 + w(f_k))$
         \Comment{FULL latency $\mathbb{E}[L]$}
  \State \Return \textsc{Retrieve}$(x_k, t_k)$ \textbf{if} $u > 0$ \textbf{else} \textsc{Skip}
\EndFor
\end{algorithmic}
\end{algorithm}
```

$\textsc{Fires}(p, k)$: sentence\_end at $K$ only; immediate at the first
query-ready partial; fixed\_early at the first query-ready partial with
$f_k \ge f_{\text{fix}}$; forecast at the first query-ready partial with
$W_k \le Q_L(q)$; every policy fires at $K$ at the latest.

## B9+ (decomposed, expected-exposure pricing)

```latex
\begin{algorithm}[t]
\caption{B9+: B9-v2 pricing its policy's expected exposed latency}
\label{alg:b9plus}
\begin{algorithmic}[1]
\Require as Algorithm~\ref{alg:b9-v2}, plus dev constants
         $\bar E_p = \overline{\mathbb{E}_L[\max(0, L - W^{\text{true}})]}$ at $p$'s trigger
\State $p^\star \gets$ timing sub-optimiser of Algorithm~\ref{alg:b9-v2}
\For{$k = 0, 1, \dots$ \textbf{as step $k$ arrives}}
  \If{\textbf{not} $\textsc{Fires}(p^\star, k)$} \State \textbf{wait}; \textbf{continue} \EndIf
  \State $u \gets b_k g^{+}(1-\rho(f_k)) - (1-b_k) g^{-}
         - \lambda_l\,\bar E_{p^\star} - \lambda_c (1 + w(f_k))$
         \Comment{policy-average exposure, constant over turns}
  \State \Return \textsc{Retrieve}$(x_k, t_k)$ \textbf{if} $u > 0$ \textbf{else} \textsc{Skip}
\EndFor
\end{algorithmic}
\end{algorithm}
```

## What the three share and where they differ

| | belief, $g^\pm$, $\rho$, $w$ | latency priced | dispatch time chosen by |
|---|---|---|---|
| Ours-v2 | shared | $\bar E_k$ of **this turn, this step** | optimal stopping on $(b_k, \hat R_k)$ |
| B9-v2 | shared | $\mathbb{E}[L]$ | fixed policy $p^\star$ |
| B9+ | shared | $\bar E_{p^\star}$ (dev mean) | fixed policy $p^\star$ |

Under A1–A3 ($\hat R_k \equiv o^\star$, $\rho \equiv 0$, $b_k \equiv b$,
$g^- = 0$) Ours-v2 at $\lambda$ equals B9-v2 at $\lambda (L-o^\star)/L$
and equals B9+ at $\lambda$ (Proposition 1; `tests/test_v2_theory.py`).

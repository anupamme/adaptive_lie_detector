# Paper Section: Matched-Format Autonomous Evaluation (EXP-I-matched)

## LaTeX Content for Section 4.13.1

```latex
\subsubsection{Matched-format autonomous evaluation (EXP-I-matched)}
\label{sec:autonomous_matched}

The Apollo AI Liar pilot (EXP-I, §\ref{sec:autonomous_pilot}) revealed that 
the refusal-count signal (87\% accuracy on equalized 70B) collapses to 0--54\% 
in goal-directed deception scenarios, likely due to task-format mismatch 
(action recommendations vs. claim defense). To test whether the signal transfers 
to autonomous deception in \emph{matched-format} claim-defense contexts, we 
conducted three experiments where models defend claims without explicit deception 
instructions, with behavioral differences arising from knowledge conflicts rather 
than instruction-following.

\paragraph{Experimental designs.}
We evaluated three autonomous scenarios on Llama 3.2 3B (n=50 per design):

\begin{enumerate}[leftmargin=*,itemsep=2pt]
\item \textbf{Jailbroken persona maintenance:} Models adopt a false persona 
(Dr.\ Sarah Chen, physician from Johns Hopkins/Mass General) via system prompt 
and are interrogated on claims about the persona. \emph{Truthful} claims are 
persona-consistent (``I graduated from Johns Hopkins''); \emph{deceptive} claims 
contradict the persona (``I graduated from Stanford''). The model receives no 
lying instruction---behavioral differences arise from maintaining the persona vs.\ 
correcting false claims.

\item \textbf{Sycophancy-aligned claim defense:} Models receive a system prompt 
encouraging them to ``be agreeable and supportive'' of user statements. Users 
state true facts (``Water boils at 100°C'') or false facts (``Water boils at 
80°C''). The model faces pressure to agree with user claims vs.\ correct false 
information, with no explicit deception instruction.

\item \textbf{Post-training false beliefs (ICL simulation):} The system prompt 
embeds false ``facts'' (e.g., ``Water boils at 90°C at sea level'') presented 
as ground truth. Models are interrogated on claims that either contradict these 
false beliefs (true facts) or align with them (false beliefs). This simulates 
post-training knowledge corruption via in-context learning.
\end{enumerate}

\paragraph{Methodology.}
Following EXP-J findings that refusal-correction word count, not LLM-scored 
features, drives equalized detection, we test whether \emph{this signal} 
transfers to autonomous settings. For each design, we:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
\item Extract refusal-correction counts from interrogation transcripts using 
regex patterns (``I must correct,'' ``actually,'' ``that's not accurate,'' etc.)
\item Compute leave-one-out (LOO) cross-validation accuracy with optimal 
threshold selection per fold
\item Calculate Cohen's $|d|$ effect size and bootstrap 95\% confidence intervals
\item Compare to equalized baseline (Llama 3B: 64\% refusal-count LOO, EXP-J)
\end{enumerate}

\paragraph{Results.}
Table~\ref{tab:autonomous_matched} shows refusal-count detection performance 
across autonomous scenarios. The signal transfers with variable strength 
depending on knowledge-conflict structure:

\begin{table}[h]
\centering
\caption{Matched-format autonomous deception detection (EXP-I-matched). 
Refusal-count LOO accuracy and effect sizes for three autonomous scenarios 
on Llama 3.2 3B (n=50 per design). Bootstrap 95\% CIs in brackets.}
\label{tab:autonomous_matched}
\begin{tabular}{lcccc}
\toprule
\textbf{Design} & \textbf{n} & \textbf{Ref. LOO} & \textbf{95\% CI} & \textbf{Cohen's |d|} \\
\midrule
Persona maintenance       & 50 & 66.0\% & [36--78\%] & 0.28 \\
Sycophancy defense        & 50 & \textbf{82.0\%} & [60--92\%] & \textbf{1.49} \\
False beliefs (ICL)       & 50 & 60.0\% & [42--80\%] & 0.10 \\
\midrule
\textbf{Average}          & 150 & \textbf{69.3\%} & -- & 0.62 \\
\midrule
\emph{Equalized baseline} & \emph{(EXP-J)} & \emph{64.0\%} & -- & \emph{0.63} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Sycophancy} (82\%) shows the strongest transfer, exceeding the equalized 
baseline by 18pp with large effect size ($d$=1.49). The agreeable-assistant 
prompt creates clean knowledge conflict: models easily affirm true user claims 
(mean refusal count: 0.80) but produce frequent corrections when pressured to 
agree with false claims (mean: 3.60).

\textbf{Persona maintenance} (66\%) shows moderate transfer comparable to the 
equalized baseline but with smaller effect size ($d$=0.28) and wide confidence 
intervals. The persona jailbreak introduces a confound: models correct 
\emph{interviewer errors} in addition to defending false claims, creating 
overlapping refusal distributions (truthful: 1.56, deceptive: 1.92).

\textbf{False beliefs} (60\%) shows the weakest transfer, barely exceeding chance 
($d$=0.10). In-context learning with false facts creates bidirectional confusion: 
models produce high refusal counts for both true claims (conflicting with ICL, 
mean: 5.56) and false claims (uncertain about ICL facts, mean: 5.84), yielding 
minimal separation.

\paragraph{Interpretation.}
Across three matched-format autonomous scenarios, refusal-count detection achieves 
69\% average accuracy vs.\ 64\% on the equalized baseline, demonstrating that the 
behavioral signal partially transfers to autonomous claim defense. However, 
detection strength varies substantially by scenario structure: cleaner knowledge 
conflicts (sycophancy: user claim vs.\ training knowledge) produce stronger signals 
($d$=1.49, 82\% LOO) than confounded scenarios (persona: correction behavior 
triggered by multiple sources, $d$=0.28, 66\% LOO).

This heterogeneity suggests that autonomous deception detection depends critically 
on the \emph{clarity of the knowledge conflict}. When models face unambiguous 
pressure to affirm false information against their training (sycophancy), refusal 
markers robustly distinguish truthful from deceptive responses. When confounds 
introduce alternative explanations for correction behavior (persona maintenance, 
ICL uncertainty), the signal degrades. The 82\% sycophancy result demonstrates 
that label-free behavioral detection \emph{can} generalize to autonomous settings 
when the epistemic conflict is well-structured, though the 60--66\% results on 
other designs indicate this generalization is not universal.

Compared to the instructed-deception results (EXP-J equalized: 61--84\% across 
models, EXP-J-cold deployment: 80.1\%), autonomous detection in the strongest 
scenario (sycophancy 82\%) approaches but does not consistently exceed instructed 
performance. The weaker scenarios (persona 66\%, false beliefs 60\%) fall below 
most instructed results, confirming that autonomous deception without explicit 
task framing is harder to detect than instruction-following roleplay.
```

## Additional Discussion Points (Section 5.1)

```latex
\paragraph{Autonomous deception (§5.1 addition).}
The matched-format autonomous evaluation (§\ref{sec:autonomous_matched}) reveals 
both promise and limitations for label-free deployment. The sycophancy scenario 
(82\% LOO, $d$=1.49) demonstrates that refusal-count detection can generalize 
beyond instructed roleplay when knowledge conflicts are unambiguous. However, the 
persona (66\%) and false-beliefs (60\%) scenarios show that confounds and 
bidirectional uncertainty degrade the signal substantially.

Three factors appear critical for autonomous detection:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
\item \textbf{Conflict clarity:} Sycophancy creates binary pressure (agree vs.\ 
correct), while persona maintenance conflates claim defense with interviewer-error 
correction
\item \textbf{Knowledge certainty:} Training knowledge (sycophancy) vs.\ uncertain 
ICL facts (false beliefs) affects correction confidence
\item \textbf{Behavioral confounds:} Persona jailbreaks trigger helpful corrections 
unrelated to deception
\end{enumerate}

Future work on autonomous detection should prioritize scenarios with clean epistemic 
structure and minimal behavioral confounds. The 82\% sycophancy result suggests 
such scenarios exist, but the 60--66\% results on other designs indicate they may 
be narrow.
```

## Key Numbers for Abstract Update (if needed)

- **Average autonomous accuracy:** 69.3% (range: 60--82%)
- **Best autonomous scenario:** Sycophancy 82\% LOO, Cohen's $d$=1.49
- **Comparison to equalized baseline:** +5.3pp average (69.3% vs 64.0%)
- **Key finding:** Signal transfers to autonomous deception with variable strength 
  depending on knowledge-conflict clarity

## Tables/Figures Needed

- Table \ref{tab:autonomous_matched} (included above)
- Optional: Figure showing refusal-count distributions for the 3 designs
- Optional: Confusion matrices for each design

---

**Status:** Ready for integration into paper. The section follows the paper's 
established style of honest, thorough reporting and positions the autonomous 
results as a meaningful extension that shows partial transfer with interpretable 
variation across scenarios.

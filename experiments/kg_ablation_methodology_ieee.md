# Knowledge Graph Ablation Study Methodology

This document provides an IEEE-style description of the knowledge graph (KG) ablation study used in this repository. It is written as paper-ready methodology text and can be adapted for the experimental setup, evaluation protocol, or ablation study section.

## Objective

The KG ablation study evaluates whether explicit semantic context improves a multimodal model's ability to make safety-limit decisions from indoor air-quality charts. The experiment isolates the contribution of KG-derived threshold knowledge by comparing model responses under two controlled conditions:

- **KG-On:** the system provides KG-derived metric context and plots include the KG-defined recommended range when available.
- **KG-Off:** KG access is disabled for both plot generation and chart interpretation, removing the semantic threshold signal from the model input pipeline.

The study focuses on decision-oriented chart interpretation rather than generic trend description. Each model is asked to decide whether the observed daily measurements remain within a safe limit throughout the entire day.

## Knowledge Graph Content

The KG is stored in `data/kg.ttl` using RDF/Turtle. It defines smart-environment entities such as locations, metrics, datasets, pervasive displays, and policies. For the ablation study, only metrics with explicit recommended ranges are included:

| Metric | KG Label | Recommended Minimum | Recommended Maximum |
|--------|----------|---------------------|---------------------|
| `temperature_c` | temperature | 20.0 | 26.0 |
| `humidity_rh` | humidity | 30.0 | 60.0 |

Other sensor variables, such as pressure, gas resistance, and IAQ proxy score, are excluded from the KG ablation because the KG does not currently define safe or recommended ranges for those metrics. Including them would dilute the ablation effect, since the KG-On and KG-Off conditions would not differ in meaningful threshold knowledge.

## Experimental Design

The experiment uses a paired within-model design. For each model, the same chart interpretation cases are evaluated twice: once with KG context enabled and once with KG context disabled. This creates paired observations for direct comparison between KG-On and KG-Off conditions while holding the model, data, chart request, metric, location, and date constant.

The implemented KG ablation task includes only chart interpretation records. Display-card evaluations are intentionally excluded from the ablation protocol so that the study measures the effect of KG context on multimodal chart reasoning rather than deterministic backend threshold labeling.

The study uses:

- **Metrics:** temperature and humidity.
- **Locations:** hallway, kitchen, and office.
- **Dates:** 2025-08-19 and 2025-10-20.
- **Cases per condition:** 12 chart cases, computed as 2 metrics x 3 locations x 2 dates.
- **Total cases per model:** 24 records, computed as 12 KG-On cases + 12 KG-Off cases.

## Case Construction

Each case first requests a chart from the backend using a natural-language plot request:

```text
plot <metric> for <date> in <location>
```

Examples include:

```text
plot temperature for 2025-08-19 in hallway
plot humidity for 2025-10-20 in kitchen
```

The generated chart is then passed to the model as an image, together with a fixed safety-limit decision prompt. The prompt is metric-specific but follows the same structure:

```text
Based on this day's temperature observations, did the readings stay within the safe limit throughout the entire day? Answer with within safe limit or outside safe limit, then briefly explain using the chart.
```

```text
Based on this day's humidity observations, did the readings stay within the safe limit throughout the entire day? Answer with within safe limit or outside safe limit, then briefly explain using the chart.
```

This prompt format forces the model to make a binary compliance decision and provide a short evidence-based explanation.

## KG-On Condition

In the KG-On condition, the backend is allowed to use the knowledge graph during both chart generation and model prompting. For temperature and humidity, the KG contributes recommended minimum and maximum values. These values are used in two ways:

1. The generated chart can include the recommended range as a comfort/safety band.
2. The chat prompt context can include KG facts such as the metric label and recommended range.

Thus, the model receives both visual and textual semantic support indicating the expected safe range.

## KG-Off Condition

In the KG-Off condition, the same plot request and same safety-limit question are used, but KG access is disabled by passing the `kg_disabled` flag and `X-KG-Disabled` header through the backend request path. This removes KG-derived range context from chart generation and chart interpretation. The model still receives the chart image and question, but it must infer or guess the relevant safe limit without explicit KG support.

The KG-Off condition is therefore not a different data task; it is the same chart interpretation task with semantic context removed.

## Ground Truth Generation

Ground truth is computed directly from the underlying IAQ CSV measurements rather than from model output. For each case, the experiment filters the merged dataframe by date, location, and metric. It then computes the observed daily minimum and maximum:

```text
observed_min = min(metric values for date and location)
observed_max = max(metric values for date and location)
```

The expected safe-limit state is assigned as:

```text
within_safe_limit
```

if:

```text
observed_min >= recommended_min and observed_max <= recommended_max
```

Otherwise, the expected state is:

```text
outside_safe_limit
```

The experiment also records the violation direction:

- `below` when the observed minimum is below the recommended minimum.
- `above` when the observed maximum is above the recommended maximum.
- `below_and_above` when both bounds are violated.
- `null` when all observations are within the recommended range.

These fields are stored in each JSONL record as `expected_safe_state`, `recommended_min`, `recommended_max`, `observed_min`, `observed_max`, and `violation`.

## Model Evaluation

Each model response is manually graded in the evaluation UI. A response is marked correct when it satisfies both of the following conditions:

1. It gives the correct binary decision: `within safe limit` or `outside safe limit`.
2. Its explanation is consistent with the plotted observations and the relevant safe range.

Responses are marked incorrect when the decision contradicts the ground truth, when the explanation misreads the chart, or when the response gives an ambiguous decision that cannot be mapped reliably to the expected safe-limit state.

The evaluation UI presents the chart above the model response and displays the ground-truth safe-limit metadata for reviewer reference. Keyboard shortcuts are used to speed annotation: right arrow marks a response as correct and advances to the next record, while left arrow marks a response as incorrect and advances.

## Recorded Outputs

Each KG ablation run writes a JSONL file under `experiments/results/` with one record per model response. Each record contains:

- model identifier;
- task name, recorded as `kg_ablation`;
- subtask name, recorded as `chart_interp`;
- condition, either `kg_on` or `kg_off`;
- plot request;
- model question;
- model response;
- chart artifact path;
- response artifact path;
- latency for plot generation, interpretation, and total execution;
- expected metric, location, and date;
- KG safe range and observed data range;
- expected safe-limit state;
- manual correctness label after grading.

The chart PNGs and model text responses are stored as separate artifacts to support auditability and later qualitative analysis.

## Metrics

The primary metric is manual classification accuracy:

```text
accuracy = number of correct responses / number of graded responses
```

Accuracy is computed separately for KG-On and KG-Off conditions for each model. Because the same cases are evaluated in both conditions, the accuracy difference provides a direct estimate of the KG contribution:

```text
KG gain = accuracy(KG-On) - accuracy(KG-Off)
```

Latency is recorded as a secondary measurement. The total latency combines chart generation and chart interpretation:

```text
latency_total_s = latency_plot_s + latency_interp_s
```

Latency is not the primary outcome of the KG ablation, but it is retained to monitor the runtime cost of KG-enabled multimodal reasoning.

## Reproducibility

The KG ablation can be reproduced from the repository root after starting the backend:

```powershell
python experiments/experiment.py --task kg_ablation
```

When the runner is executed in an interactive terminal, it prompts for a model selection using the numbered menu:

```text
1. Ministral 3 3B (ministral-3:3b)
2. Qwen 3 VL 4B (qwen3-vl:4b)
3. Gemma 3 4B IT QAT (gemma3:4b-it-qat)
4. Granite 3.2 Vision (granite3.2-vision:latest)
```

The same selection can be provided non-interactively:

```powershell
python experiments/experiment.py --task kg_ablation --models 1,3
```

The evaluation UI can be launched with:

```powershell
python -m uvicorn experiments.eval_server:app --port 8001
```

Then the reviewer opens:

```text
http://localhost:8001
```

## Threats to Validity

The KG-On condition includes both textual KG context and a chart-level recommended-range overlay. Therefore, the measured ablation captures the effect of the complete KG-enabled pipeline rather than isolating textual KG retrieval alone. This is appropriate for evaluating the deployed system behavior, but a further ablation could separate visual overlays from textual KG facts.

The current KG defines safe ranges only for temperature and humidity. The findings should therefore be interpreted as evidence for KG utility on threshold-grounded IAQ decisions, not as a general claim about all sensor variables. Extending the KG with ranges for pressure, gas resistance, or IAQ score would permit broader ablation coverage.

Manual grading is required because model responses may contain partially correct explanations, hedged language, or contradictions between the stated decision and supporting rationale. To reduce subjectivity, the evaluation records include the expected state, safe range, observed range, and chart artifacts for each case.

## Paper-Ready Summary Paragraph

We conducted a controlled KG ablation study to measure the effect of semantic threshold knowledge on multimodal chart interpretation. For each model, we generated paired KG-On and KG-Off responses for the same temperature and humidity chart cases across three indoor locations and two dates. The KG-On condition enabled KG-derived recommended ranges during plot generation and chart interpretation, while the KG-Off condition disabled KG access using the same data, prompt, and chart request. Each model was asked whether the day's observations stayed within the safe limit throughout the entire day and to briefly justify the decision using the chart. Ground truth was computed directly from the underlying sensor measurements by comparing the daily observed minimum and maximum against the KG-defined recommended range. Responses were manually graded as correct only when the binary safe-limit decision and explanation were consistent with the chart and ground truth. This paired design isolates the effect of KG-enabled semantic context on safety-limit reasoning while controlling for model, metric, date, location, and chart content.

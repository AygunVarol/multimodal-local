"""Fixed experiment cases used by the paper evaluation scripts."""

from __future__ import annotations

from itertools import product


LOCATIONS = ["hallway", "kitchen", "office"]

METRICS = [
    ("temperature_c", "temperature"),
    ("humidity_rh", "humidity"),
    ("pressure_hpa", "pressure"),
    ("gas_resistance_ohms", "gas resistance"),
    ("IAQ_proxy", "IAQ"),
]

PLOT_DATES = ["2025-08-19", "2025-10-20"]

DISPLAY_DATES = [
    ("2025-08-19", "08:15"),
    ("2025-08-20", "15:15"),
    ("2025-08-25", "11:45"),
    ("2025-09-03", "18:30"),
    ("2025-10-20", "09:00"),
]

NL_SQL_CASES = [
    {
        "question": f"What is the average {metric_label} of {location}?",
        "expected_col": metric_col,
        "expected_agg": "mean",
        "location": location,
    }
    for location, (metric_col, metric_label) in product(LOCATIONS, METRICS)
] + [
    {
        "question": f"What is the {agg_label} {metric_label} of {location}?",
        "expected_col": metric_col,
        "expected_agg": agg_name,
        "location": location,
    }
    for location, metric_col, metric_label, agg_name, agg_label in [
        ("hallway", "temperature_c", "temperature", "min", "minimum"),
        ("hallway", "humidity_rh", "humidity", "max", "maximum"),
        ("hallway", "pressure_hpa", "pressure", "min", "minimum"),
        ("hallway", "gas_resistance_ohms", "gas resistance", "max", "maximum"),
        ("hallway", "IAQ_proxy", "IAQ", "min", "minimum"),
        ("kitchen", "temperature_c", "temperature", "max", "maximum"),
        ("kitchen", "humidity_rh", "humidity", "min", "minimum"),
        ("kitchen", "pressure_hpa", "pressure", "max", "maximum"),
        ("kitchen", "gas_resistance_ohms", "gas resistance", "min", "minimum"),
        ("kitchen", "IAQ_proxy", "IAQ", "max", "maximum"),
        ("office", "temperature_c", "temperature", "min", "minimum"),
        ("office", "humidity_rh", "humidity", "max", "maximum"),
        ("office", "pressure_hpa", "pressure", "min", "minimum"),
        ("office", "gas_resistance_ohms", "gas resistance", "max", "maximum"),
        ("office", "IAQ_proxy", "IAQ", "max", "maximum"),
    ]
]

DISPLAY_CARD_CASES = [
    {
        "date": date,
        "location": location,
        "time": time_text,
        "metric": metric_col,
    }
    for date, time_text in DISPLAY_DATES
    for location in LOCATIONS
    for metric_col in ("temperature_c", "humidity_rh")
]

PLOT_CASES = [
    {
        "request": f"plot {metric_label} for {date_text} in {location}",
        "expected_metric": metric_col,
        "expected_location": location,
        "expected_date": date_text,
    }
    for date_text in PLOT_DATES
    for location in LOCATIONS
    for metric_col, metric_label in METRICS
]

_CHART_QUESTIONS = [
    "explain the trend",
    "what stands out in this chart",
    "summarize the main pattern",
    "is there any anomaly to note",
    "describe whether the signal looks stable",
]

_CHART_RUBRICS = {
    "temperature_c": ["identify the main trend", "note comfort-band crossings if visible", "do not invent values"],
    "humidity_rh": ["identify the main trend", "note low-humidity risk if visible", "do not invent values"],
    "pressure_hpa": ["identify the main trend", "mention major fluctuations only if visible", "do not invent values"],
    "gas_resistance_ohms": ["identify the main trend", "mention spikes or dips if visible", "do not invent values"],
    "IAQ_proxy": ["identify the main trend", "mention notable peaks or drops if visible", "do not invent values"],
}

CHART_INTERP_CASES = [
    {
        "plot_request": case["request"],
        "question": _CHART_QUESTIONS[index % len(_CHART_QUESTIONS)],
        "rubric": _CHART_RUBRICS[case["expected_metric"]],
    }
    for index, case in enumerate(PLOT_CASES)
]

KG_ABLATION_METRICS = {"temperature_c", "humidity_rh"}

_KG_ABLATION_SAFE_LIMIT_QUESTIONS = {
    "temperature_c": (
        "Based on this day's temperature observations, did the readings stay within the safe limit throughout the entire day? "
        "Answer with within safe limit or outside safe limit, then briefly explain using the chart."
    ),
    "humidity_rh": (
        "Based on this day's humidity observations, did the readings stay within the safe limit throughout the entire day? "
        "Answer with within safe limit or outside safe limit, then briefly explain using the chart."
    ),
}

_KG_ABLATION_RUBRICS = {
    "temperature_c": [
        "use the temperature safe range when available",
        "decide whether the plotted observations stay within the safe limit",
        "briefly cite visible chart evidence",
    ],
    "humidity_rh": [
        "use the humidity safe range when available",
        "decide whether the plotted observations stay within the safe limit",
        "briefly cite visible chart evidence",
    ],
}

KG_ABLATION_CHART_CASES = [
    {
        "plot_request": case["request"],
        "question": _KG_ABLATION_SAFE_LIMIT_QUESTIONS[case["expected_metric"]],
        "rubric": _KG_ABLATION_RUBRICS[case["expected_metric"]],
        "expected_metric": case["expected_metric"],
        "expected_location": case["expected_location"],
        "expected_date": case["expected_date"],
    }
    for case in PLOT_CASES
    if case["expected_metric"] in KG_ABLATION_METRICS
]

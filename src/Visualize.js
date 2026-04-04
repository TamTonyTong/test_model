import React, { useEffect, useMemo, useState } from "react";
import {
    Bar,
    BarChart,
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from "recharts";
import "./styles/Visualize.css";

const MODEL_VISUALIZE_CONFIG = [
    {
        key: "cnn-lstm",
        label: "CNN-LSTM",
        folder: "artifacts_cnn_lstm",
        historyCsv: "tbrgs_cnn_lstm_global_history.csv",
        siteSummaryCsv: "cnn_lstm_global_per_site_summary.csv",
        imageFiles: [
            { title: "Training Curve", file: "tbrgs_cnn_lstm_global_training_curve.png" },
            { title: "Train vs Validation Diagnostic", file: "cnn_lstm_train_vs_val_diagnostic.png" }
        ]
    },
    {
        key: "gru",
        label: "GRU",
        folder: "artifacts_gru",
        historyCsv: "tbrgs_gru_global_history.csv",
        siteSummaryCsv: "gru_global_per_site_summary.csv",
        imageFiles: [
            { title: "Training History", file: "gru_global_training_history.png" }
        ]
    },
    {
        key: "lstm",
        label: "LSTM",
        folder: "artifacts_lstm",
        historyCsv: "tbrgs_lstm_global_history.csv",
        siteSummaryCsv: "lstm_global_per_site_summary.csv",
        imageFiles: [
            { title: "Training History", file: "lstm_global_training_history.png" }
        ]
    },
    {
        key: "lstm-gru",
        label: "LSTM-GRU",
        folder: "artifacts_lstm_gru",
        historyCsv: "tbrgs_lstm_gru_global_history.csv",
        siteSummaryCsv: "lstm_gru_global_per_site_summary.csv",
        imageFiles: [
            { title: "Training History", file: "lstm_gru_global_training_history.png" }
        ]
    }
];

const DEFAULT_MODEL_KEY = MODEL_VISUALIZE_CONFIG[0].key;

const HISTORY_SERIES_CONFIG = [
    { key: "loss", label: "Loss", color: "#38bdf8", axis: "left" },
    { key: "mae", label: "MAE", color: "#22c55e", axis: "left" },
    { key: "rmse", label: "RMSE", color: "#f59e0b", axis: "left" },
    { key: "val_loss", label: "Val Loss", color: "#6366f1", axis: "left" },
    { key: "val_mae", label: "Val MAE", color: "#a78bfa", axis: "left" },
    { key: "val_rmse", label: "Val RMSE", color: "#f43f5e", axis: "left" },
    { key: "learning_rate", label: "Learning Rate", color: "#f97316", axis: "right" }
];

const SITE_SERIES_CONFIG = [
    { key: "r2", label: "R2", color: "#22c55e" },
    { key: "rmse", label: "RMSE", color: "#38bdf8" },
    { key: "mae", label: "MAE", color: "#a78bfa" },
    { key: "mape", label: "MAPE (%)", color: "#f97316" }
];

function parseCsvLine(line) {
    const values = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < line.length; i += 1) {
        const char = line[i];

        if (char === '"') {
            if (inQuotes && line[i + 1] === '"') {
                current += '"';
                i += 1;
            } else {
                inQuotes = !inQuotes;
            }
            continue;
        }

        if (char === "," && !inQuotes) {
            values.push(current);
            current = "";
            continue;
        }

        current += char;
    }

    values.push(current);
    return values;
}

function parseCsv(text) {
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    if (lines.length < 2) return [];

    const headers = parseCsvLine(lines[0]).map(col => col.trim());

    return lines.slice(1).map(line => {
        const cols = parseCsvLine(line);
        const row = {};

        headers.forEach((header, idx) => {
            row[header] = (cols[idx] || "").trim();
        });

        return row;
    });
}

function toNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : null;
}

function pickValue(row, keys) {
    for (const key of keys) {
        if (Object.prototype.hasOwnProperty.call(row, key)) {
            return row[key];
        }
    }
    return null;
}

function formatValue(value, digits = 4) {
    if (!Number.isFinite(value)) return "N/A";
    return value.toFixed(digits);
}

function formatHistoryValue(metricKey, value) {
    if (!Number.isFinite(value)) return "N/A";
    if (metricKey === "learning_rate") {
        return value.toExponential(2);
    }
    return value.toFixed(5);
}

function formatSiteMetricValue(metricKey, value) {
    if (!Number.isFinite(value)) return "N/A";
    if (metricKey === "mape") return `${value.toFixed(2)}%`;
    if (metricKey === "r2") return value.toFixed(4);
    return value.toFixed(5);
}

export default function Visualize() {
    const [selectedModelKey, setSelectedModelKey] = useState(DEFAULT_MODEL_KEY);
    const [activeHistoryMetric, setActiveHistoryMetric] = useState("loss");
    const [activeSiteMetric, setActiveSiteMetric] = useState("r2");
    const [historyRows, setHistoryRows] = useState([]);
    const [siteRows, setSiteRows] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const selectedModel = useMemo(
        () => MODEL_VISUALIZE_CONFIG.find(model => model.key === selectedModelKey) || MODEL_VISUALIZE_CONFIG[0],
        [selectedModelKey]
    );

    useEffect(() => {
        let cancelled = false;

        const loadModelData = async () => {
            setLoading(true);
            setError("");
            setHistoryRows([]);
            setSiteRows([]);

            try {
                const historyPath = `/modelVisualize/${selectedModel.folder}/${selectedModel.historyCsv}`;
                const sitePath = `/modelVisualize/${selectedModel.folder}/${selectedModel.siteSummaryCsv}`;

                const [historyResponse, siteResponse] = await Promise.all([
                    fetch(historyPath),
                    fetch(sitePath)
                ]);

                if (!historyResponse.ok || !siteResponse.ok) {
                    throw new Error("One or more visualization files could not be loaded.");
                }

                const [historyCsvText, siteCsvText] = await Promise.all([
                    historyResponse.text(),
                    siteResponse.text()
                ]);

                const parsedHistoryRows = parseCsv(historyCsvText).map(row => ({
                    loss: toNumber(row.loss),
                    mae: toNumber(row.mae),
                    rmse: toNumber(row.rmse),
                    val_loss: toNumber(row.val_loss),
                    val_mae: toNumber(row.val_mae),
                    val_rmse: toNumber(row.val_rmse),
                    learning_rate: toNumber(row.learning_rate)
                }));

                const parsedSiteRows = parseCsv(siteCsvText).map(row => ({
                    site: pickValue(row, ["site_id", "Site", "SITE", "site"]),
                    rmse: toNumber(pickValue(row, ["RMSE", "rmse"])),
                    mae: toNumber(pickValue(row, ["MAE", "mae"])),
                    mape: toNumber(pickValue(row, ["MAPE (%)", "MAPE", "mape", "mape (%)"])),
                    r2: toNumber(pickValue(row, ["R2", "r2"])),
                    samples: toNumber(pickValue(row, ["n_test", "Samples", "samples", "N_TEST"]))
                }));

                if (!cancelled) {
                    setHistoryRows(parsedHistoryRows);
                    setSiteRows(parsedSiteRows);
                }
            } catch (loadError) {
                if (!cancelled) {
                    setError(loadError.message || "Failed to load visualization files.");
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        };

        loadModelData();

        return () => {
            cancelled = true;
        };
    }, [selectedModel]);

    const trainingSummary = useMemo(() => {
        if (historyRows.length === 0) {
            return null;
        }

        let bestValRmse = Number.POSITIVE_INFINITY;
        let bestValLoss = Number.POSITIVE_INFINITY;
        let bestEpoch = null;

        historyRows.forEach((row, idx) => {
            if (Number.isFinite(row.val_rmse) && row.val_rmse < bestValRmse) {
                bestValRmse = row.val_rmse;
                bestEpoch = idx + 1;
            }
            if (Number.isFinite(row.val_loss) && row.val_loss < bestValLoss) {
                bestValLoss = row.val_loss;
            }
        });

        const lastEpoch = historyRows[historyRows.length - 1];

        return {
            epochs: historyRows.length,
            bestValRmse: Number.isFinite(bestValRmse) ? bestValRmse : null,
            bestValLoss: Number.isFinite(bestValLoss) ? bestValLoss : null,
            bestEpoch,
            finalRmse: lastEpoch?.rmse,
            finalValRmse: lastEpoch?.val_rmse
        };
    }, [historyRows]);

    const siteSummary = useMemo(() => {
        if (siteRows.length === 0) {
            return null;
        }

        const rowsWithRmse = siteRows.filter(row => Number.isFinite(row.rmse));
        const sortedByRmse = [...rowsWithRmse].sort((a, b) => a.rmse - b.rmse);
        const avgRmse = rowsWithRmse.length > 0
            ? rowsWithRmse.reduce((sum, row) => sum + row.rmse, 0) / rowsWithRmse.length
            : null;

        return {
            totalSites: siteRows.length,
            avgRmse,
            bestSite: sortedByRmse[0] || null,
            topRows: sortedByRmse.slice(0, 10)
        };
    }, [siteRows]);

    const historyChartRows = useMemo(() => {
        return historyRows.map((row, idx) => ({
            epoch: idx + 1,
            ...row
        }));
    }, [historyRows]);

    const latestHistoryValues = useMemo(() => {
        if (historyRows.length === 0) {
            return null;
        }
        return historyRows[historyRows.length - 1];
    }, [historyRows]);

    const topSitesByR2 = useMemo(() => {
        return siteRows
            .filter(row => Number.isFinite(row.r2) && row.site !== null && row.site !== undefined && row.site !== "")
            .sort((a, b) => b.r2 - a.r2)
            .slice(0, 20)
            .map(row => ({
                site: String(row.site),
                rmse: row.rmse,
                mae: row.mae,
                mape: row.mape,
                r2: row.r2
            }));
    }, [siteRows]);

    const topSiteMetricAverages = useMemo(() => {
        if (topSitesByR2.length === 0) {
            return null;
        }

        const calculateAverage = key => {
            const values = topSitesByR2
                .map(row => row[key])
                .filter(value => Number.isFinite(value));

            if (values.length === 0) {
                return null;
            }

            return values.reduce((sum, value) => sum + value, 0) / values.length;
        };

        return {
            r2: calculateAverage("r2"),
            rmse: calculateAverage("rmse"),
            mae: calculateAverage("mae"),
            mape: calculateAverage("mape")
        };
    }, [topSitesByR2]);

    return (
        <div className="app visualize-page">
            <main className="app-main visualize-main">
                <div className="visualize-column">
                    <section className="card visualize-panel">
                        <div className="card-title">Model Result Visualizer</div>
                        <p className="visualize-lead">
                            Select a trained model to inspect its global training history and per-site evaluation summary.
                        </p>

                        <div className="form-group" style={{ marginBottom: 0 }}>
                            <label className="form-label" htmlFor="visualize-model-select">Model</label>
                            <select
                                id="visualize-model-select"
                                className="form-select"
                                value={selectedModel.key}
                                onChange={event => setSelectedModelKey(event.target.value)}
                            >
                                {MODEL_VISUALIZE_CONFIG.map(model => (
                                    <option key={model.key} value={model.key}>{model.label}</option>
                                ))}
                            </select>
                        </div>

                        <div className="visualize-status-row">
                            <span className="visualize-chip">Folder: {selectedModel.folder}</span>
                            {loading ? <span className="visualize-chip loading">Loading files...</span> : null}
                        </div>
                        {error ? <div className="error-banner">⚠ {error}</div> : null}
                    </section>

                    <section className="card visualize-panel">
                        <div className="card-title">Training Summary</div>
                        {!trainingSummary ? (
                            <div className="visualize-placeholder">No history data available.</div>
                        ) : (
                            <>
                                <div className="visualize-metric-grid">
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Epochs</span>
                                        <strong className="visualize-metric-value">{trainingSummary.epochs}</strong>
                                    </article>
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Best Val RMSE</span>
                                        <strong className="visualize-metric-value">{formatValue(trainingSummary.bestValRmse, 5)}</strong>
                                    </article>
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Best Epoch</span>
                                        <strong className="visualize-metric-value">{trainingSummary.bestEpoch ?? "N/A"}</strong>
                                    </article>
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Final Val RMSE</span>
                                        <strong className="visualize-metric-value">{formatValue(trainingSummary.finalValRmse, 5)}</strong>
                                    </article>
                                </div>

                                <div className="visualize-history-head">
                                    <div>
                                        <p className="visualize-history-title">Interactive Training History</p>
                                        <p className="visualize-history-subtitle">
                                            Epoch is used as the x-axis row index for all metrics.
                                        </p>
                                    </div>

                                    <div className="visualize-history-toggle-grid">
                                        {HISTORY_SERIES_CONFIG.map(series => (
                                            <button
                                                key={series.key}
                                                type="button"
                                                className="visualize-history-toggle"
                                                data-active={activeHistoryMetric === series.key}
                                                onClick={() => setActiveHistoryMetric(series.key)}
                                            >
                                                <span className="visualize-history-toggle-label">{series.label}</span>
                                                <span className="visualize-history-toggle-value">
                                                    {formatHistoryValue(series.key, latestHistoryValues?.[series.key])}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="visualize-history-chart-wrap">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <LineChart
                                            data={historyChartRows}
                                            margin={{ left: 10, right: 14, top: 8, bottom: 4 }}
                                        >
                                            <CartesianGrid vertical={false} stroke="rgba(148, 163, 184, 0.22)" />
                                            <XAxis
                                                dataKey="epoch"
                                                tickLine={false}
                                                axisLine={false}
                                                tickMargin={8}
                                                minTickGap={24}
                                                stroke="#94a3b8"
                                            />
                                            <YAxis
                                                yAxisId="left"
                                                tickLine={false}
                                                axisLine={false}
                                                tickMargin={8}
                                                stroke="#94a3b8"
                                                width={66}
                                                tickFormatter={value => Number(value).toFixed(3)}
                                            />
                                            <YAxis
                                                yAxisId="right"
                                                orientation="right"
                                                tickLine={false}
                                                axisLine={false}
                                                tickMargin={8}
                                                stroke="#f97316"
                                                width={78}
                                                tickFormatter={value => Number(value).toExponential(1)}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    background: "#0f172a",
                                                    borderColor: "rgba(148, 163, 184, 0.35)",
                                                    borderRadius: "10px",
                                                    color: "#e2e8f0"
                                                }}
                                                labelFormatter={value => `Epoch ${value}`}
                                                formatter={(value, name) => [
                                                    formatHistoryValue(name, Number(value)),
                                                    HISTORY_SERIES_CONFIG.find(series => series.key === name)?.label || name
                                                ]}
                                            />
                                            {HISTORY_SERIES_CONFIG.map(series => (
                                                <Line
                                                    key={series.key}
                                                    yAxisId={series.axis}
                                                    type="monotone"
                                                    dataKey={series.key}
                                                    stroke={series.color}
                                                    strokeWidth={activeHistoryMetric === series.key ? 2.6 : 1.6}
                                                    strokeOpacity={activeHistoryMetric === series.key ? 1 : 0.28}
                                                    dot={false}
                                                    isAnimationActive={false}
                                                />
                                            ))}
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>

                            </>
                        )}
                    </section>
                </div>

                <div className="visualize-column visualize-column-right">       
                    <section className="card visualize-panel">
                        <div className="card-title">Per-Site Evaluation</div>
                        {!siteSummary ? (
                            <div className="visualize-placeholder">No per-site summary data available.</div>
                        ) : (
                            <>
                                <div className="visualize-metric-grid">
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Total Sites</span>
                                        <strong className="visualize-metric-value">{siteSummary.totalSites}</strong>
                                    </article>
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Average RMSE</span>
                                        <strong className="visualize-metric-value">{formatValue(siteSummary.avgRmse, 5)}</strong>
                                    </article>
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Best Site</span>
                                        <strong className="visualize-metric-value">{siteSummary.bestSite?.site ?? "N/A"}</strong>
                                    </article>
                                    <article className="visualize-metric-card">
                                        <span className="visualize-metric-label">Best Site RMSE</span>
                                        <strong className="visualize-metric-value">{formatValue(siteSummary.bestSite?.rmse, 5)}</strong>
                                    </article>
                                </div>

                            </>
                        )}

                        {/* <div className="card-title"></div> */}
                        {topSitesByR2.length === 0 ? (
                            <div className="visualize-placeholder">No valid R2 values available for top-site comparison.</div>
                        ) : (
                            <>
                                <div className="visualize-history-head">
                                    <div>
                                        <p className="visualize-history-title">Interactive Per-Site Metric Chart</p>
                                        <p className="visualize-history-subtitle">
                                            Top 20 rows ranked by R2 across the selected model dataset.
                                        </p>
                                    </div>

                                    <div className="visualize-history-toggle-grid">
                                        {SITE_SERIES_CONFIG.map(series => (
                                            <button
                                                key={series.key}
                                                type="button"
                                                className="visualize-history-toggle"
                                                data-active={activeSiteMetric === series.key}
                                                onClick={() => setActiveSiteMetric(series.key)}
                                            >
                                                <span className="visualize-history-toggle-label">{series.label}</span>
                                                <span className="visualize-history-toggle-value">
                                                    {formatSiteMetricValue(series.key, topSiteMetricAverages?.[series.key])}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                <div className="visualize-history-chart-wrap">
                                    <ResponsiveContainer width="100%" height={280}>
                                        <BarChart
                                            data={topSitesByR2}
                                            margin={{ left: 10, right: 10, top: 8, bottom: 8 }}
                                        >
                                            <CartesianGrid vertical={false} stroke="rgba(148, 163, 184, 0.22)" />
                                            <XAxis
                                                dataKey="site"
                                                tickLine={false}
                                                axisLine={false}
                                                tickMargin={8}
                                                interval={0}
                                                angle={-35}
                                                textAnchor="end"
                                                height={66}
                                                stroke="#94a3b8"
                                            />
                                            <YAxis
                                                tickLine={false}
                                                axisLine={false}
                                                tickMargin={8}
                                                stroke="#94a3b8"
                                                width={68}
                                                tickFormatter={value => formatSiteMetricValue(activeSiteMetric, Number(value))}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    background: "#0f172a",
                                                    borderColor: "rgba(148, 163, 184, 0.35)",
                                                    borderRadius: "10px",
                                                    color: "#e2e8f0"
                                                }}
                                                labelFormatter={value => `Site ${value}`}
                                                formatter={(value, name) => [
                                                    formatSiteMetricValue(name, Number(value)),
                                                    SITE_SERIES_CONFIG.find(series => series.key === name)?.label || name
                                                ]}
                                            />
                                            <Bar
                                                dataKey={activeSiteMetric}
                                                fill={SITE_SERIES_CONFIG.find(series => series.key === activeSiteMetric)?.color || "#38bdf8"}
                                                radius={[4, 4, 0, 0]}
                                            />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </>
                        )}
                    </section>

                    
                </div>
            </main>
        </div>
    );
}

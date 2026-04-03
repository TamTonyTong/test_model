import React, { useEffect, useMemo, useState } from "react";
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

function formatPercent(value) {
    if (!Number.isFinite(value)) return "N/A";
    return `${value.toFixed(2)}%`;
}

export default function Visualize() {
    const [selectedModelKey, setSelectedModelKey] = useState(DEFAULT_MODEL_KEY);
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

    const recentEpochRows = useMemo(() => {
        if (historyRows.length === 0) {
            return [];
        }

        const startIndex = Math.max(0, historyRows.length - 8);
        return historyRows.slice(startIndex).map((row, idx) => ({
            epoch: startIndex + idx + 1,
            ...row
        }));
    }, [historyRows]);

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

                                <div className="visualize-table-wrap">
                                    <table className="visualize-table">
                                        <thead>
                                            <tr>
                                                <th>Epoch</th>
                                                <th>Loss</th>
                                                <th>RMSE</th>
                                                <th>Val Loss</th>
                                                <th>Val RMSE</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {recentEpochRows.map(row => (
                                                <tr key={row.epoch}>
                                                    <td>{row.epoch}</td>
                                                    <td>{formatValue(row.loss, 5)}</td>
                                                    <td>{formatValue(row.rmse, 5)}</td>
                                                    <td>{formatValue(row.val_loss, 5)}</td>
                                                    <td>{formatValue(row.val_rmse, 5)}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </>
                        )}
                    </section>
                </div>

                <div className="visualize-column">
                    <section className="card visualize-panel">
                        <div className="card-title">Training Plots</div>
                        <div className="visualize-image-grid">
                            {selectedModel.imageFiles.map(image => (
                                <figure key={image.file} className="visualize-image-card">
                                    <img
                                        src={`/modelVisualize/${selectedModel.folder}/${image.file}`}
                                        alt={`${selectedModel.label} ${image.title}`}
                                        loading="lazy"
                                    />
                                    <figcaption>{image.title}</figcaption>
                                </figure>
                            ))}
                        </div>
                    </section>

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

                                <div className="visualize-table-wrap">
                                    <table className="visualize-table">
                                        <thead>
                                            <tr>
                                                <th>Site</th>
                                                <th>RMSE</th>
                                                <th>MAE</th>
                                                <th>MAPE</th>
                                                <th>R2</th>
                                                <th>Samples</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {siteSummary.topRows.map(row => (
                                                <tr key={`${row.site}-${row.rmse}`}>
                                                    <td>{row.site || "N/A"}</td>
                                                    <td>{formatValue(row.rmse, 5)}</td>
                                                    <td>{formatValue(row.mae, 5)}</td>
                                                    <td>{formatPercent(row.mape)}</td>
                                                    <td>{formatValue(row.r2, 4)}</td>
                                                    <td>{Number.isFinite(row.samples) ? row.samples : "N/A"}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </>
                        )}
                    </section>
                </div>
            </main>
        </div>
    );
}
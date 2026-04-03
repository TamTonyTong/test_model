# Test Cases for Map Routing and Prediction

## Scope
This document defines 12 functional and robustness test scenarios for the map routing flow in the React frontend and Flask prediction backend integration.

## Preconditions
- Frontend app is running.
- Dataset is loaded from `/mapInfo/Traffic_Count_Locations_FILTERED.csv`.
- Unless otherwise specified, backend API is available at `http://127.0.0.1:5000/predict`.

---

## 1. Valid Route Calculation
**Input**
- Origin: valid OBJECTID (example: 33485)
- Destination: valid OBJECTID (example: 32758)
- Model: `CNN-LSTM`

**Steps**
1. Set Origin to `33485`.
2. Set Destination to `32758`.
3. Select model `CNN-LSTM`.
4. Click **Find Top 5 Paths**.

**Expected**
- Up to 5 candidate routes are returned and drawn.
- Route lines are visible on the map.
- Route time summary is shown.
- Best route estimate appears in summary metrics.
- UI remains responsive.

---

## 2. Same Origin and Destination
**Input**
- Origin: `33485`
- Destination: `33485`

**Steps**
1. Set Origin to `33485`.
2. Set Destination to `33485`.
3. Attempt to calculate route.

**Expected**
- Route calculation is blocked or returns no route gracefully.
- No crash or unhandled exception.
- Optional warning/alert is shown to user.

---

## 3. Invalid Node Input (Non-existent ID)
**Input**
- Origin: `99999` (not in dataset)
- Destination: valid OBJECTID (example: `32758`)

**Steps**
1. Type `99999` in Origin.
2. Type `32758` Destination.

**Expected**
- Origin cannot be resolved (`resolveFromInput` returns `null`).
- Origin remains unset.
- **Find Top 5 Paths** stays disabled.
- UI does not crash.

---

## 4. Invalid Text Input (Wrong Location Name)
**Input**
- Origin: `abcxyz`
- Destination: valid OBJECTID

**Steps**
1. Type `abcxyz` in Origin input.
2. Provide a valid Destination.

**Expected**
- No matching suggestion is found.
- Origin remains `null`.
- Route button remains disabled.
- UI stays stable.

---

<!-- ## 5. Missing SCATS ID (Dataset Issue)
**Scenario**
- One or more nodes have `scats = null`.

**Steps**
1. Select an origin/destination where one node has no SCATS ID.
2. Run route calculation.

**Expected**
- Fallback to `nodeId` is used in prediction payload.
- `/predict` calls still succeed.
- Route calculation completes without API failure. -->

---

## 5. API Failure / Backend Down
**Scenario**
- Flask backend is stopped.

**Steps**
1. Stop backend service.
2. Set Origin to `33485`.
3. Set Destination to `32758`.
4. Run route calculation.

**Expected**
- `getFlow()` falls back to default flow `1000` and message displays "Backend is not connected, the result is just based on search algorithm without traffic flow predicted."
- Paths are still recalculated and displayed (if graph path exists).
- No frontend crash.
- Errors are handled gracefully.

---

## 6. Extreme Prediction Values
**Scenario**
- API returns extreme or invalid predictions:
- Very high flow (example: `100000`)
- `NaN` or `null`

**Steps**
1. Start frontend and backend services.
2. Open browser DevTools Console (to inspect route recalculation logs).
3. In `backend/app.py`, locate response field `"predicted_flow_per_hour"` in `/predict` return block.
4. For sub-test A (very high value), temporarily set `"predicted_flow_per_hour": 100000`.
5. Save backend file and rerun backend service.
6. In frontend map, set Origin `33485` and Destination `32758`, then click **Find Top 5 Paths**.
7. Verify route still renders and travel-time estimate increases compared to normal run.
8. For sub-test B (`null`), temporarily set `"predicted_flow_per_hour": None`, then rerun backend and run route again.
9. Verify invalid value falls back to flow `1000` (check segment logs in Console) and UI remains stable.
10. For sub-test C (`NaN`), temporarily set `"predicted_flow_per_hour": "NaN"`, then rerun backend and run route again.
11. Verify invalid value falls back to flow `1000`, no `NaN` appears in path totals, and route summary remains visible.
12. Restore original backend response line `"predicted_flow_per_hour": round(pred_flow_hour, 2)` after testing.

**Expected**
- Invalid values (`NaN`, `null`) are replaced with fallback `1000`.
- Very high finite flows increase travel-time estimates.
- No `NaN` propagation to path totals.
- UI remains stable.

---

## 7. Graph Edge Case (Disconnected Nodes)
**Scenario**
- Chosen nodes are in disconnected graph components.

**Steps**
1. Select origin/destination known to be disconnected.
2. Click **Find Top 5 Paths**.

**Expected**
- `yenKShortest` returns empty list.
- User sees alert: `No path found`.
- No crash.

---

## 8. Large Dataset Performance Test
**Scenario**
- Near maximum node load (`MAX_NODES = 5000`).

**Steps**
1. Load dataset near 5000 map nodes.
2. Select valid origin and destination.
3. Run route calculation and observe responsiveness.

**Expected**
- Map renders within reasonable time.
- Route calculation completes without freezing browser.
- Flow prefetch executes with configured concurrency.
- UI remains interactive.

---

## 9. Model Switching Test
**Scenario**
- Switch model sequence: `CNN-LSTM` -> `GRU` -> `LSTM`.

**Steps**
1. Compute route with `CNN-LSTM`.
2. Switch to `GRU` and recompute.
3. Switch to `LSTM` and recompute.

**Expected**
- Flow cache is cleared on model change.
- New prediction calls use selected model.
- Route results update after each switch.

---

## 10. Input Typing + Suggestion Test
**Scenario**
- User types partial IDs and location text.

**Steps**
1. Type partial OBJECTID in Origin.
2. Type partial location name in Destination.
3. Pick suggestion from dropdown for both fields.

**Expected**
- Suggestions appear as user types.
- Selecting a suggestion updates origin/destination values correctly.
- Path state resets appropriately after input changes.
- Route button enables only when both nodes are valid.

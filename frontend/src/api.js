const BASE = "http://localhost:8000";

export async function fetchLogs() {
  try {
    const res = await fetch(`${BASE}/api/logs`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    // Backend returns { logs: [...objects] } — pass through as-is, no string parsing
    return data?.logs ?? [];
  } catch (err) {
    console.error("fetchLogs error", err);
    return [];
  }
}

export async function fetchClips() {
  try {
    const res = await fetch(`${BASE}/api/clips`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data?.clips ?? [];
  } catch (err) {
    console.error("fetchClips error", err);
    return [];
  }
}
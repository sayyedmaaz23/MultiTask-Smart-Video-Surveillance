import axios from "axios";
const BASE = "http://localhost:8000";

export async function fetchLogs() {
  try {
    const res = await axios.get(`${BASE}/api/logs`);
    const rawLogs = res.data?.logs ?? [];

    // Convert: "2025-10-25 12:10:03 - Smoking detected"
    // Into:   { timestamp: "...", type: "smoking", message: "Smoking detected" }
    const parsedLogs = rawLogs.map(line => {
      const [timestamp, message] = line.split(" - ");

      let type = "unknown";
      if (message.toLowerCase().includes("smoking")) type = "smoking";
      if (message.toLowerCase().includes("group")) type = "group";

      return { timestamp, message, type };
    });

    return parsedLogs.reverse(); // newest first (optional)
  } catch (e) {
    console.error("fetchLogs error", e);
    return [];
  }
}

export function snapshotUrl(relativePath) {
  return `${BASE}${relativePath}`;
}

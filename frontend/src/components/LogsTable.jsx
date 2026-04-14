import React from "react";
import { Box, Card, CardContent, Typography, Stack, Chip } from "@mui/material";
import TerminalIcon from "@mui/icons-material/Terminal";

/**
 * Backend sends log lines as plain strings in this format:
 *   "2026-04-13T23:44:22.123456 - Burglary [23:44:07 → 23:44:22] (conf: 0.7395)"
 *
 * This parser pulls out the pieces so we can display them nicely.
 */
function parseLogLine(line) {
  if (typeof line !== "string") return { raw: String(line) };

  // timestamp — everything before the first " - "
  const dashIdx = line.indexOf(" - ");
  const timestamp = dashIdx > -1 ? line.slice(0, dashIdx).trim() : null;
  const rest      = dashIdx > -1 ? line.slice(dashIdx + 3).trim() : line;

  // window — optional "[HH:MM:SS → HH:MM:SS]"
  const windowMatch = rest.match(/\[(\d{2}:\d{2}:\d{2})\s*→\s*(\d{2}:\d{2}:\d{2})\]/);
  const clipStart   = windowMatch ? windowMatch[1] : null;
  const clipEnd     = windowMatch ? windowMatch[2] : null;

  // confidence — optional "(conf: 0.7395)"
  const confMatch = rest.match(/\(conf:\s*([\d.]+)\)/);
  const conf      = confMatch ? parseFloat(confMatch[1]) : null;

  // type — everything before the window bracket (or conf bracket)
  const typeRaw = rest
    .replace(/\[.*?\]/, "")
    .replace(/\(conf:.*?\)/, "")
    .trim();

  return { timestamp, type: typeRaw || null, clipStart, clipEnd, conf, raw: line };
}

export default function LogsTable({ logs = [] }) {
  return (
    <Box sx={{ maxHeight: 320, overflow: "auto", mt: 1 }}>
      {logs.length === 0 && (
        <Typography variant="body2" sx={{ color: "#789", textAlign: "center", py: 3 }}>
          <TerminalIcon sx={{ fontSize: 18, verticalAlign: "middle", mr: 1 }} />
          No logs yet
        </Typography>
      )}
      <Stack spacing={1.5} sx={{ px: 1, pb: 1 }}>
        {logs.map((entry, idx) => {
          const { timestamp, type, clipStart, clipEnd, conf } = parseLogLine(entry);
          const timeLabel = timestamp
            ? (() => { try { return new Date(timestamp).toLocaleTimeString(); } catch { return timestamp; } })()
            : null;
          const confLabel = conf != null ? `${(conf * 100).toFixed(1)}%` : null;
          const window    = clipStart && clipEnd ? `${clipStart} → ${clipEnd}` : null;

          return (
            <Card
              key={idx}
              elevation={3}
              sx={{
                bgcolor: "rgba(51,191,255,0.06)",
                borderLeft: "4px solid var(--primary)",
                color: "#e1e9fa",
                borderRadius: 2,
              }}
            >
              <CardContent sx={{ py: 1, px: 1.5, "&:last-child": { pb: 1 } }}>
                {/* Top row: type + confidence badge */}
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.5 }}>
                  <Typography variant="body2" sx={{ color: "#33bfff", fontWeight: 700, fontSize: "0.85rem" }}>
                    <TerminalIcon sx={{ fontSize: 14, verticalAlign: "middle", mr: 0.5 }} />
                    {type || "Log entry"}
                  </Typography>
                  {confLabel && (
                    <Chip
                      label={confLabel}
                      size="small"
                      sx={{
                        height: 17,
                        fontSize: "0.67rem",
                        bgcolor: "rgba(255,255,255,0.08)",
                        color: "#33bfff",
                        border: "1px solid #33bfff55",
                      }}
                    />
                  )}
                </Box>

                {/* Clip window */}
                {window && (
                  <Typography variant="caption" sx={{ color: "#33bfff", fontFamily: "monospace", display: "block" }}>
                    🕐 {window}
                  </Typography>
                )}

                {/* Wall-clock time */}
                {timeLabel && (
                  <Typography variant="caption" sx={{ color: "#7f93b2" }}>
                    {timeLabel}
                  </Typography>
                )}
              </CardContent>
            </Card>
          );
        })}
      </Stack>
    </Box>
  );
}

import React, { useState } from "react";
import {
  Box, Card, CardContent, Typography, Stack, Chip,
  Dialog, DialogTitle, DialogContent, DialogActions,
  Button, IconButton,
} from "@mui/material";
import TerminalIcon from "@mui/icons-material/Terminal";
import PlayCircleOutlineIcon from "@mui/icons-material/PlayCircleOutline";
import CloseIcon from "@mui/icons-material/Close";
import OndemandVideoIcon from "@mui/icons-material/OndemandVideo";

/**
 * Accepts either:
 *  - Structured objects: { timestamp, type, message, confidence, clip_start, clip_end, clip_file }
 *  - Legacy plain strings (fallback)
 */
function parseEntry(entry) {
  if (typeof entry === "object" && entry !== null) return entry;

  // Legacy plain-string fallback
  const line = String(entry);
  const dashIdx = line.indexOf(" - ");
  const timestamp = dashIdx > -1 ? line.slice(0, dashIdx).trim() : null;
  const rest      = dashIdx > -1 ? line.slice(dashIdx + 3).trim() : line;

  const windowMatch = rest.match(/\[(\d{2}:\d{2}:\d{2})\s*→\s*(\d{2}:\d{2}:\d{2})\]/);
  const confMatch   = rest.match(/\(conf:\s*([\d.]+)\)/);
  const typeRaw     = rest.replace(/\[.*?\]/, "").replace(/\(conf:.*?\)/, "").trim();

  return {
    timestamp,
    type:       typeRaw || "Log entry",
    clip_start: windowMatch?.[1] ?? null,
    clip_end:   windowMatch?.[2] ?? null,
    confidence: confMatch ? parseFloat(confMatch[1]) : null,
    clip_file:  null,
  };
}

function ClipModal({ open, onClose, clipFile, logEntry }) {
  const videoUrl = clipFile ? `http://localhost:8000/clips/${clipFile}` : null;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: "rgba(10, 18, 35, 0.97)",
          backdropFilter: "blur(16px)",
          border: "1px solid rgba(51,191,255,0.25)",
          borderRadius: 3,
          boxShadow: "0 8px 48px rgba(0,0,0,0.7)",
        },
      }}
    >
      <DialogTitle
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          color: "#33bfff",
          fontWeight: 700,
          pb: 1,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <OndemandVideoIcon />
          {logEntry?.type ?? "Anomaly"} Clip
          {logEntry?.clip_start && logEntry?.clip_end && (
            <Chip
              label={`${logEntry.clip_start} → ${logEntry.clip_end}`}
              size="small"
              sx={{
                ml: 1,
                fontFamily: "monospace",
                fontSize: "0.72rem",
                bgcolor: "rgba(51,191,255,0.12)",
                color: "#33bfff",
                border: "1px solid #33bfff44",
              }}
            />
          )}
        </Box>
        <IconButton onClick={onClose} size="small" sx={{ color: "#7f93b2" }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ pt: 0 }}>
        {videoUrl ? (
          <Box
            component="video"
            src={videoUrl}
            controls
            autoPlay
            sx={{
              width: "100%",
              borderRadius: 2,
              border: "1px solid rgba(51,191,255,0.2)",
              bgcolor: "#000",
              maxHeight: 480,
              display: "block",
            }}
          />
        ) : (
          <Box
            sx={{
              textAlign: "center",
              py: 6,
              color: "#7f93b2",
            }}
          >
            <OndemandVideoIcon sx={{ fontSize: 48, mb: 1, opacity: 0.4 }} />
            <Typography variant="body2">
              No clip available for this alert.
              <br />
              YOLO-only alerts (Smoking, Group) don't produce saved clips.
            </Typography>
          </Box>
        )}

        {/* Meta row */}
        <Box sx={{ mt: 1.5, display: "flex", gap: 1, flexWrap: "wrap" }}>
          {logEntry?.confidence != null && (
            <Chip
              label={`Confidence: ${(logEntry.confidence * 100).toFixed(1)}%`}
              size="small"
              sx={{
                bgcolor: "rgba(255,172,48,0.1)",
                color: "#ffac30",
                border: "1px solid #ffac3055",
                fontSize: "0.7rem",
              }}
            />
          )}
          {logEntry?.timestamp && (
            <Chip
              label={(() => {
                try { return new Date(logEntry.timestamp).toLocaleTimeString(); }
                catch { return logEntry.timestamp; }
              })()}
              size="small"
              sx={{
                bgcolor: "rgba(51,191,255,0.06)",
                color: "#7f93b2",
                border: "1px solid #33bfff22",
                fontSize: "0.7rem",
              }}
            />
          )}
        </Box>
      </DialogContent>

      <DialogActions sx={{ px: 2.5, pb: 2 }}>
        <Button onClick={onClose} sx={{ color: "#7f93b2" }}>
          Close
        </Button>
        {videoUrl && (
          <Button
            variant="outlined"
            size="small"
            href={videoUrl}
            target="_blank"
            rel="noopener noreferrer"
            sx={{ borderColor: "#33bfff55", color: "#33bfff" }}
          >
            Open in new tab
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}

export default function LogsTable({ logs = [] }) {
  const [selectedEntry, setSelectedEntry] = useState(null);

  const handleClick = (entry) => {
    setSelectedEntry(entry);
  };

  const handleClose = () => setSelectedEntry(null);

  return (
    <>
      <Box sx={{ maxHeight: 320, overflow: "auto", mt: 1 }}>
        {logs.length === 0 && (
          <Typography variant="body2" sx={{ color: "#789", textAlign: "center", py: 3 }}>
            <TerminalIcon sx={{ fontSize: 18, verticalAlign: "middle", mr: 1 }} />
            No logs yet
          </Typography>
        )}
        <Stack spacing={1.5} sx={{ px: 1, pb: 1 }}>
          {logs.map((rawEntry, idx) => {
            const entry = parseEntry(rawEntry);
            const {
              timestamp,
              type,
              clip_start,
              clip_end,
              confidence,
              clip_file,
            } = entry;

            const timeLabel = timestamp
              ? (() => { try { return new Date(timestamp).toLocaleTimeString(); } catch { return timestamp; } })()
              : null;
            const confLabel   = confidence != null ? `${(confidence * 100).toFixed(1)}%` : null;
            const window      = clip_start && clip_end ? `${clip_start} → ${clip_end}` : null;
            const hasClip     = Boolean(clip_file);
            const isClickable = true; // always clickable — shows "no clip" message for YOLO alerts

            return (
              <Card
                key={idx}
                elevation={3}
                onClick={() => handleClick(entry)}
                sx={{
                  bgcolor: hasClip
                    ? "rgba(51,191,255,0.10)"
                    : "rgba(51,191,255,0.06)",
                  borderLeft: `4px solid ${hasClip ? "#33bfff" : "var(--primary)"}`,
                  color: "#e1e9fa",
                  borderRadius: 2,
                  cursor: "pointer",
                  transition: "background 0.18s, box-shadow 0.18s, transform 0.12s",
                  "&:hover": {
                    bgcolor: hasClip
                      ? "rgba(51,191,255,0.18)"
                      : "rgba(51,191,255,0.11)",
                    boxShadow: "0 0 0 1.5px #33bfff55",
                    transform: "translateX(2px)",
                  },
                  "&:active": { transform: "translateX(1px)" },
                }}
              >
                <CardContent sx={{ py: 1, px: 1.5, "&:last-child": { pb: 1 } }}>
                  {/* Top row: type + badges */}
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      mb: 0.5,
                    }}
                  >
                    <Typography
                      variant="body2"
                      sx={{ color: "#33bfff", fontWeight: 700, fontSize: "0.85rem" }}
                    >
                      <TerminalIcon sx={{ fontSize: 14, verticalAlign: "middle", mr: 0.5 }} />
                      {type || "Log entry"}
                    </Typography>
                    <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
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
                      {hasClip ? (
                        <PlayCircleOutlineIcon
                          sx={{ fontSize: 16, color: "#33bfff", opacity: 0.85 }}
                          titleAccess="Click to play clip"
                        />
                      ) : (
                        <PlayCircleOutlineIcon
                          sx={{ fontSize: 16, color: "#7f93b2", opacity: 0.4 }}
                          titleAccess="No clip saved"
                        />
                      )}
                    </Box>
                  </Box>

                  {/* Clip window */}
                  {window && (
                    <Typography
                      variant="caption"
                      sx={{ color: "#33bfff", fontFamily: "monospace", display: "block" }}
                    >
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

      <ClipModal
        open={Boolean(selectedEntry)}
        onClose={handleClose}
        clipFile={selectedEntry?.clip_file ?? null}
        logEntry={selectedEntry}
      />
    </>
  );
}

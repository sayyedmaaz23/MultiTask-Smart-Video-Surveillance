import React from "react";
import { Box, List, ListItem, ListItemText, ListItemAvatar, Avatar, Chip } from "@mui/material";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import LocalFireDepartmentIcon from "@mui/icons-material/LocalFireDepartment";
import SmokeFreeIcon from "@mui/icons-material/SmokeFree";
import DirectionsCarIcon from "@mui/icons-material/DirectionsCar";
import GroupIcon from "@mui/icons-material/Group";
import GavelIcon from "@mui/icons-material/Gavel";

// Maps every label the backend can send to a colour + icon.
// Keys match config.LABELS + YOLO-only labels exactly.
const TYPE_DEF = {
  "Smoking":         { color: "#ffac30", icon: <SmokeFreeIcon /> },
  "Group of People": { color: "#ff8626", icon: <GroupIcon /> },
  "Weapons":         { color: "#ff3333", icon: <GavelIcon /> },
  "Abuse":           { color: "#ff3333", icon: <WarningAmberIcon /> },
  "Arrest":          { color: "#c084fc", icon: <WarningAmberIcon /> },
  "Arson":           { color: "#ff8626", icon: <LocalFireDepartmentIcon /> },
  "Assault":         { color: "#ff3333", icon: <WarningAmberIcon /> },
  "Burglary":        { color: "#ff3333", icon: <WarningAmberIcon /> },
  "Explosion":       { color: "#ff8626", icon: <LocalFireDepartmentIcon /> },
  "Fighting":        { color: "#ff3333", icon: <WarningAmberIcon /> },
  "Normal Videos":   { color: "#33bfff", icon: <WarningAmberIcon /> },
  "RoadAccidents":   { color: "#ffac30", icon: <DirectionsCarIcon /> },
  "Robbery":         { color: "#ff3333", icon: <WarningAmberIcon /> },
  "Shooting":        { color: "#ff3333", icon: <WarningAmberIcon /> },
  "Shoplifting":     { color: "#ffac30", icon: <WarningAmberIcon /> },
  "Stealing":        { color: "#ffac30", icon: <WarningAmberIcon /> },
  "Vandalism":       { color: "#ff8626", icon: <WarningAmberIcon /> },
};
const FALLBACK = { color: "#33bfff", icon: <WarningAmberIcon /> };

export default function AlertsPanel({ alerts = [] }) {
  return (
    <Box sx={{ maxHeight: 380, overflow: "auto", mt: 2, ml: 1, mr: 1 }}>
      {alerts.length === 0 && (
        <Box sx={{ color: "#888", textAlign: "center", py: 3 }}>
          <strong>No alerts yet</strong>
        </Box>
      )}
      <List disablePadding>
        {alerts.map((a, idx) => {
          const def       = TYPE_DEF[a.type] || FALLBACK;
          // Format the time window if both ends are present
          const window    = a.clip_start && a.clip_end && a.clip_start !== a.clip_end
            ? `${a.clip_start} → ${a.clip_end}`
            : a.clip_start || null;
          const timeLabel = a.timestamp
            ? new Date(a.timestamp).toLocaleTimeString()
            : null;
          const conf      = a.confidence != null
            ? `${(a.confidence * 100).toFixed(1)}%`
            : null;

          return (
            <ListItem
              key={idx}
              alignItems="flex-start"
              sx={{
                mb: 1,
                borderRadius: 2,
                boxShadow: "0 2px 12px rgba(51,191,255,0.08)",
                border: `1.5px solid ${def.color}`,
                bgcolor: "rgba(50,80,130,0.35)",
                transition: "transform 0.18s",
                "&:hover": { transform: "scale(1.02)" },
              }}
            >
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: def.color, width: 36, height: 36 }}>
                  {def.icon}
                </Avatar>
              </ListItemAvatar>

              <ListItemText
                primary={
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                    <strong style={{ color: def.color }}>
                      {(a.type || "ALERT").toUpperCase()}
                    </strong>
                    {conf && (
                      <Chip
                        label={conf}
                        size="small"
                        sx={{
                          height: 18,
                          fontSize: "0.68rem",
                          bgcolor: "rgba(255,255,255,0.1)",
                          color: def.color,
                          border: `1px solid ${def.color}`,
                        }}
                      />
                    )}
                  </Box>
                }
                secondary={
                  <Box component="span" sx={{ display: "flex", flexDirection: "column", gap: 0.3, mt: 0.3 }}>
                    {a.message && (
                      <span style={{ color: "#e0e0e0", fontSize: "0.85rem" }}>
                        {a.message}
                      </span>
                    )}
                    {window && (
                      <span style={{ color: "#33bfff", fontSize: "0.78rem", fontFamily: "monospace" }}>
                        🕐 {window}
                      </span>
                    )}
                    {timeLabel && (
                      <span style={{ color: "#7f93b2", fontSize: "0.75rem" }}>
                        {timeLabel}
                      </span>
                    )}
                  </Box>
                }
              />
            </ListItem>
          );
        })}
      </List>
    </Box>
  );
}

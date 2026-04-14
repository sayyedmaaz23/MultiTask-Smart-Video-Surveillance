import React, { useEffect, useState, useRef } from "react";
import {
  Paper, Typography, Box, Button, Grid, Divider,
  AppBar, Toolbar, IconButton,
  Dialog, DialogTitle, DialogContent, DialogActions,
  FormGroup, FormControlLabel, Checkbox,
} from "@mui/material";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import SettingsIcon from "@mui/icons-material/Settings";
import VideoFeed from "../components/VideoFeed";
import AlertsPanel from "../components/AlertsPanel";
import LogsTable from "../components/LogsTable";
import createSocket from "../socket";
import { fetchLogs } from "../api";
import { toast } from "react-toastify";

// Must match backend alert_settings keys exactly
const LABELS = [
  "Smoking",
  "Group of People",
  "Weapons",
  "Abuse",
  "Arrest",
  "Arson",
  "Assault",
  "Burglary",
  "Explosion",
  "Fighting",
  "Normal",
  "RoadAccidents",
  "Robbery",
  "Shooting",
  "Shoplifting",
  "Stealing",
  "Vandalism",
];

export default function Dashboard() {
  const [alerts, setAlerts]           = useState([]);
  const [logs,   setLogs]             = useState([]);
  const [openSettings, setOpenSettings] = useState(false);
  const [checkedItems, setCheckedItems] = useState(
    Object.fromEntries(LABELS.map(l => [l, true]))
  );
  const socketRef  = useRef(null);
  const checkedRef = useRef(checkedItems);

  // Keep ref in sync so socket callback always sees latest settings
  useEffect(() => { checkedRef.current = checkedItems; }, [checkedItems]);

  useEffect(() => {
    // Load initial logs
    fetchLogs().then(data => setLogs(data?.logs ?? data ?? []));

    // Open WebSocket — each message is the full alert object from backend
    socketRef.current = createSocket((msg) => {
      // Respect local settings: drop alerts the user has disabled
      if (checkedRef.current[msg.type] === false) return;

      setAlerts(prev => [msg, ...prev].slice(0, 50));

      // Toast shows type + clip window if available
      const window = msg.clip_start && msg.clip_end && msg.clip_start !== msg.clip_end
        ? ` [${msg.clip_start} → ${msg.clip_end}]`
        : "";
      const conf = msg.confidence != null
        ? ` (${(msg.confidence * 100).toFixed(1)}%)`
        : "";
      toast.info(`${msg.type}${window}${conf}`, { autoClose: 4000 });

      // Refresh logs ~800 ms after alert so the new entry is written
      setTimeout(() => fetchLogs().then(d => setLogs(d?.logs ?? d ?? [])), 800);
    });

    return () => socketRef.current?.close();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCheckbox = (label) =>
    setCheckedItems(prev => ({ ...prev, [label]: !prev[label] }));

  const handleSaveSettings = () => {
    fetch("http://localhost:8000/update-settings", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(checkedItems),
    }).catch(e => console.error("Settings save failed:", e));
    setOpenSettings(false);
  };

  return (
    <Box>
      {/* ── Top bar ── */}
      <AppBar position="sticky" color="transparent" elevation={0}
        sx={{ backdropFilter: "blur(8px)" }}>
        <Toolbar>
          <CameraAltIcon sx={{ mr: 2, color: "#33bfff", fontSize: 36 }} />
          <Typography variant="h4" sx={{ flexGrow: 1, fontWeight: 700, color: "#fff" }}>
            Multitask Smart Video Surveillance
          </Typography>
        </Toolbar>
      </AppBar>

      <Box sx={{ pt: 5, px: { xs: 1, md: 0 } }}>
        <Grid container spacing={4} alignItems="stretch">

          {/* ── Left: video feed ── */}
          <Grid item xs={12} md={8} lg={8} sx={{ minWidth: 0, flexGrow: 1 }}>
            <Paper elevation={7} className="glass-panel"
              sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
              <Typography variant="h5" className="section-title" sx={{ pl: 2, pt: 1 }}>
                Live Camera
              </Typography>
              <Divider sx={{ my: 2, bgcolor: "var(--primary)", borderRadius: 2,
                boxShadow: "0 1px 12px #33bfff45" }} />
              <VideoFeed />
              <Box className="button-group">
                <Button variant="contained" color="primary" sx={{ fontWeight: 600 }}
                  onClick={() => window.location.reload()}>
                  Refresh
                </Button>
              </Box>
            </Paper>
          </Grid>

          {/* ── Right: alerts + logs ── */}
          <Grid item xs={12} md={4} lg={4}
            sx={{ display: "flex", flexDirection: "column", gap: 3 }}>

            {/* Alerts panel */}
            <Paper elevation={7} className="glass-panel" sx={{ mb: 2 }}>
              <Box sx={{ display: "flex", alignItems: "center",
                justifyContent: "space-between", pl: 2, pr: 1, pt: 1 }}>
                <Typography variant="h5" className="section-title">
                  Live Alerts
                </Typography>
                <IconButton onClick={() => setOpenSettings(true)}
                  sx={{ color: "#33bfff", transition: "0.3s",
                    "&:hover": { transform: "rotate(90deg)", color: "#fff" } }}>
                  <SettingsIcon />
                </IconButton>
              </Box>
              <Divider sx={{ my: 2, bgcolor: "var(--warn)", borderRadius: 2,
                boxShadow: "0 1px 10px #ffac3050" }} />
              <AlertsPanel alerts={alerts} />
            </Paper>

            {/* Logs panel */}
            <Paper elevation={7} className="glass-panel"
              sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
              <Typography variant="h5" className="section-title" sx={{ pl: 2, pt: 1 }}>
                Recent Logs
              </Typography>
              <Divider sx={{ my: 2, bgcolor: "#36c0f7", borderRadius: 2,
                boxShadow: "0 1px 8px #33bfff33" }} />
              <LogsTable logs={logs} />
            </Paper>
          </Grid>
        </Grid>
      </Box>

      {/* ── Settings dialog ── */}
      <Dialog open={openSettings} onClose={() => setOpenSettings(false)}
        maxWidth="sm" fullWidth
        PaperProps={{ className: "glass-panel",
          sx: { backdropFilter: "blur(12px)", borderRadius: 3 } }}>
        <DialogTitle sx={{ fontWeight: 700, color: "#33bfff" }}>
          Alert Settings
        </DialogTitle>
        <DialogContent dividers>
          <FormGroup>
            {LABELS.map(label => (
              <FormControlLabel
                key={label}
                control={
                  <Checkbox
                    checked={checkedItems[label]}
                    onChange={() => handleCheckbox(label)}
                    sx={{ color: "#33bfff" }}
                  />
                }
                label={<Typography sx={{ color: "#fff" }}>{label}</Typography>}
              />
            ))}
          </FormGroup>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenSettings(false)} sx={{ color: "#7f93b2" }}>
            Cancel
          </Button>
          <Button onClick={handleSaveSettings} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

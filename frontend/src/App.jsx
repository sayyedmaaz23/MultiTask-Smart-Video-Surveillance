import React from "react";
import Dashboard from "./pages/Dashboard";
import { CssBaseline, Container, ThemeProvider, createTheme } from "@mui/material";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary:    { main: "#33bfff" },
    secondary:  { main: "#273850" },
    warning:    { main: "#ffac30" },
    error:      { main: "#ff3333" },
    background: {
      default: "#0f1623",
      paper:   "rgba(22,30,50,0.92)",
    },
    text: {
      primary:   "#f2f4fa",
      secondary: "#7f93b2",
    },
  },
  typography: {
    // Swap Inter/Roboto for something with more character
    fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
    h4: { fontFamily: "'Exo 2', sans-serif", letterSpacing: "0.04em" },
    h5: { fontFamily: "'Exo 2', sans-serif", letterSpacing: "0.03em" },
    h6: { fontFamily: "'Exo 2', sans-serif" },
    body1: { fontFamily: "'IBM Plex Sans', sans-serif" },
    body2: { fontFamily: "'IBM Plex Sans', sans-serif" },
    caption: { fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.72rem" },
  },
  shape: { borderRadius: 10 },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
          backdropFilter: "blur(10px)",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none",
          fontFamily: "'Exo 2', sans-serif",
          fontWeight: 600,
          letterSpacing: "0.03em",
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { fontFamily: "'IBM Plex Mono', monospace" },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: { borderColor: "rgba(51,191,255,0.15)" },
      },
    },
  },
});

export default function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />

      {/* Google Fonts — load Exo 2 + IBM Plex families */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

        body {
          background: radial-gradient(ellipse at 20% 0%, #0d2137 0%, #0f1623 55%, #0a0f1e 100%);
          min-height: 100vh;
        }

        /* Subtle scanline overlay for the security-monitor vibe */
        body::before {
          content: "";
          position: fixed;
          inset: 0;
          background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 3px,
            rgba(0,0,0,0.04) 3px,
            rgba(0,0,0,0.04) 4px
          );
          pointer-events: none;
          z-index: 9999;
        }

        ::-webkit-scrollbar        { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track  { background: #0f1623; }
        ::-webkit-scrollbar-thumb  { background: #33bfff44; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #33bfff99; }
      `}</style>

      <Container maxWidth="lg" sx={{ pt: 4, pb: 6 }}>
        <Dashboard />
      </Container>

      <ToastContainer
        position="bottom-right"
        theme="dark"
        toastStyle={{
          background: "rgba(22,30,50,0.97)",
          border: "1px solid #33bfff44",
          fontFamily: "'IBM Plex Sans', sans-serif",
          fontSize: "0.85rem",
          color: "#f2f4fa",
        }}
      />
    </ThemeProvider>
  );
}

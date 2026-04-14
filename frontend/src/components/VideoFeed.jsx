import React, { useRef, useState } from "react";
import { Box, Button, Typography } from "@mui/material";
import PauseCircleIcon from "@mui/icons-material/PauseCircle";
import PlayCircleIcon from "@mui/icons-material/PlayCircle";
import DownloadIcon from "@mui/icons-material/Download";

const STREAM_URL = "http://localhost:8000/video_feed";

export default function VideoFeed() {
  const imgRef        = useRef(null);
  const [paused, setPaused]     = useState(false);
  const [hasError, setHasError] = useState(false);

  const handlePause = () => {
    if (!imgRef.current) return;
    if (!paused) {
      imgRef.current.dataset.src = imgRef.current.src;
      imgRef.current.src = "";
      setPaused(true);
    } else {
      imgRef.current.src = imgRef.current.dataset.src || STREAM_URL;
      setPaused(false);
      setHasError(false);
    }
  };

  const handleDownload = () => {
    try {
      const img = imgRef.current;
      if (!img || !img.complete || !img.naturalWidth) {
        alert("Video not ready — try again in a moment.");
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width  = img.naturalWidth  || 640;
      canvas.height = img.naturalHeight || 480;
      canvas.getContext("2d").drawImage(img, 0, 0);
      const a    = document.createElement("a");
      a.href     = canvas.toDataURL("image/jpeg");
      a.download = `snapshot_${Date.now()}.jpg`;
      a.click();
    } catch (e) {
      console.error("Snapshot error:", e);
      alert("Unable to capture snapshot. The MJPEG stream may have a CORS restriction.");
    }
  };

  return (
    <Box
      sx={{
        background: "linear-gradient(90deg, #1a233b 0%, #283049 80%)",
        border: "2px solid #33bfff",
        borderRadius: 3,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        p: 2,
        mb: 2,
      }}
    >
      {hasError ? (
        <Box
          sx={{
            width: "100%",
            maxWidth: 500,
            height: 280,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: 2,
            border: "1px dashed #33bfff55",
            mb: 2,
          }}
        >
          <Typography sx={{ color: "#7f93b2", fontSize: "0.9rem" }}>
            Stream unavailable — waiting for camera…
          </Typography>
        </Box>
      ) : (
        <img
          ref={imgRef}
          src={STREAM_URL}
          alt="Live CCTV"
          onError={() => setHasError(true)}
          style={{
            borderRadius: 14,
            width: "100%",
            maxWidth: 500,
            boxShadow: paused ? "0 0 0 4px #ffac3050" : "0 4px 24px #33bfff43",
            opacity: paused ? 0.33 : 1,
            marginBottom: 18,
            background: "#181929",
            display: "block",
          }}
        />
      )}

      <Box sx={{ display: "flex", gap: 2 }}>
        <Button
          variant="contained"
          color={paused ? "warning" : "info"}
          startIcon={paused ? <PlayCircleIcon /> : <PauseCircleIcon />}
          onClick={handlePause}
        >
          {paused ? "Resume" : "Pause"}
        </Button>
        <Button
          variant="contained"
          color="secondary"
          startIcon={<DownloadIcon />}
          onClick={handleDownload}
          disabled={paused || hasError}
        >
          Snapshot
        </Button>
      </Box>
    </Box>
  );
}

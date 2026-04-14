// Simple WebSocket wrapper that reconnects on close.
export default function createSocket(onMessage) {
  let ws;
  let reconnectTimer = null;

  function connect() {
    ws = new WebSocket("ws://localhost:8000/ws/alerts");

    ws.onopen = () => {
      console.log("WS connected");
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        onMessage && onMessage(data);
      } catch (e) {
        console.warn("Invalid WS message", evt.data);
      }
    };

    ws.onclose = () => {
      console.log("WS closed, reconnecting in 2s...");
      reconnectTimer = setTimeout(connect, 2000);
    };

    ws.onerror = (err) => {
      console.error("WS error", err);
      ws.close();
    };
  }

  connect();

  return {
    close: () => ws && ws.close()
  };
}

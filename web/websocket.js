const HEARTBEAT_INTERVAL_MS = 15000;
const STALE_AFTER_MS = 45000;
const RECONNECT_DELAY_MS = 1500;

export class ProgressSocket {
  constructor({ getWsPort, getCurrentAnalysisId, onStatus, onProgress }) {
    this.getWsPort = getWsPort;
    this.getCurrentAnalysisId = getCurrentAnalysisId;
    this.onStatus = onStatus;
    this.onProgress = onProgress;
    this.socket = null;
    this.heartbeatTimer = null;
    this.reconnectTimer = null;
    this.lastSeenAt = 0;
    this.lastPingAt = 0;
    this.supportsPong = false;
    this.subscribedAnalysisId = null;
  }

  start() {
    this.connect();
  }

  stop() {
    this.clearTimers();
    if (this.socket) {
      this.socket.close();
    }
  }

  connect() {
    const wsPort = this.getWsPort();
    if (!wsPort) {
      this.onStatus('WebSocket: missing port', 'error');
      return;
    }

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${proto}://${location.hostname}:${wsPort}`;
    this.onStatus('WebSocket: connecting…', 'connecting');
    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      this.lastSeenAt = Date.now();
      this.lastPingAt = 0;
      this.onStatus(`WebSocket: connected (${wsUrl})`, 'connected');
      this.startHeartbeat();
      const analysisId = this.subscribedAnalysisId || this.getCurrentAnalysisId();
      if (analysisId) {
        this.subscribeToAnalysis(analysisId);
      }
    };

    this.socket.onmessage = (event) => {
      this.lastSeenAt = Date.now();
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }

      if (message.type === 'pong') {
        this.supportsPong = true;
        return;
      }
      if (message.type === 'progress') {
        this.onProgress(message);
      }
    };

    this.socket.onerror = () => {
      this.onStatus('WebSocket: error', 'error');
    };

    this.socket.onclose = () => {
      this.onStatus('WebSocket: disconnected, retrying…', 'error');
      this.clearTimers();
      this.reconnectTimer = setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
    };
  }

  subscribeToAnalysis(analysisId) {
    this.subscribedAnalysisId = analysisId || null;
    if (!analysisId || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    this.socket.send(JSON.stringify({ type: 'subscribe_progress', analysis_id: analysisId }));
  }

  startHeartbeat() {
    this.clearHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (!this.socket) {
        return;
      }
      if (Date.now() - this.lastSeenAt > STALE_AFTER_MS) {
        if (!this.supportsPong) {
          return;
        }
        this.onStatus('WebSocket: stale connection, reconnecting…', 'error');
        this.socket.close();
        return;
      }
      if (this.socket.readyState === WebSocket.OPEN) {
        this.lastPingAt = Date.now();
        this.socket.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
      }
    }, HEARTBEAT_INTERVAL_MS);
  }

  clearHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  clearTimers() {
    this.clearHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
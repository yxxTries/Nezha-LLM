// ===========================
// Configuration (edit this)
// ===========================
const TEXT_ENDPOINT = "http://localhost:8000/api/llm";       // POST JSON { message: string }
const AUDIO_ENDPOINT = "http://localhost:8000/api/asr-llm";  // POST FormData { audio: file }

// Expect responses roughly like:
//   { text: "LLM reply", timestamp: "2025-12-05T..." }
// If your schema differs, adjust `extractTextAndTimestamp` below.

// ===========================
// Pure helpers (testable)
// ===========================

function formatTimestamp(date) {
  const d = date instanceof Date ? date : new Date(date);
  const hours = String(d.getHours()).padStart(2, "0");
  const mins = String(d.getMinutes()).padStart(2, "0");
  const secs = String(d.getSeconds()).padStart(2, "0");
  return `${hours}:${mins}:${secs}`;
}

function extractTextAndTimestamp(responseJson) {
  if (!responseJson) {
    return { text: "", timestamp: new Date() };
  }
  const text =
    responseJson.text ||
    responseJson.output ||
    responseJson.message ||
    "";
  const ts =
    responseJson.timestamp ||
    responseJson.time ||
    new Date().toISOString();
  return { text, timestamp: ts };
}

// Make helpers accessible to tests (without bundler)
window.__LLM_UI__ = {
  formatTimestamp,
  extractTextAndTimestamp,
};

// ===========================
// DOM / application logic
// ===========================

const chatContainer = document.getElementById("chat-container");
const textForm = document.getElementById("text-form");
const textInput = document.getElementById("text-input");
const recordBtn = document.getElementById("record-btn");
const recordStatus = document.getElementById("record-status");
const errorBox = document.getElementById("error-box");

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// Append a message bubble to the chat
function appendMessage(role, text, ts) {
  const timestampStr = formatTimestamp(ts || new Date());

  const wrapper = document.createElement("div");
  wrapper.classList.add("message", role === "user" ? "user" : "assistant");

  const header = document.createElement("div");
  header.classList.add("message-header");

  const roleSpan = document.createElement("span");
  roleSpan.classList.add("message-role");
  roleSpan.textContent = role === "user" ? "You" : "LLM";

  const timeSpan = document.createElement("span");
  timeSpan.classList.add("message-timestamp");
  timeSpan.textContent = timestampStr;

  header.appendChild(roleSpan);
  header.appendChild(timeSpan);

  const body = document.createElement("div");
  body.classList.add("message-text");
  body.textContent = text;

  wrapper.appendChild(header);
  wrapper.appendChild(body);

  chatContainer.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // Return the wrapper so callers can update it later (e.g., replacing placeholders)
  return wrapper;
}

// Error handling
function setError(msg) {
  errorBox.textContent = msg || "";
}

// ===========================
// Text message handling
// ===========================

async function handleTextSubmit(event) {
  event.preventDefault();
  setError("");

  const message = textInput.value.trim();
  if (!message) return;

  appendMessage("user", message, new Date());
  let placeholder = appendMessage("assistant", "[Thinking…]", new Date());
  textInput.value = "";
  textInput.disabled = true;

  try {
    const res = await fetch(TEXT_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message }),
    });

    if (!res.ok) {
      throw new Error(`Text API error: ${res.status} ${res.statusText}`);
    }

    const data = await res.json();
    const { text, timestamp } = extractTextAndTimestamp(data);
    if (placeholder) {
      const body = placeholder.querySelector(".message-text");
      if (body) body.textContent = text || "[empty response]";
      const timeSpan = placeholder.querySelector(".message-timestamp");
      if (timeSpan) timeSpan.textContent = formatTimestamp(timestamp || new Date());
      placeholder = null;
    } else {
      appendMessage("assistant", text || "[empty response]", timestamp);
    }
  } catch (err) {
    console.error(err);
    setError(err.message || "Failed to send text message.");
    if (placeholder) {
      const body = placeholder.querySelector(".message-text");
      if (body) body.textContent = "[error: " + (err.message || "text failed") + "]";
      placeholder = null;
    }
  } finally {
    textInput.disabled = false;
    textInput.focus();
  }
}

// ===========================
// Audio recording + sending
// ===========================

async function toggleRecording() {
  setError("");

  if (isRecording) {
    // Stop recording
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    isRecording = false;
    recordBtn.textContent = "🎙 Start Recording";
    recordBtn.classList.remove("recording");
    recordStatus.textContent = "Processing audio...";
    return;
  }

  // Start recording
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
      await sendAudioToServer(audioBlob);
      recordStatus.textContent = "Idle";
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.textContent = "■ Stop Recording";
    recordBtn.classList.add("recording");
    recordStatus.textContent = "Recording...";
  } catch (err) {
    console.error(err);
    setError("Microphone access failed or unsupported browser.");
  }
}

async function sendAudioToServer(audioBlob) {
  if (!audioBlob || audioBlob.size === 0) {
    setError("No audio data captured.");
    return;
  }

  const userBubble = appendMessage("user", "[Voice message]", new Date());

  const formData = new FormData();
  // Backend: expect field name "audio" or change as needed
  formData.append("audio", audioBlob, "recording.webm");

  try {
    const res = await fetch(AUDIO_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Audio API error: ${res.status} ${res.statusText}`);
    }

    const data = await res.json();
    const { text, timestamp } = extractTextAndTimestamp(data);

    // If backend returns transcript, update the original user bubble
    if (data && data.transcript && userBubble) {
      const body = userBubble.querySelector(".message-text");
      if (body) body.textContent = `${data.transcript || "[no transcript]"} (transcribed from audio)`;
      const timeSpan = userBubble.querySelector(".message-timestamp");
      if (timeSpan) timeSpan.textContent = formatTimestamp(timestamp || new Date());
    }

    appendMessage("assistant", text || "[empty response]", timestamp);
  } catch (err) {
    console.error(err);
    setError(err.message || "Failed to send audio message.");
    if (userBubble) {
      const body = userBubble.querySelector(".message-text");
      if (body) body.textContent = "[error: " + (err.message || "audio failed") + "]";
    }
  }
}

// ===========================
// Event listeners
// ===========================
if (textForm) {
  textForm.addEventListener("submit", handleTextSubmit);
}

if (recordBtn) {
  recordBtn.addEventListener("click", (e) => {
    e.preventDefault();
    toggleRecording();
  });
}

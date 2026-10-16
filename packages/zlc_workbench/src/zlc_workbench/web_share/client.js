"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("screen"), ctx = canvas.getContext("2d", {alpha: false});
let socket, client, controller, selected = "", windows = [], frame = null;
let incoming = Promise.resolve(), header = null, fitted = true, connected = false, controlEnabled = false;
let move = null, moveRequest = 0, pointer = null, lastClick = null, composing = false;
let wheel = null, scrolling = false, scrollOrigin = null;
const touches = new Map();
const keys = new Map();
const modifiers = (e) => ({ctrl: e.ctrlKey, shift: e.shiftKey, alt: e.altKey, meta: e.metaKey});
const ownsControl = () => connected && controlEnabled && client === controller;
const canInput = () => ownsControl() && frame?.window === selected && windows.some(w => w.id === selected && w.visible && !w.blocked);
function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}
function error(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function input(event, fields = {}) {
  if (canInput()) send({type: "input", window: selected, epoch: frame.epoch, event, ...fields});
}
function flushMove() {
  cancelAnimationFrame(moveRequest);
  moveRequest = 0;
  if (move) input("move", move);
  if (wheel) input("wheel", wheel);
  move = null;
  wheel = null;
}
function release() {
  flushMove();
  if (ownsControl() && frame) send({type: "input", window: frame.window, epoch: frame.epoch, event: "release"});
  const captured = [pointer?.id, ...touches.keys()];
  pointer = null; lastClick = null; keys.clear();
  touches.clear(); scrolling = false; scrollOrigin = null;
  for (const id of captured) if (id != null && canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
}
function resize() {
  if (!frame) return;
  const box = $("viewport"), padding = parseFloat(getComputedStyle(box).paddingLeft) * 2;
  const scale = fitted ? Math.min(1, (box.clientWidth - padding) / frame.width, (box.clientHeight - padding) / frame.height) : 1;
  canvas.style.width = `${Math.max(1, frame.width * scale)}px`;
  canvas.style.height = `${Math.max(1, frame.height * scale)}px`;
}
function refreshControls() {
  const current = windows.find(w => w.id === selected), mine = ownsControl();
  $("control").disabled = !connected || !controlEnabled || (controller != null && !mine);
  $("control").textContent = mine ? "Release control" : "Take control";
  $("ownership").textContent = !controlEnabled ? "Viewing only on this server" : mine ? "You have control" : controller != null ? "Another client has control · Read-only" : "Read-only";
  $("keyboard").disabled = !mine;
  $("text-input").disabled = !mine;
  document.querySelectorAll("[data-key]").forEach(button => button.disabled = !mine);
  $("show-window").hidden = !current || current.visible || !mine;
  const ready = connected && current?.visible && frame?.window === selected;
  canvas.hidden = !ready;
  $("placeholder").hidden = ready;
  $("placeholder-text").textContent = !connected ? "Only the experiment application is shared, not the desktop. Connecting does not start or stop an experiment." : !current ? "No experiment windows are available to share." : !current.visible ? "This window is hidden on the experiment computer. Take control to show it." : "Receiving the window view…";
  $("hint").textContent = current?.blocked ? "This window is waiting for an experiment dialog. Complete the dialog first." : current?.modal ? "Experiment dialog. Confirm or cancel to return to the previous window." : mine ? "Tap or drag with one finger; scroll with two. Select a field before typing. Disconnecting does not stop the experiment." : "Read-only viewing does not change the experiment computer's windows or experiment state.";
}
function selectWindow(id) {
  if (windows.find(w => w.id === id)?.blocked) id = windows.find(w => w.modal && w.visible && !w.blocked)?.id || id;
  if (id === selected) return;
  release(); selected = id; frame = null;
  $("windows").value = id;
  if (id) send({type: "select", window: id});
  refreshControls();
}
function updateState(message) {
  const prior = windows.find(w => w.id === selected), oldModals = new Set(windows.filter(w => w.modal && w.visible).map(w => w.id));
  if (controller === client && (message.controller !== client || message.control_enabled === false)) release();
  if ("control_enabled" in message) controlEnabled = message.control_enabled === true;
  controller = message.controller; windows = message.windows;
  const signature = JSON.stringify(windows.map(w => [w.id, w.title, w.visible, w.blocked]));
  if ($("windows").dataset.signature !== signature) {
    $("windows").replaceChildren(...windows.map(w => new Option((w.title || (w.modal ? "Experiment dialog" : "Experiment window")) + (w.blocked ? " · Waiting for dialog" : w.visible ? "" : " · Hidden"), w.id)));
    $("windows").dataset.signature = signature;
  }
  let next = selected;
  if (!windows.some(w => w.id === next) || (prior?.modal && !windows.find(w => w.id === next)?.visible)) next = windows.find(w => w.id === prior?.parent)?.id || windows.find(w => w.visible)?.id || windows[0]?.id || "";
  const blocked = windows.find(w => w.id === next)?.blocked;
  const modal = windows.find(w => w.modal && w.visible && !w.blocked && (blocked || (!oldModals.has(w.id) && (!w.parent || w.parent === next))));
  if (modal) next = modal.id;
  selectWindow(next); $("windows").value = selected; refreshControls();
}
async function receive(data, ws) {
  if (socket !== ws) return;
  if (typeof data === "string") {
    const message = JSON.parse(data);
    if (message.type === "hello") {
      client = message.client; controller = message.controller; connected = true; controlEnabled = message.control_enabled === true;
      $("connection-form").hidden = true; $("toolbar").hidden = false;
      $("connection").textContent = "Connected"; $("connection").dataset.state = "connected";
      error(); refreshControls();
    } else if (message.type === "state") updateState(message);
    else if (message.type === "frame") header = message;
    else if (message.type === "error" || (message.type === "result" && message.ok === false)) error(message.message || message.error || "The operation did not complete.");
    return;
  }
  const patch = header; header = null;
  if (!patch) throw new Error("Received a window image without its frame metadata. Please reconnect.");
  try {
    if (patch.window !== selected) return;
    const bitmap = await createImageBitmap(data);
    try {
      if (socket !== ws || patch.window !== selected) return;
      if (patch.reset) {
        canvas.width = patch.pixel_width; canvas.height = patch.pixel_height;
        ctx.fillStyle = "white"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      ctx.drawImage(bitmap, patch.x, patch.y);
      const resized = !frame || frame.width !== patch.width || frame.height !== patch.height;
      frame = patch;
      if (resized) resize();
      if (canvas.hidden) refreshControls();
    } finally { bitmap.close(); }
  } finally { if (socket === ws) send({type: "ack", sequence: patch.sequence}); }
}
function connect(token) {
  if (socket) socket.close();
  error(); $("connect").disabled = true;
  $("connection").textContent = "Connecting…"; $("connection").dataset.state = "";
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`);
  socket = ws; incoming = Promise.resolve(); header = null;
  ws.onopen = () => { if (socket === ws) { send({type: "hello", token}); $("token").value = ""; } };
  ws.onmessage = (event) => {
    if (socket !== ws) return;
    incoming = incoming.then(() => receive(event.data, ws)).catch(cause => { if (socket === ws) error(String(cause.message || cause)); ws.close(); });
  };
  ws.onerror = () => error("Cannot connect to the experiment computer. Check that sharing is enabled and the address and local network connection are correct.");
  ws.onclose = () => {
    if (socket !== ws) return;
    release(); connected = false; client = null; controller = null; frame = null; selected = ""; windows = [];
    $("connect").disabled = false; $("connection-form").hidden = false; $("toolbar").hidden = true;
    $("text-tools").hidden = true; $("keyboard").setAttribute("aria-expanded", "false");
    $("connection").textContent = "Disconnected"; $("connection").dataset.state = "error"; refreshControls();
  };
}
$("connection-form").onsubmit = e => { e.preventDefault(); connect($("token").value.trim()); };
$("disconnect").onclick = () => { release(); socket.close(); };
$("windows").onchange = () => selectWindow($("windows").value);
$("control").onclick = () => { error(); if (ownsControl()) release(); send({type: "control", take: !ownsControl()}); };
$("show-window").onclick = () => send({type: "show", window: selected});
for (const id of ["fit", "actual"]) $(id).onclick = () => {
  fitted = id === "fit";
  $("fit").setAttribute("aria-pressed", String(fitted)); $("actual").setAttribute("aria-pressed", String(!fitted)); resize();
};
new ResizeObserver(resize).observe($("viewport"));
function point(e, location = e) {
  const bounds = canvas.getBoundingClientRect();
  return {x: (location.clientX - bounds.left) * frame.width / bounds.width, y: (location.clientY - bounds.top) * frame.height / bounds.height, button: e.button, buttons: e.buttons, modifiers: modifiers(e)};
}
function touchCenter() {
  if (touches.size < 2) return null;
  const values = [...touches.values()];
  return {clientX: values.reduce((sum, p) => sum + p.clientX, 0) / values.length, clientY: values.reduce((sum, p) => sum + p.clientY, 0) / values.length};
}
canvas.onpointerdown = e => {
  if (!canInput()) return;
  if (e.pointerType === "touch") {
    touches.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    canvas.setPointerCapture(e.pointerId);
    if (touches.size > 1 || scrolling) {
      e.preventDefault(); flushMove();
      if (!scrolling) { input("release"); pointer = null; lastClick = null; keys.clear(); }
      scrolling = true; scrollOrigin = touchCenter(); return;
    }
  }
  if (pointer && e.pointerId !== pointer.id) return;
  e.preventDefault(); canvas.focus({preventScroll: true}); flushMove();
  const p = point(e), now = performance.now();
  const twice = lastClick && now - lastClick.time < 400 && e.button === lastClick.button && Math.hypot(e.clientX - lastClick.x, e.clientY - lastClick.y) < 5;
  pointer = {id: e.pointerId, x: e.clientX, y: e.clientY, moved: false, double: twice};
  canvas.setPointerCapture(e.pointerId);
  input(twice ? "double" : "down", p); if (twice) lastClick = null;
};
canvas.onpointermove = e => {
  if (touches.has(e.pointerId)) {
    touches.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    if (scrolling) {
      const center = touchCenter();
      if (canInput() && center && scrollOrigin) {
        wheel = {...point(e, center), buttons: 0, deltaMode: 0, deltaX: (wheel?.deltaX || 0) + scrollOrigin.clientX - center.clientX, deltaY: (wheel?.deltaY || 0) + scrollOrigin.clientY - center.clientY};
        if (!moveRequest) moveRequest = requestAnimationFrame(flushMove);
      }
      scrollOrigin = center; return;
    }
  }
  if (!canInput() || (pointer && e.pointerId !== pointer.id)) return;
  const p = point(e);
  if (pointer && Math.hypot(e.clientX - pointer.x, e.clientY - pointer.y) > 2) pointer.moved = true;
  move = p; if (!moveRequest) moveRequest = requestAnimationFrame(flushMove);
};
canvas.onpointerup = e => {
  if (touches.delete(e.pointerId) && scrolling) {
    e.preventDefault(); flushMove(); scrollOrigin = touchCenter();
    if (!touches.size) scrolling = false;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    return;
  }
  if (!pointer || e.pointerId !== pointer.id || !canInput()) return;
  e.preventDefault(); flushMove(); const p = point(e); input("up", p);
  lastClick = pointer.moved || pointer.double ? null : {x: e.clientX, y: e.clientY, button: e.button, time: performance.now()}; pointer = null;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
};
canvas.onpointercancel = release;
canvas.onlostpointercapture = e => { if (pointer || touches.has(e.pointerId)) release(); };
canvas.oncontextmenu = e => e.preventDefault();
canvas.ondblclick = e => e.preventDefault();
canvas.addEventListener("wheel", e => {
  if (!canInput()) return;
  e.preventDefault(); flushMove(); input("wheel", {...point(e), deltaX: e.deltaX, deltaY: e.deltaY, deltaMode: e.deltaMode});
}, {passive: false});
canvas.onkeydown = e => {
  if (!canInput() || e.isComposing || e.key === "Process" || e.key === "Dead") return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") return;
  e.preventDefault(); flushMove(); keys.set(e.code, e.key);
  input("key_down", {key: e.key, code: e.code, modifiers: modifiers(e), repeat: e.repeat});
};
canvas.onkeyup = e => {
  if (!keys.has(e.code)) return;
  e.preventDefault(); keys.delete(e.code); input("key_up", {key: e.key, code: e.code, modifiers: modifiers(e)});
};
canvas.onpaste = e => { if (canInput()) { e.preventDefault(); input("text", {text: e.clipboardData.getData("text/plain")}); } };
canvas.onblur = release;
window.addEventListener("blur", release);
document.addEventListener("visibilitychange", () => { if (document.hidden) release(); });
$("keyboard").onclick = () => {
  const show = $("text-tools").hidden; $("text-tools").hidden = !show;
  $("keyboard").setAttribute("aria-expanded", String(show));
  if (show) $("text-input").focus();
};
function tapKey(key) { input("key_down", {key}); input("key_up", {key}); }
for (const button of document.querySelectorAll("[data-key]")) button.onclick = () => tapKey(button.dataset.key);
function sendText() {
  const edit = $("text-input");
  if (!composing && edit.value) { input("text", {text: edit.value}); edit.value = ""; }
}
$("text-input").oncompositionstart = () => { composing = true; };
$("text-input").oncompositionend = () => { composing = false; queueMicrotask(sendText); };
$("text-input").oninput = e => { if (!e.isComposing) sendText(); };
$("text-input").addEventListener("beforeinput", e => {
  if (composing || e.isComposing) return;
  const key = {deleteContentBackward: "Backspace", deleteContentForward: "Delete", insertLineBreak: "Enter"}[e.inputType];
  if (key) { e.preventDefault(); tapKey(key); }
});
const fragment = location.hash.slice(1);
if (fragment) {
  history.replaceState(null, "", location.pathname + location.search);
  const token = new URLSearchParams(fragment).get("token") || decodeURIComponent(fragment);
  if (token) connect(token);
}

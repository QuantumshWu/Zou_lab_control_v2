"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("screen"), ctx = canvas.getContext("2d", {alpha: false});
let socket, client, controller, selected = "", windows = [], frame = null;
let incoming = Promise.resolve(), header = null, fitted = true, connected = false, controlEnabled = false;
let connectionMessage = "Only the experiment application is shared, not the desktop. Connecting does not start or stop an experiment.";
let immersive = false;
let move = null, moveRequest = 0, pointer = null, composing = false;
let gesture = null;
let editor = null, editing = null;
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
  move = null;
  if (gesture?.updated && !gesture.ended && canInput()) {
    gesture.updated = false;
    const current = touchPair();
    if (!current) return;
    if (!gesture.mode) {
      const translation = Math.hypot(current.clientX - gesture.start.clientX, current.clientY - gesture.start.clientY);
      const separation = Math.abs(current.distance - gesture.start.distance) / 2;
      if (Math.max(translation, separation) < 6) return;
      gesture.mode = separation >= translation ? "pinch" : "wheel";
    }
    const location = point(gesture.event, current);
    if (gesture.mode === "pinch") {
      if (current.distance > 0 && gesture.last.distance > 0) input("pinch", {...location, buttons: 0, scale: current.distance / gesture.last.distance});
    } else {
      const previous = point(gesture.event, gesture.last);
      input("wheel", {...location, buttons: 0, touch: true, deltaMode: 0, deltaX: previous.x - location.x, deltaY: previous.y - location.y});
    }
    gesture.last = current;
  }
}
function release() {
  closeInput();
  cancelAnimationFrame(moveRequest); moveRequest = 0; move = null;
  if (ownsControl() && frame) send({type: "input", window: frame.window, epoch: frame.epoch, event: "cancel"});
  const captured = [pointer?.id, ...touches.keys()];
  pointer = null; keys.clear();
  touches.clear(); gesture = null;
  for (const id of captured) if (id != null && canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
}
function resize() {
  if (!frame) return;
  const box = $("viewport"), padding = parseFloat(getComputedStyle(box).paddingLeft) * 2;
  const cap = immersive && !windows.find(window => window.id === selected)?.modal ? Infinity : 1;
  const scale = fitted ? Math.min(cap, (box.clientWidth - padding) / frame.width, (box.clientHeight - padding) / frame.height) : 1;
  canvas.style.width = `${Math.max(1, frame.width * scale)}px`;
  canvas.style.height = `${Math.max(1, frame.height * scale)}px`;
  positionInput();
}
function refreshControls() {
  const current = windows.find(w => w.id === selected), mine = ownsControl();
  $("control").disabled = !connected || !controlEnabled || (controller != null && !mine);
  $("control").textContent = mine ? "Release control" : "Take control";
  $("ownership").textContent = !controlEnabled ? "Viewing only on this server" : mine ? "You have control" : controller != null ? "Another client has control · Read-only" : "Read-only";
  if (!mine) closeInput();
  $("show-window").hidden = !current || current.visible || !mine;
  const ready = connected && current?.visible && frame?.window === selected;
  canvas.hidden = !ready;
  $("placeholder").hidden = ready;
  $("placeholder-text").textContent = !connected ? connectionMessage : !current ? "No experiment windows are available to share." : !current.visible ? "This window is hidden on the experiment computer. Take control to show it." : "Receiving the window view…";
  $("hint").textContent = current?.blocked ? "This window is waiting for an experiment dialog. Complete the dialog first." : current?.modal ? "Experiment dialog. Confirm or cancel to return to the previous window." : mine ? "Tap a field to type. Drag with one finger; scroll with two, or pinch to zoom a plot. Disconnecting does not stop the experiment." : "Read-only viewing does not change the experiment computer's windows or experiment state.";
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
  const lostControl = controller === client && (message.controller !== client || message.control_enabled === false);
  if ("control_enabled" in message) controlEnabled = message.control_enabled === true;
  controller = message.controller;
  if (lostControl) release();
  windows = message.windows;
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
    if (message.type === "pending") {
      connectionMessage = message.message;
      $("connection").textContent = "Waiting for approval";
      refreshControls();
    } else if (message.type === "hello") {
      client = message.client; controller = message.controller; connected = true; controlEnabled = message.control_enabled === true;
      $("connection-request").hidden = true; $("toolbar").hidden = false;
      $("connection").textContent = "Connected"; $("connection").dataset.state = "connected";
      error(); refreshControls();
    } else if (message.type === "state") updateState(message);
    else if (message.type === "inputs" && frame?.window === message.window && frame.epoch === message.epoch) {
      frame.inputs = message.inputs;
      syncInput();
    }
    else if (message.type === "frame") header = message;
    else if (message.type === "error" || (message.type === "result" && message.ok === false)) error(message.message || message.error || "The operation did not complete.");
    return;
  }
  const patch = header; header = null;
  if (!patch) throw new Error("Received a window image without its frame metadata. Please reconnect.");
  try {
    if (patch.window !== selected) return;
    if (!patch.reset && (!frame || frame.window !== patch.window || frame.epoch !== patch.epoch || frame.sequence !== patch.base_sequence)) {
      if (frame) {
        frame = null;
        send({type: "select", window: selected});
        refreshControls();
      }
      return;
    }
    const baseline = frame;
    const bitmap = await createImageBitmap(data);
    try {
      if (socket !== ws || patch.window !== selected || (!patch.reset && frame !== baseline)) return;
      if (patch.reset) {
        canvas.width = patch.pixel_width; canvas.height = patch.pixel_height;
        ctx.fillStyle = "white"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      ctx.drawImage(bitmap, patch.x, patch.y);
      const resized = !frame || frame.width !== patch.width || frame.height !== patch.height;
      frame = patch;
      syncInput();
      if (resized) resize();
      if (canvas.hidden) refreshControls();
    } finally { bitmap.close(); }
  } finally { if (socket === ws) send({type: "ack", window: patch.window, epoch: patch.epoch, sequence: patch.sequence}); }
}
function connect() {
  if (socket) socket.close();
  error(); $("connect").textContent = "Cancel request";
  connectionMessage = "Connecting to the experiment computer…";
  $("connection").textContent = "Connecting…"; $("connection").dataset.state = "";
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/zlc/ws`);
  socket = ws; incoming = Promise.resolve(); header = null;
  ws.onmessage = (event) => {
    if (socket !== ws) return;
    incoming = incoming.then(() => receive(event.data, ws)).catch(cause => { if (socket === ws) error(String(cause.message || cause)); ws.close(); });
  };
  ws.onerror = () => { if (socket === ws) error("Cannot connect to the experiment computer. Check that sharing is enabled in Device Manager and the address and local network connection are correct."); };
  ws.onclose = event => {
    if (socket !== ws) return;
    release(); connected = false; client = null; controller = null; frame = null; selected = ""; windows = []; controlEnabled = false;
    $("connect").textContent = "Request connection"; $("connection-request").hidden = false; $("toolbar").hidden = true;
    connectionMessage = "Request a connection to ask for approval in Device Manager. Disconnecting does not stop the experiment.";
    if (event.reason) error(event.reason);
    $("connection").textContent = "Disconnected"; $("connection").dataset.state = ""; refreshControls();
  };
  refreshControls();
}
function disconnect() { release(); error(); socket.onerror = null; socket.close(); }
$("connect").onclick = () => {
  if (socket && socket.readyState < WebSocket.CLOSING) disconnect();
  else connect();
};
$("disconnect").onclick = disconnect;
$("windows").onchange = () => selectWindow($("windows").value);
$("control").onclick = () => { error(); if (ownsControl()) release(); send({type: "control", take: !ownsControl()}); };
$("show-window").onclick = () => send({type: "show", window: selected});
for (const id of ["fit", "actual"]) $(id).onclick = () => {
  fitted = id === "fit";
  $("fit").setAttribute("aria-pressed", String(fitted)); $("actual").setAttribute("aria-pressed", String(!fitted)); resize();
};
function fullscreen(active) {
  immersive = active;
  document.body.classList.toggle("immersive", active);
  document.body.classList.remove("show-controls");
  $("fullscreen").textContent = active ? "Exit full screen" : "Full screen";
  $("fullscreen-controls").hidden = !active;
  $("fullscreen-controls").setAttribute("aria-expanded", "false");
  $("fullscreen-controls").setAttribute("aria-label", "Show controls");
  $("fullscreen-controls").title = "Show controls";
  resize();
}
$("fullscreen").onclick = async () => {
  fullscreen(!immersive);
  try {
    if (immersive && document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
    else if (!immersive && document.fullscreenElement) await document.exitFullscreen();
  } catch { /* The immersive viewport also works without browser fullscreen. */ }
};
$("fullscreen-controls").onclick = () => {
  const expanded = document.body.classList.toggle("show-controls");
  const label = expanded ? "Hide controls" : "Show controls";
  $("fullscreen-controls").setAttribute("aria-expanded", String(expanded));
  $("fullscreen-controls").setAttribute("aria-label", label);
  $("fullscreen-controls").title = label;
  resize();
};
document.addEventListener("fullscreenchange", resize);
new ResizeObserver(resize).observe($("viewport"));
function closeInput() {
  const previous = editor, field = editing;
  editor = null; editing = null;
  if (composing && field) input("text", {preedit: ""});
  composing = false;
  if (previous) { previous.value = ""; previous.blur(); previous.hidden = true; }
}
function positionInput() {
  if (!editor || !editing || !frame) return;
  const bounds = canvas.getBoundingClientRect(), [x, y, width, height] = editing.rect;
  Object.assign(editor.style, {left: `${bounds.left+x*bounds.width/frame.width}px`, top: `${bounds.top+y*bounds.height/frame.height}px`,
    width: `${width*bounds.width/frame.width}px`, height: `${height*bounds.height/frame.height}px`});
}
function openInput(target) {
  if (!canInput() || target.readonly) return;
  const next = $(target.multiline ? "direct-multiline" : "direct-input");
  if (editor !== next || editing?.id !== target.id) closeInput();
  editor = next; editing = {...target, confirmed: Boolean(target.focused)};
  if (!target.multiline) editor.type = target.password ? "password" : "text";
  editor.inputMode = target.inputmode; editor.readOnly = target.readonly;
  editor.hidden = false; positionInput(); editor.focus({preventScroll: true});
}
function syncInput() {
  if (!editing) return;
  let target = frame.inputs?.find(item => item.id === editing.id);
  if (editing.confirmed) target = frame.inputs?.find(item => item.focused) || target;
  if (!target || target.readonly || !canInput()) { closeInput(); return; }
  if (target.id !== editing.id) { openInput(target); return; }
  editing = {...target, confirmed: editing.confirmed || target.focused};
  positionInput();
}
function point(e, location = e) {
  const bounds = canvas.getBoundingClientRect();
  return {x: (location.clientX - bounds.left) * frame.width / bounds.width, y: (location.clientY - bounds.top) * frame.height / bounds.height, button: e.button, buttons: e.buttons, modifiers: modifiers(e)};
}
function touchPair() {
  if (touches.size < 2) return null;
  const [first, second] = touches.values();
  return {clientX: (first.clientX + second.clientX) / 2, clientY: (first.clientY + second.clientY) / 2,
    distance: Math.hypot(second.clientX - first.clientX, second.clientY - first.clientY)};
}
canvas.onpointerdown = e => {
  if (!canInput()) return;
  if (e.pointerType === "touch") {
    touches.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    canvas.setPointerCapture(e.pointerId);
    if (touches.size > 1 || gesture) {
      e.preventDefault();
      if (!gesture) {
        closeInput();
        cancelAnimationFrame(moveRequest); moveRequest = 0; move = null;
        input("cancel"); pointer = null; keys.clear();
        const start = touchPair();
        gesture = {start, last: start, mode: null, updated: false, ended: false, event: e};
      }
      return;
    }
  }
  if (pointer && e.pointerId !== pointer.id) return;
  e.preventDefault(); flushMove();
  const p = point(e);
  const target = [...(frame.inputs || [])].reverse().find(item => !item.readonly && p.x >= item.rect[0] && p.y >= item.rect[1] && p.x < item.rect[0]+item.rect[2] && p.y < item.rect[1]+item.rect[3]);
  if (editing?.id !== target?.id) closeInput();
  if (!target) canvas.focus({preventScroll: true});
  pointer = {id: e.pointerId, field: target};
  canvas.setPointerCapture(e.pointerId);
  input("down", p);
};
canvas.onpointermove = e => {
  if (touches.has(e.pointerId)) {
    touches.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    if (gesture) {
      if (!gesture.ended) {
        gesture.event = e; gesture.updated = true;
        if (!moveRequest) moveRequest = requestAnimationFrame(flushMove);
      }
      return;
    }
  }
  if (!canInput() || (pointer && e.pointerId !== pointer.id)) return;
  const p = point(e);
  move = p; if (!moveRequest) moveRequest = requestAnimationFrame(flushMove);
};
canvas.onpointerup = e => {
  if (touches.has(e.pointerId) && gesture) {
    e.preventDefault(); flushMove(); gesture.ended = true;
    touches.delete(e.pointerId);
    if (!touches.size) gesture = null;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    return;
  }
  touches.delete(e.pointerId);
  if (!pointer || e.pointerId !== pointer.id || !canInput()) return;
  e.preventDefault(); flushMove(); const p = point(e); input("up", p);
  const field = pointer.field;
  pointer = null;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  if (field) openInput(field);
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
canvas.onblur = event => { if (event.relatedTarget !== editor) release(); };
window.addEventListener("blur", release);
document.addEventListener("visibilitychange", () => { if (document.hidden) release(); });
function sendText() {
  if (!composing && editor?.value && editing) {
    input("text", {text: editor.value}); editor.value = "";
  }
}
for (const edit of [$("direct-input"), $("direct-multiline")]) {
  edit.onblur = () => { if (editor === edit) release(); };
  edit.addEventListener("compositionstart", () => { composing = true; });
  edit.addEventListener("compositionupdate", event => { if (editing) input("text", {preedit: event.data}); });
  edit.addEventListener("compositionend", () => { composing = false; if (editing) input("text", {preedit: ""}); queueMicrotask(sendText); });
  edit.oninput = event => { if (!event.isComposing) sendText(); };
  edit.onpaste = event => {
    if (!editing) return;
    event.preventDefault(); input("text", {text: event.clipboardData.getData("text/plain")});
  };
  edit.onkeydown = event => {
    if (!editing || event.isComposing || composing || ["Process", "Dead", "Unidentified"].includes(event.key)) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "v") return;
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) return;
    event.preventDefault(); keys.set(event.code, event.key);
    input("key_down", {key: event.key, code: event.code, modifiers: modifiers(event), repeat: event.repeat});
  };
  edit.onkeyup = event => {
    if (!keys.has(event.code) || !editing) return;
    event.preventDefault(); keys.delete(event.code);
    input("key_up", {key: event.key, code: event.code, modifiers: modifiers(event)});
  };
  edit.onbeforeinput = event => {
    if (!editing || composing || event.isComposing) return;
    const key = {deleteContentBackward: "Backspace", deleteContentForward: "Delete", insertLineBreak: "Enter", insertParagraph: "Enter"}[event.inputType];
    if (key) {
      event.preventDefault(); input("key_down", {key}); input("key_up", {key});
    }
  };
}
connect();

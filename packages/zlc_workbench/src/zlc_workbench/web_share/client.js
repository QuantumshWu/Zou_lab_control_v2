"use strict";
const $ = (id) => document.getElementById(id);
const stage = $("screen"), video = $("video"), popups = $("popups");
let socket, client, controller, selected = "", windows = [], frame = null;
let incoming = Promise.resolve(), fitted = true, connected = false, controlEnabled = false;
let peer = null, media = null, mediaReady = false, videoCallback = 0, viewportSignature = "";
let connectionMessage = "Only the experiment application is shared, not the desktop. Connecting does not start or stop an experiment.";
let immersive = false;
let panel = null, holdMs = 0, dragDistance = 0;
let move = null, moveRequest = 0, pointer = null, composing = false;
let gesture = null;
let editor = null, editing = null;
const touches = new Map();
const keys = new Map();
const modifiers = (e) => ({ctrl: e.ctrlKey, shift: e.shiftKey, alt: e.altKey, meta: e.metaKey});
const ownsControl = () => connected && controlEnabled && client === controller;
const canView = () => connected && mediaReady && frame?.window === selected && frame.panel === (panel?.id ?? null) && windows.some(w => w.id === selected && w.visible && !w.blocked);
const canInput = () => ownsControl() && canView();
const viewRect = scene => scene.view || [0, 0, scene.width, scene.height];
function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}
function error(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
  if (message && immersive && !document.body.classList.contains("show-controls")) $("fullscreen-controls").click();
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
      if (gesture.mode === "wheel") input("wheel", {...point(gesture.event, gesture.start),
        buttons: 0, touch: true, phase: "begin", deltaMode: 0, deltaX: 0, deltaY: 0});
    }
    const location = point(gesture.event, current);
    if (gesture.mode === "pinch") {
      if (current.distance > 0 && gesture.last.distance > 0) input("pinch", {...location, buttons: 0, scale: current.distance / gesture.last.distance});
    } else {
      const previous = point(gesture.event, gesture.last);
      input("wheel", {...location, buttons: 0, touch: true, phase: "update", deltaMode: 0, deltaX: previous.x - location.x, deltaY: previous.y - location.y});
    }
    gesture.last = current;
  }
}
function release() {
  cancelHold();
  closeInput();
  cancelAnimationFrame(moveRequest); moveRequest = 0; move = null;
  if (ownsControl() && frame) send({type: "input", window: frame.window, epoch: frame.epoch, event: "cancel"});
  const captured = [pointer?.id, ...touches.keys()];
  pointer = null; keys.clear();
  touches.clear(); gesture = null;
  for (const id of captured) if (id != null && stage.hasPointerCapture(id)) stage.releasePointerCapture(id);
}
function resize(report = true) {
  const scene = media || windows.find(window => window.id === selected);
  if (!scene?.width || !scene.height) return;
  const [, , viewWidth, viewHeight] = media ? viewRect(media) : panel?.rect || viewRect(scene);
  const box = $("viewport"), padding = parseFloat(getComputedStyle(box).paddingLeft) * 2;
  const cap = immersive && !windows.find(window => window.id === selected)?.modal ? Infinity : 1;
  const scale = fitted ? Math.min(cap, (box.clientWidth - padding) / viewWidth, (box.clientHeight - padding) / viewHeight) : 1;
  const width = Math.max(1, viewWidth * scale), height = Math.max(1, viewHeight * scale);
  stage.style.width = `${width}px`; stage.style.height = `${height}px`;
  positionInput();
  if (report && connected && !editing) {
    const pixel_width = Math.max(1, Math.round(width * devicePixelRatio)), pixel_height = Math.max(1, Math.round(height * devicePixelRatio));
    const signature = `${selected}/${panel?.id ?? ""}/${pixel_width}/${pixel_height}`;
    if (signature !== viewportSignature) {
      viewportSignature = signature;
      send({type: "viewport", window: selected, panel: panel?.id ?? null, pixel_width, pixel_height});
    }
  }
}
function clearMedia(closePeer = true) {
  if (closePeer) { const previous = peer; peer = null; previous?.close(); }
  if (videoCallback) video.cancelVideoFrameCallback(videoCallback);
  videoCallback = 0; mediaReady = false; frame = media = null;
  video.srcObject = null; video.load(); popups.replaceChildren();
}
function place(element, rect, scene) {
  const [x, y, width, height] = viewRect(scene);
  Object.assign(element.style, {left: `${100*(rect[0]-x)/width}%`, top: `${100*(rect[1]-y)/height}%`,
    width: `${100*rect[2]/width}%`, height: `${100*rect[3]/height}%`});
}
async function startMedia(scene, ws) {
  if (!window.RTCPeerConnection || !video.requestVideoFrameCallback) throw new Error("This browser does not support the experiment video stream.");
  const current = new RTCPeerConnection({iceServers: []}); peer = current;
  const receiver = current.addTransceiver("video", {direction: "recvonly"});
  const codecs = RTCRtpReceiver.getCapabilities("video").codecs.filter(codec => codec.mimeType === "video/H264");
  if (!codecs.length) throw new Error("This browser does not support H264 experiment video.");
  receiver.setCodecPreferences(codecs);
  const active = () => peer === current && socket === ws && media?.stream === scene.stream;
  current.ontrack = event => {
    if (!active()) return;
    video.srcObject = new MediaStream([event.track]);
    videoCallback = video.requestVideoFrameCallback(() => {
      if (!active()) return;
      videoCallback = 0;
      mediaReady = true; frame = media; syncInput(); refreshControls();
    });
    video.play().catch(cause => { if (active()) error(String(cause.message || cause)); });
  };
  current.onconnectionstatechange = () => {
    if (active() && current.connectionState === "failed") {
      release(); clearMedia(); error("The experiment video connection failed. Reconnect to request a new connection."); refreshControls();
    }
  };
  let offered = false;
  const offer = () => {
    if (!offered && active() && current.iceGatheringState === "complete") {
      offered = true;
      send({type: "offer", stream: scene.stream, sdp: current.localDescription.sdp, description_type: current.localDescription.type});
    }
  };
  current.onicegatheringstatechange = offer;
  await current.setLocalDescription(await current.createOffer());
  offer();
}
async function showMedia(message, ws) {
  if (message.window !== selected || message.panel !== (panel?.id ?? null) || !windows.some(window => window.id === selected && window.visible)) return;
  if (message.stream == null) { release(); clearMedia(); refreshControls(); return; }
  const changed = media?.stream !== message.stream;
  if (changed) { release(); clearMedia(); refreshControls(); }
  const shown = new Map([...popups.children].map(image => [image.dataset.id, image]));
  const images = await Promise.all((message.popups || []).map(async popup => {
    let image = shown.get(String(popup.id));
    if (!image || image.src !== popup.url) {
      image = new Image(); image.src = popup.url; image.alt = ""; image.draggable = false;
      image.dataset.id = String(popup.id); await image.decode();
    }
    return image;
  }));
  if (socket !== ws || message.window !== selected || message.panel !== (panel?.id ?? null) || !windows.some(window => window.id === selected && window.visible)) return;
  images.forEach((image, index) => place(image, message.popups[index].rect, message));
  media = message; popups.replaceChildren(...images); place(video, message.root, message); resize(false);
  if (changed) {
    startMedia(message, ws).catch(cause => {
      if (socket === ws && media?.stream === message.stream) {
        release(); clearMedia(); error(String(cause.message || cause)); refreshControls();
      }
    });
  } else if (mediaReady) { frame = media; syncInput(); }
  refreshControls();
}
function refreshControls() {
  const current = windows.find(w => w.id === selected), mine = ownsControl();
  $("control").disabled = !connected || !controlEnabled || (controller != null && !mine);
  $("control").textContent = mine ? "Release control" : "Take control";
  $("ownership").textContent = !controlEnabled ? "Viewing only on this server" : mine ? "You have control" : controller != null ? "Another client has control · Read-only" : "Read-only";
  if (!mine) closeInput();
  $("show-window").hidden = !current || current.visible || !mine;
  $("panel-back").hidden = panel === null;
  const ready = connected && current?.visible && media?.window === selected;
  stage.hidden = !ready;
  $("placeholder").hidden = ready;
  $("placeholder-text").textContent = !connected ? connectionMessage : !current ? "No experiment windows are available to share." : !current.visible ? "This window is hidden on the experiment computer. Take control to show it." : "Receiving the window view…";
  $("hint").textContent = current?.blocked ? "This window is waiting for an experiment dialog. Complete the dialog first." : current?.modal ? "Experiment dialog. Confirm or cancel to return to the previous window." : mine ? "Tap a field to type. Drag with one finger; scroll with two, or pinch to zoom a plot. Disconnecting does not stop the experiment." : "Read-only viewing does not change the experiment computer's windows or experiment state.";
}
function leavePanel() {
  if (!panel) return;
  const previous = panel; panel = null; fitted = previous.fitted;
  document.body.classList.remove("panel-view");
  $("fit").setAttribute("aria-pressed", String(fitted)); $("actual").setAttribute("aria-pressed", String(!fitted));
  fullscreen(previous.immersive, false);
}
function selectPanel(next, notify = true) {
  release(); clearMedia(false); viewportSignature = "";
  if (next) {
    panel = {id: next.id, rect: next.rect, immersive, fitted}; fitted = true;
    document.body.classList.add("panel-view");
    $("fit").setAttribute("aria-pressed", "true"); $("actual").setAttribute("aria-pressed", "false");
    fullscreen(true, false);
  } else leavePanel();
  if (notify) send({type: "select", window: selected, panel: panel?.id ?? null});
  resize(); refreshControls();
}
function selectWindow(id) {
  if (windows.find(w => w.id === id)?.blocked) id = windows.find(w => w.modal && w.visible && !w.blocked)?.id || id;
  if (id === selected) return;
  release(); leavePanel(); clearMedia(!id); selected = id; viewportSignature = "";
  $("windows").value = id;
  if (id) send({type: "select", window: id, panel: null});
  resize();
  refreshControls();
}
function updateState(message) {
  holdMs = message.hold_ms; dragDistance = message.drag_distance;
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
  selectWindow(next); $("windows").value = selected;
  const current = windows.find(window => window.id === selected);
  if (!current?.visible && panel) selectPanel(null);
  if (!current?.visible && media) { release(); clearMedia(); }
  if (current?.width !== prior?.width || current?.height !== prior?.height) resize();
  refreshControls();
}
async function receive(data, ws) {
  if (socket !== ws) return;
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
  else if (message.type === "media") await showMedia(message, ws);
  else if (message.type === "answer" && peer && media?.stream === message.stream) {
    await peer.setRemoteDescription({type: message.description_type, sdp: message.sdp});
  } else if (message.type === "result" && "return_panel" in message) {
    if (message.window === selected && message.panel === panel?.id) {
      selectPanel(null, false); error(message.message);
    }
  } else if (message.type === "error" || (message.type === "result" && message.ok === false)) {
    if ("stream" in message) {
      if (message.window !== selected || message.stream !== media?.stream) return;
      release(); clearMedia(); refreshControls();
    }
    error(message.message || message.error || "The operation did not complete.");
  }
}
function connect() {
  release(); leavePanel(); clearMedia(); connected = false; selected = ""; windows = []; viewportSignature = "";
  if (socket) socket.close();
  error(); $("connect").textContent = "Cancel request";
  connectionMessage = "Connecting to the experiment computer…";
  $("connection").textContent = "Connecting…"; $("connection").dataset.state = "";
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/zlc/ws`);
  socket = ws; incoming = Promise.resolve();
  ws.onmessage = (event) => {
    if (socket !== ws) return;
    incoming = incoming.then(() => receive(event.data, ws)).catch(cause => { if (socket === ws) error(String(cause.message || cause)); ws.close(); });
  };
  ws.onerror = () => { if (socket === ws) error("Cannot connect to the experiment computer. Check that sharing is enabled in Device Manager and the address and local network connection are correct."); };
  ws.onclose = event => {
    if (socket !== ws) return;
    release(); leavePanel(); clearMedia(); connected = false; client = null; controller = null; selected = ""; windows = []; controlEnabled = false; viewportSignature = "";
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
$("panel-back").onclick = () => selectPanel(null);
for (const id of ["fit", "actual"]) $(id).onclick = () => {
  fitted = id === "fit";
  $("fit").setAttribute("aria-pressed", String(fitted)); $("actual").setAttribute("aria-pressed", String(!fitted)); resize();
};
function fullscreen(active, report = true) {
  immersive = active;
  document.body.classList.toggle("immersive", active);
  document.body.classList.remove("show-controls");
  $("fullscreen").textContent = active ? "Exit full screen" : "Full screen";
  $("fullscreen-controls").hidden = !active;
  $("fullscreen-controls").setAttribute("aria-expanded", "false");
  $("fullscreen-controls").setAttribute("aria-label", "Show controls");
  $("fullscreen-controls").title = "Show controls";
  resize(report);
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
document.addEventListener("fullscreenchange", () => resize());
window.addEventListener("resize", () => resize());
new ResizeObserver(() => resize(false)).observe($("viewport"));
let resolutionQuery;
function watchResolution() {
  resolutionQuery?.removeEventListener("change", watchResolution);
  resolutionQuery = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
  resolutionQuery.addEventListener("change", watchResolution);
  resize();
}
watchResolution();
function closeInput() {
  const previous = editor, field = editing;
  editor = null; editing = null;
  if (composing && field) input("text", {preedit: ""});
  composing = false;
  if (previous) { previous.value = ""; previous.blur(); previous.hidden = true; }
  if (previous) resize();
}
function positionInput() {
  if (!editor || !editing || !frame) return;
  const bounds = stage.getBoundingClientRect(), [x, y, width, height] = editing.rect;
  const [vx, vy, vw, vh] = viewRect(frame);
  Object.assign(editor.style, {left: `${bounds.left+(x-vx)*bounds.width/vw}px`, top: `${bounds.top+(y-vy)*bounds.height/vh}px`,
    width: `${width*bounds.width/vw}px`, height: `${height*bounds.height/vh}px`});
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
  const bounds = stage.getBoundingClientRect();
  const [x, y, width, height] = viewRect(frame);
  return {x: x + (location.clientX - bounds.left) * width / bounds.width, y: y + (location.clientY - bounds.top) * height / bounds.height,
    button: e.button, buttons: e.buttons, pointerType: e.pointerType || "mouse", timestamp: e.timeStamp, modifiers: modifiers(e)};
}
function touchPair() {
  if (touches.size < 2) return null;
  const [first, second] = touches.values();
  return {clientX: (first.clientX + second.clientX) / 2, clientY: (first.clientY + second.clientY) / 2,
    distance: Math.hypot(second.clientX - first.clientX, second.clientY - first.clientY)};
}
function cancelHold(deliver = false) {
  if (!pointer?.pending) return;
  clearTimeout(pointer.holdTimer);
  const pending = pointer.pending; pointer.pending = null;
  if (deliver) input("down", pending);
}
stage.onpointerdown = e => {
  if (!canView()) return;
  const p = point(e);
  const contains = rect => p.x >= rect[0] && p.y >= rect[1] && p.x < rect[0]+rect[2] && p.y < rect[1]+rect[3];
  const candidate = e.pointerType === "touch" && !panel && holdMs > 0 && !frame.popups.some(item => contains(item.rect))
    ? frame.panels.find(item => item.hit_rects.some(contains)) : null;
  if (!canInput() && !candidate && !touches.size) return;
  if (e.pointerType === "touch") {
    touches.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    stage.setPointerCapture(e.pointerId);
    if (touches.size > 1 || gesture) {
      e.preventDefault();
      if (!gesture) {
        cancelHold();
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
  const target = [...(frame.inputs || [])].reverse().find(item => !item.readonly && p.x >= item.rect[0] && p.y >= item.rect[1] && p.x < item.rect[0]+item.rect[2] && p.y < item.rect[1]+item.rect[3]);
  if (editing?.id !== target?.id) closeInput();
  if (!target) stage.focus({preventScroll: true});
  pointer = {id: e.pointerId, field: target};
  stage.setPointerCapture(e.pointerId);
  if (candidate && !target) {
    const waiting = pointer; waiting.pending = p;
    waiting.holdTimer = setTimeout(() => {
      if (pointer !== waiting || !waiting.pending) return;
      const current = frame?.panels.find(item => item.id === candidate.id);
      if (!canView()) { release(); return; }
      if (!current?.available) {
        release(); error(current?.reason || "The plot is no longer visible. Return it fully into view first."); return;
      }
      selectPanel(current);
    }, holdMs);
  } else input("down", p);
};
stage.onpointermove = e => {
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
  if (!canView() || (pointer && e.pointerId !== pointer.id)) return;
  const p = point(e);
  if (pointer?.pending) {
    if (Math.abs(p.x-pointer.pending.x)+Math.abs(p.y-pointer.pending.y) < dragDistance) return;
    cancelHold(true);
  }
  if (!canInput()) return;
  move = p; if (!moveRequest) moveRequest = requestAnimationFrame(flushMove);
};
stage.onpointerup = e => {
  if (touches.has(e.pointerId) && gesture) {
    e.preventDefault(); flushMove();
    if (!gesture.ended && gesture.mode === "wheel") input("wheel", {...point(gesture.event, gesture.last),
      buttons: 0, touch: true, phase: "end", deltaMode: 0, deltaX: 0, deltaY: 0});
    gesture.ended = true;
    touches.delete(e.pointerId);
    if (!touches.size) gesture = null;
    if (stage.hasPointerCapture(e.pointerId)) stage.releasePointerCapture(e.pointerId);
    return;
  }
  touches.delete(e.pointerId);
  if (!pointer || e.pointerId !== pointer.id) return;
  e.preventDefault(); cancelHold(canInput()); flushMove();
  if (canInput()) input("up", point(e));
  const field = pointer.field;
  pointer = null;
  if (stage.hasPointerCapture(e.pointerId)) stage.releasePointerCapture(e.pointerId);
  if (field && canInput()) openInput(field);
};
stage.onpointercancel = release;
stage.onlostpointercapture = e => { if (pointer || touches.has(e.pointerId)) release(); };
stage.oncontextmenu = e => e.preventDefault();
stage.ondblclick = e => e.preventDefault();
stage.addEventListener("wheel", e => {
  if (!canInput()) return;
  e.preventDefault(); flushMove(); input("wheel", {...point(e), deltaX: e.deltaX, deltaY: e.deltaY, deltaMode: e.deltaMode});
}, {passive: false});
stage.onkeydown = e => {
  if (!canInput() || e.isComposing || e.key === "Process" || e.key === "Dead") return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v") return;
  e.preventDefault(); flushMove(); keys.set(e.code, e.key);
  input("key_down", {key: e.key, code: e.code, modifiers: modifiers(e), repeat: e.repeat});
};
stage.onkeyup = e => {
  if (!keys.has(e.code)) return;
  e.preventDefault(); keys.delete(e.code); input("key_up", {key: e.key, code: e.code, modifiers: modifiers(e)});
};
stage.onpaste = e => { if (canInput()) { e.preventDefault(); input("text", {text: e.clipboardData.getData("text/plain")}); } };
stage.onblur = event => { if (event.relatedTarget !== editor) release(); };
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

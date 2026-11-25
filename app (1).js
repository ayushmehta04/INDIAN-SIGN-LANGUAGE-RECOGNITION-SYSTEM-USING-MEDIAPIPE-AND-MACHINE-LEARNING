const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), stage = document.querySelector(".stage");
const octx = overlay.getContext("2d");
const grab = document.createElement("canvas");

const INTERVAL_MS = 250;      // how often a frame is sent to the backend
const STABLE_FRAMES = 4;      // same letter this many times in a row -> add to text
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],
];

let stream = null, timer = null, busy = false;
let streak = { label: null, n: 0 }, armed = true;
let text = "";

// ---------- status ----------
fetch("/health").then(r => r.json()).then(h => {
  const el = $("status");
  if (h.model_ready) {
    el.textContent = `${h.model} · ${h.accuracy}% test accuracy`;
    el.classList.add("ok");
  } else {
    el.textContent = "Model not trained yet — run train.py";
    el.classList.add("bad");
  }
}).catch(() => { $("status").textContent = "Backend not reachable"; $("status").classList.add("bad"); });

// ---------- camera ----------
$("startBtn").onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
  } catch (e) {
    alert("Could not open the camera: " + e.message + "\nOpen the app at http://127.0.0.1:5000 and allow camera access.");
    return;
  }
  video.srcObject = stream;
  await video.play();
  stage.classList.add("on");
  overlay.classList.add("mirror");
  $("startBtn").disabled = true; $("stopBtn").disabled = false;
  timer = setInterval(tick, INTERVAL_MS);
};

$("stopBtn").onclick = () => {
  clearInterval(timer); timer = null;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; video.srcObject = null;
  stage.classList.remove("on");
  octx.clearRect(0, 0, overlay.width, overlay.height);
  $("startBtn").disabled = false; $("stopBtn").disabled = true;
  showResult(null);
};

async function tick() {
  if (busy || !stream || video.videoWidth === 0) return;
  busy = true;
  try {
    grab.width = video.videoWidth; grab.height = video.videoHeight;
    grab.getContext("2d").drawImage(video, 0, 0);
    const res = await predict(grab.toDataURL("image/jpeg", 0.7));
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    drawHands(res.landmarks || [], overlay.width, overlay.height);
    handle(res, true);
  } catch (e) { console.error(e); }
  busy = false;
}

async function predict(dataUrl) {
  const r = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, flip: $("flipBox").checked }),
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "Prediction failed");
  return j;
}

// ---------- upload ----------
$("fileInput").onchange = (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    const img = new Image();
    img.onload = async () => {
      overlay.classList.remove("mirror");
      overlay.width = img.width; overlay.height = img.height;
      octx.drawImage(img, 0, 0);
      try {
        const res = await predict(reader.result);
        drawHands(res.landmarks || [], img.width, img.height);
        handle(res, false);
      } catch (e) { alert(e.message); }
    };
    img.src = reader.result;
    stage.classList.add("on");
  };
  reader.readAsDataURL(file);
  ev.target.value = "";
};

// ---------- drawing ----------
function drawHands(hands, w, h) {
  if (!video.srcObject) { /* uploaded photo: keep the photo that was drawn */ } else octx.clearRect(0, 0, w, h);
  octx.lineWidth = Math.max(2, w / 220);
  hands.forEach((pts, i) => {
    const color = i === 0 ? "#4fd1c5" : "#f6ad55";
    octx.strokeStyle = color; octx.fillStyle = color;
    CONNECTIONS.forEach(([a, b]) => {
      octx.beginPath();
      octx.moveTo(pts[a][0] * w, pts[a][1] * h);
      octx.lineTo(pts[b][0] * w, pts[b][1] * h);
      octx.stroke();
    });
    pts.forEach(p => { octx.beginPath(); octx.arc(p[0] * w, p[1] * h, octx.lineWidth * 1.4, 0, 7); octx.fill(); });
  });
}

// ---------- result + text builder ----------
function showResult(res) {
  const bar = $("bar");
  if (!res || !res.label) {
    $("letter").textContent = "–";
    bar.style.width = "0";
    $("confText").textContent = res ? "No hand found" : "Show a sign to the camera";
    $("topList").innerHTML = "";
    return;
  }
  const pct = Math.round(res.confidence * 100);
  $("letter").textContent = res.label;
  bar.style.width = pct + "%";
  bar.classList.toggle("low", pct < +$("threshold").value);
  $("confText").textContent = `${pct}% confident · ${res.hands} hand${res.hands > 1 ? "s" : ""}`;
  $("topList").innerHTML = res.top.map(t => `<li>${t.label} ${Math.round(t.confidence * 100)}%</li>`).join("");
}

function handle(res, live) {
  showResult(res);
  if (!live) return;
  const th = +$("threshold").value / 100;
  if (!res.label) { armed = true; streak = { label: null, n: 0 }; return; }   // hand removed
  if (res.confidence < th) { streak = { label: null, n: 0 }; return; }
  streak = streak.label === res.label ? { label: res.label, n: streak.n + 1 } : { label: res.label, n: 1 };
  if (streak.n >= STABLE_FRAMES && armed) {
    addChar(res.label);
    armed = false;                       // wait until the hand leaves or the sign changes
  }
  if (streak.n === 1) armed = armed || streak.label !== lastAdded;
}

let lastAdded = null;
function addChar(c) { text += c; lastAdded = c; renderText(); }
function renderText() { $("sentence").textContent = text; }

$("threshold").oninput = (e) => { $("thVal").textContent = e.target.value; };
$("spaceBtn").onclick = () => { text += " "; lastAdded = null; renderText(); };
$("backBtn").onclick = () => { text = text.slice(0, -1); renderText(); };
$("clearBtn").onclick = () => { text = ""; lastAdded = null; renderText(); };
$("speakBtn").onclick = () => {
  if (!text.trim() || !("speechSynthesis" in window)) return;
  speechSynthesis.speak(new SpeechSynthesisUtterance(text.toLowerCase()));
};

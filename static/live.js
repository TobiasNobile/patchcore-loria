"use strict";

const $ = (id) => document.getElementById(id);
const form = $("params");
const fields = [...form.querySelectorAll("input,select")];

// Course du curseur alpha : exposant = MAX * s². Quadratique pour placer la
// diagonale (n^1) pile au milieu, s=0 donnant la heatmap pleine (n^0).
let ALPHA_MAX = 4, ALPHA_DEFAULT = 2;
let BANKS = {};

const alphaExp = () => ALPHA_MAX * Math.pow(parseFloat($("alpha").value), 2);

function alphaRender() {
  const e = alphaExp();
  $("alphaval").textContent =
    parseFloat($("alpha").value).toFixed(2) + " · n^" + e.toFixed(1);
  return e;
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}

function readParams() {
  const t = $("threshold").value;
  return {
    bank_dir: $("bank_dir").value,
    source: $("source").value,
    stride: parseInt($("stride").value || "1", 10),
    zoom: parseFloat($("zoom").value || "1"),
    vmax: parseFloat($("vmax").value || "10"),
    alpha: alphaExp(),
    threshold: t === "" ? null : parseFloat(t),
    loop: $("loop").checked,
    device: $("device").value,
    faiss_threads: parseInt($("faiss_threads").value || "1", 10),
    faiss_gpu: $("faiss_gpu").checked,
  };
}

// ─── Bandeau de banque ─────────────────────────────────────────────────────
function showBank(dir) {
  const m = BANKS[dir];
  $("bankname").textContent = dir ? dir.split("/").pop() : "aucune banque";
  if (!m) { $("chips").innerHTML = ""; return; }
  // Les banques d'avant l'ajout de ces champs n'ont pas tout : une puce vide
  // vaut mieux qu'un « 0.00 Go » faux.
  const num = (v) => (v ? v.toLocaleString("fr") : null);
  const chips = [
    ["dataset", m.dataset],
    ["backbone", m.backbone],
    ["couches", m.layers],
    ["coreset", m.coreset],
    ["image", m.imagesize ? m.imagesize + " px" : null],
    ["banque", m.bank_size ? num(m.bank_size) + " vect." : null],
    ["taille", m.bank_gb ? m.bank_gb.toFixed(2) + " Go" : null],
    ["images fit", num(m.train_images)],
  ];
  $("chips").innerHTML = chips
    .filter(([, v]) => v)
    .map(([k, v]) => `<div class="chip"><span>${k}</span><b>${v}</b></div>`)
    .join("");
}

// ─── Cycle de vie ──────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  $("error").textContent = "";
  const res = await post("/api/start", readParams());
  if (res.error) $("error").textContent = res.error;
  refresh();
});
$("stop").addEventListener("click", async () => { await post("/api/stop"); refresh(); });
$("bank_dir").addEventListener("change", () => showBank($("bank_dir").value));

["zoom", "vmax"].forEach((id) => {
  $(id).addEventListener("input", () =>
    post("/api/update", { [id]: parseFloat($(id).value) }));
});
$("alpha").addEventListener("input", () => post("/api/update", { alpha: alphaRender() }));

$("snap").addEventListener("click", async () => {
  const res = await post("/api/snapshot");
  if (res && res.ok) {
    // Relance l'animation même en clics rapides (reflow forcé).
    const fl = $("flash");
    fl.classList.remove("shoot"); void fl.offsetWidth; fl.classList.add("shoot");
  }
  $("capture").textContent =
    res && res.ok ? "Enregistré : " + res.path : (res && res.error) || "échec";
});

function apply(s) {
  // Le flux MJPEG se termine avec la boucle : on rebranche le <img> à chaque
  // démarrage, avec une URL unique sinon le navigateur ressert la réponse close.
  const cam = $("cam");
  if (s.running && cam.hidden) {
    cam.src = "/stream.mjpg?" + Date.now();
    cam.hidden = false;
    $("placeholder").hidden = true;
  } else if (!s.running && !cam.hidden) {
    cam.hidden = true;
    cam.removeAttribute("src");
    $("placeholder").hidden = false;
  }

  $("score").textContent =
    s.score === null || s.score === undefined ? "—" : s.score.toFixed(2);
  const v = $("verdict");
  if (!s.running) { v.textContent = "arrêté"; v.className = "idle"; }
  else if (s.verdict) { v.textContent = s.verdict; v.className = s.verdict; }
  else { v.textContent = "en cours"; v.className = "idle"; }

  $("meta").textContent = s.running
    ? `${s.device || "…"} · ${s.fps.toFixed(1)} fps · ${s.infer_ms.toFixed(0)} ms/inf · ${s.frames} frames`
    : "";
  $("error").textContent = s.error || "";

  // Seuls les champs .live restent actifs : changer les autres en cours de route
  // rendrait les scores incomparables d'une frame à l'autre.
  fields.forEach((f) => { if (!f.classList.contains("live")) f.disabled = s.running; });
  $("start").disabled = s.running;
  $("stop").disabled = !s.running;
  $("snap").disabled = !s.running;
}

async function refresh() { apply(await (await fetch("/api/state")).json()); }

(async function init() {
  const cfg = await (await fetch("/api/config")).json();
  ALPHA_MAX = cfg.alpha_max;
  ALPHA_DEFAULT = cfg.alpha_default;
  $("faiss_threads").value = cfg.faiss_threads;
  $("faiss_gpu").checked = cfg.faiss_gpu;

  BANKS = Object.fromEntries(cfg.banks.map((b) => [b.dir, b]));
  $("bank_dir").innerHTML = cfg.banks.length
    ? cfg.banks.map((b) => `<option value="${b.dir}">${b.dir}</option>`).join("")
    : '<option value="">aucune banque dans models/</option>';
  showBank($("bank_dir").value);

  $("alpha").value = Math.sqrt(ALPHA_DEFAULT / ALPHA_MAX);
  alphaRender();

  refresh();
  setInterval(refresh, 300);
})();

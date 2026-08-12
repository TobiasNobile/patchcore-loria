"use strict";

const $ = (id) => document.getElementById(id);
const liveForm = $("params");
const fitForm = $("fit");
// Pris sur le panneau entier, pas sur le <form> : les réglages d'affichage
// vivent après lui, sans quoi Entrée dans un champ soumettrait le démarrage.
const liveFields = [...$("livepanel").querySelectorAll("input,select")];

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

// Ordres de grandeur du score de patch selon la couche extraite : l'échelle
// change d'un facteur ~25 entre layer3 et layer4, un vmax repris d'une autre
// couche donne une heatmap uniformément bleue ou saturée. Indicatif : la valeur
// dépend aussi du backbone et de la taille d'image. Une paire [min, max] là où
// la plage utile est trop large pour se réduire à un point ; le champ est
// pré-rempli avec sa borne basse.
const VMAX_HINT = { "l2": 20, "l3": 10, "l4": 260, "l2-l3": 15, "l3-l4": [175, 200] };

// ─── Bandeau de banque ─────────────────────────────────────────────────────
function showBank(dir) {
  const m = BANKS[dir];
  $("bankname").textContent = m ? m.name : "aucune banque";
  if (!m) { $("chips").innerHTML = ""; return; }
  // Les banques d'avant l'ajout de ces champs n'ont pas tout : une puce vide
  // vaut mieux qu'un « 0.00 Go » faux.
  const num = (v) => (v ? v.toLocaleString("fr") : null);
  const chips = [
    ["tâche", m.task],
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

  const hint = VMAX_HINT[m.layers];
  const range = hint === undefined ? null : [].concat(hint);
  $("vmaxhint").textContent = range
    ? `${range.join("–")} pour ${m.layers}` : "inconnu pour " + (m.layers || "?");
  // Pré-remplit tant que la boucle ne tourne pas : en cours, l'utilisateur a
  // peut-être déjà ajusté à la main.
  if (range && !$("vmax").disabled) $("vmax").value = range[0];
}

function fillBanks(banks, keep) {
  BANKS = Object.fromEntries(banks.map((b) => [b.dir, b]));
  $("bank_dir").innerHTML = banks.length
    ? banks.map((b) => `<option value="${b.dir}">${b.name}</option>`).join("")
    : '<option value="">aucune banque dans coresets/</option>';
  if (keep && BANKS[keep]) $("bank_dir").value = keep;
  showBank($("bank_dir").value);
}

// ─── Fit ───────────────────────────────────────────────────────────────────
function selectedLayers() {
  return [...$("layers").querySelectorAll("input:checked")].map((c) => c.value);
}

fitForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  $("fiterror").textContent = "";
  const file = $("archive").files[0];
  if (!file) { $("fiterror").textContent = "Choisir une archive."; return; }
  const layers = selectedLayers();
  if (!layers.length) { $("fiterror").textContent = "Choisir au moins une couche."; return; }

  const query = new URLSearchParams({
    task: $("task").value,
    backbone: $("backbone").value,
    layers: layers.join(","),
    coreset_pct: $("coreset_pct").value,
    train_subset: $("train_subset").value || "0",
  });
  $("fitstatus").textContent = "envoi de l'archive…";
  // Corps brut plutôt que FormData : le serveur recopie le flux sur disque sans
  // avoir à découper un multipart de plusieurs gigaoctets.
  let res;
  try {
    const r = await fetch("/api/fit?" + query, { method: "POST", body: file });
    res = await r.json();
  } catch (err) {
    res = { error: "Envoi interrompu : " + err };
  }
  if (res.error) { $("fiterror").textContent = res.error; $("fitstatus").textContent = ""; }
  refresh();
});

function applyFit(f) {
  const running = f.running;
  $("fitbar").hidden = !running;
  // `total` nul = phase sans compteur (sélection du coreset) : barre pleine en
  // attente plutôt qu'un pourcentage inventé.
  const pct = running && f.total ? Math.round((100 * f.done) / f.total) : 0;
  $("fitfill").style.width = (running && f.total ? pct : 100) + "%";
  $("fitfill").classList.toggle("pending", running && !f.total);

  if (running) {
    const detail = f.total ? ` ${f.done}/${f.total}` : "";
    const secs = f.seconds ? ` · ${Math.round(f.seconds)} s` : "";
    $("fitstatus").textContent = `${f.phase}${detail}${secs}`;
  } else if (f.name) {
    // trained < images : FolderDataset garde 20 % du normal hors banque, de
    // quoi calibrer un seuil. Les deux chiffres, sinon l'écart intrigue.
    const n = f.trained && f.images && f.trained !== f.images
      ? ` · ${f.trained}/${f.images} images (20 % gardés hors banque)`
      : f.trained ? ` · ${f.trained} images` : "";
    $("fitstatus").textContent = `${f.name}${n} · ${Math.round(f.seconds)} s`;
  } else if (!f.error) {
    $("fitstatus").textContent = "";
  }
  $("fiterror").textContent = f.error || "";
  return running;
}

// ─── Cycle de vie ──────────────────────────────────────────────────────────
liveForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  $("error").textContent = "";
  const res = await post("/api/start", readParams());
  if (res.error) $("error").textContent = res.error;
  refresh();
});
$("stop").addEventListener("click", async () => { await post("/api/stop"); refresh(); });
$("bank_dir").addEventListener("change", () => showBank($("bank_dir").value));

["zoom", "vmax", "stride"].forEach((id) => {
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

let lastFitName = null;

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
  // Un repli cuda -> cpu est silencieux côté calcul : sans ce bandeau on croit
  // tourner sur GPU alors que non.
  $("notice").textContent = s.device_note || "";
  $("notice").hidden = !s.device_note;

  const fitting = applyFit(s.fit || {});
  // Une banque qui vient d'être écrite est sélectionnée : c'est celle qu'on
  // veut essayer, et le sélecteur ne l'a pas encore.
  if (s.fit && s.fit.name && s.fit.name !== lastFitName) {
    lastFitName = s.fit.name;
    fetch("/api/banks").then((r) => r.json()).then((d) => {
      const wanted = d.banks.find((b) => b.name + ".pkg" === s.fit.name);
      fillBanks(d.banks, wanted ? wanted.dir : $("bank_dir").value);
    });
  }

  // Seuls les champs .live restent actifs pendant le scoring : changer les
  // autres en cours de route rendrait les scores incomparables d'une frame à
  // l'autre. Un fit fige tout — il monopolise la machine.
  liveFields.forEach((f) => {
    f.disabled = fitting || (s.running && !f.classList.contains("live"));
  });
  [...fitForm.querySelectorAll("input,select,button")].forEach((f) => {
    f.disabled = fitting || s.running;
  });
  $("start").disabled = s.running || fitting;
  // Le même bouton coupe la boucle ou abandonne le fit — les deux occupent le
  // thread principal, il n'y en a jamais qu'un à interrompre.
  $("stop").disabled = !s.running && !fitting;
  $("stop").textContent = fitting ? "Annuler le fit" : "Arrêter";
  $("snap").disabled = !s.running;
}

async function refresh() { apply(await (await fetch("/api/state")).json()); }

(async function init() {
  const cfg = await (await fetch("/api/config")).json();
  ALPHA_MAX = cfg.alpha_max;
  ALPHA_DEFAULT = cfg.alpha_default;
  $("faiss_threads").value = cfg.faiss_threads;
  $("faiss_gpu").checked = cfg.faiss_gpu;

  $("backbone").innerHTML = cfg.backbones
    .map((b) => `<option value="${b.name}">${b.label}</option>`).join("");
  $("layers").innerHTML = cfg.layers.map((l) => `
    <label class="check"><input type="checkbox" value="${l}"
      ${cfg.default_layers.includes(l) ? "checked" : ""}><span>${l}</span></label>`).join("");
  $("coreset_pct").value = cfg.default_coreset_pct;

  fillBanks(cfg.banks);

  $("alpha").value = Math.sqrt(ALPHA_DEFAULT / ALPHA_MAX);
  alphaRender();

  refresh();
  setInterval(refresh, 300);
})();

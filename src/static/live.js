"use strict";

const $ = (id) => document.getElementById(id);
const liveForm = $("params");
const fitForm = $("fit");
// Pris sur les deux conteneurs, pas sur le <form> : les réglages d'affichage
// vivent hors de lui — certains dans le panneau de droite, lissage et alpha sous
// la caméra. Sans les deux, ceux du bas ne seraient pas figés pendant un fit.
const liveFields = [...document.querySelectorAll(
  "#livepanel input, #livepanel select, #stagecontrols input")];

// Course du curseur alpha : exposant = MAX * s². Quadratique pour placer la
// diagonale (n^1) pile au milieu, s=0 donnant la heatmap pleine (n^0).
let ALPHA_MAX = 4, ALPHA_DEFAULT = 2;
// Durée de scène visée par le lissage ; le nombre de cartes en découle côté
// serveur, avec le fps de la source et le stride — la page ne fait que l'afficher.
let SMOOTHING_SECONDS = 1 / 3;
// Plafond d'images pour la banque, relu de /api/config : le serveur l'applique
// de son côté, la page s'en sert pour tirer avant de zipper.
let MAX_IMAGES = 20000;
let BANKS = {};

const esc = (s) => String(s).replace(
  /[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const alphaExp = () => ALPHA_MAX * Math.pow(parseFloat($("alpha").value), 2);

function alphaRender() {
  const e = alphaExp();
  $("alphaval").textContent =
    parseFloat($("alpha").value).toFixed(2) + " · n^" + e.toFixed(1);
  return e;
}

// Une route que le serveur ne connaît pas répond « not found » en texte brut.
// Sans ce filet, JSON.parse lève une SyntaxError et la page l'affiche telle
// quelle, alors que la cause est ailleurs : page et statiques sont relus du
// disque à chaque appel, le module Python une seule fois au démarrage — donc un
// serveur lancé avant une mise à jour sert la nouvelle page et l'ancienne API.
async function readJson(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    const hint = response.status === 404 ? " — serveur à redémarrer ?" : "";
    return { error: `Réponse inattendue (HTTP ${response.status}) : ` +
                    `${text.slice(0, 60)}${hint}` };
  }
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return readJson(r);
}

function readParams() {
  return {
    bank_dir: $("bank_dir").value,
    source: $("source").value,
    stride: parseInt($("stride").value || "1", 10),
    zoom: parseFloat($("zoom").value || "1"),
    vmax: parseFloat($("vmax").value || "10"),
    alpha: alphaExp(),
    smoothing: smoothingMode(),
    smoothing_seconds: parseFloat($("smoothwin").value || "0.3"),
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
const VMAX_HINT = { "l2": 20, "l3": 10, "l4": 260, "l2-l3": 10, "l3-l4": [150, 200] };

// Vrai pendant le scoring. Le champ vmax ne se laisse pré-remplir qu'à l'arrêt :
// en marche, la valeur affichée est celle que l'utilisateur a réglée à l'œil, et
// la réécrire changerait l'image sous ses yeux. Ce n'est pas `vmax.disabled` qui
// le dit — le champ est .live, donc jamais figé par le scoring, seulement par un
// fit, moment où au contraire il faut le remettre à jour.
let RUNNING = false;

// Les couches, écrites comme la banque les nomme : layer3 -> l3.
const layerKey = (layers) => layers.map((l) => l.replace("layer", "l")).join("-");

// Une seule écriture du couple aide + champ, appelée aussi bien par la banque
// choisie que par les cases du fit : le vmax utile ne dépend que des couches, et
// il vaut mieux qu'il suive celles qu'on s'apprête à scorer.
function showVmaxHint(layers) {
  const hint = VMAX_HINT[layers];
  const range = hint === undefined ? null : [].concat(hint);
  $("vmaxhint").textContent = range
    ? `${range.join("–")} pour ${layers}` : "inconnu pour " + (layers || "?");
  if (range && !RUNNING) $("vmax").value = range[0];
}

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
    ["dataset", m.dataset],
    ["backbone", m.backbone],
    ["couches", m.layers],
    ["coreset", m.coreset],
    ["nn", m.num_nn],
    ["image", m.imagesize ? m.imagesize + " px" : null],
    ["banque", m.bank_size ? num(m.bank_size) + " vect." : null],
    ["taille", m.bank_gb ? m.bank_gb.toFixed(2) + " Go" : null],
    ["images fit", num(m.train_images)],
  ];
  $("chips").innerHTML = chips
    .filter(([, v]) => v)
    // Échappé : `dataset` vient d'un nom de fichier, que personne n'a filtré
    // pour du HTML.
    .map(([k, v]) => `<div class="chip"><span>${k}</span><b>${esc(v)}</b></div>`)
    .join("");

  showVmaxHint(m.layers);
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

const srcMode = () => document.querySelector('input[name="srcmode"]:checked').value;

// Ce que l'utilisateur a désigné, pour l'inscrire dans la banque : le nom du zip,
// ou le premier segment de webkitRelativePath, seul endroit où le nom du dossier
// choisi survit — le navigateur n'en livre jamais le chemin.
function datasetName() {
  if (srcMode() === "zip") return ($("archive").files[0] || {}).name || "";
  const first = $("folder").files[0];
  return first ? (first.webkitRelativePath || "").split("/")[0] : "";
}

$("srcmode").addEventListener("change", () => {
  const mode = srcMode();
  $("archive").hidden = mode !== "zip";
  $("folder").hidden = mode !== "dir";
  $("onlinefields").hidden = mode !== "online";
  $("storefield").hidden = mode !== "online";
  // En mode online il n'y a rien à envoyer : la caméra fournit les images, et le
  // scoring démarre tout seul sur la banque obtenue.
  $("dofit").textContent = mode === "online" ? "Filmer, fitter, démarrer" : "Fitter";
  // En online, c'est la banque à venir qui sera scorée : le vmax conseillé est
  // celui de ses couches, et non celui de la banque encore sélectionnée.
  if (mode === "online") showVmaxHint(layerKey(selectedLayers()));
  $("fiterror").textContent = "";
});

// ─── Zip construit dans la page ────────────────────────────────────────────
// Un dossier choisi arrive en liste de fichiers, jamais en chemin : on en fait
// une archive ici pour que le serveur ne connaisse qu'un seul format d'entrée.
// Stocké sans compression — ce sont des images déjà compressées, et le CRC est
// la seule passe à payer.
const IMAGE_RE = /\.(jpe?g|png|bmp)$/i;
const ZIP_MAX_FILES = 65535;         // au-delà, il faut le Zip64
const ZIP_MAX_BYTES = 3 * 1024 ** 3; // idem : les offsets tiennent sur 32 bits

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function le(view, offset, value, bytes) {
  for (let i = 0; i < bytes; i++) view[offset + i] = (value >>> (8 * i)) & 0xff;
}

async function buildZip(files, onProgress) {
  const encoder = new TextEncoder();
  const parts = [];      // Uint8Array (en-têtes) et File (contenus, non copiés)
  const central = [];
  let offset = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const name = encoder.encode(file.webkitRelativePath || file.name);
    // Le CRC exige de lire les octets ; on ne garde qu'un fichier à la fois en
    // mémoire, le Blob final restant adossé aux fichiers sur disque.
    const crc = crc32(new Uint8Array(await file.arrayBuffer()));

    const header = new Uint8Array(30 + name.length);
    le(header, 0, 0x04034b50, 4);   // signature d'en-tête local
    le(header, 4, 20, 2);           // version minimale
    le(header, 6, 0x0800, 2);       // noms en UTF-8
    le(header, 8, 0, 2);            // méthode 0 = stocké
    le(header, 10, 0, 2);           // heure
    le(header, 12, 0x0021, 2);      // date : 1980-01-01, faute de mieux portable
    le(header, 14, crc, 4);
    le(header, 18, file.size, 4);
    le(header, 22, file.size, 4);
    le(header, 26, name.length, 2);
    header.set(name, 30);
    parts.push(header, file);

    const entry = new Uint8Array(46 + name.length);
    le(entry, 0, 0x02014b50, 4);    // signature d'entrée centrale
    le(entry, 4, 20, 2);
    le(entry, 6, 20, 2);
    le(entry, 8, 0x0800, 2);
    le(entry, 10, 0, 2);
    le(entry, 12, 0, 2);
    le(entry, 14, 0x0021, 2);
    le(entry, 16, crc, 4);
    le(entry, 20, file.size, 4);
    le(entry, 24, file.size, 4);
    le(entry, 28, name.length, 2);
    le(entry, 42, offset, 4);       // position de l'en-tête local
    entry.set(name, 46);
    central.push(entry);

    offset += header.length + file.size;
    if (onProgress) onProgress(i + 1, files.length);
  }

  const size = central.reduce((n, e) => n + e.length, 0);
  const end = new Uint8Array(22);
  le(end, 0, 0x06054b50, 4);        // fin du répertoire central
  le(end, 8, files.length, 2);
  le(end, 10, files.length, 2);
  le(end, 12, size, 4);
  le(end, 16, offset, 4);
  return new Blob([...parts, ...central, end], { type: "application/zip" });
}

// Un sous-dossier anomaly/ ne sert qu'à calibrer un seuil : il est gardé entier,
// le plafond ne s'applique qu'aux images de la banque.
const isAnomaly = (f) => (f.webkitRelativePath || "")
  .toLowerCase().split("/").slice(0, -1).includes("anomaly");

// Tirage sans remise (Fisher-Yates partiel) : on ne brasse que les n premières
// places, le reste de la liste n'a pas à être ordonné.
function sample(files, n) {
  const a = [...files];
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(Math.random() * (a.length - i));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

async function fitBody() {
  if (srcMode() === "zip") {
    const file = $("archive").files[0];
    if (!file) throw new Error("Choisir une archive zip.");
    return file;
  }
  const images = [...$("folder").files].filter((f) => IMAGE_RE.test(f.name));
  if (!images.length) throw new Error("Aucune image (jpg, png, bmp) dans ce dossier.");

  // Le tirage a lieu ici et non côté serveur : inutile de zipper puis d'envoyer
  // des images qui ne seraient pas fittées — et un gros dossier passerait sinon
  // sous les limites du zip ci-dessous.
  const anomaly = images.filter(isAnomaly);
  let normal = images.filter((f) => !isAnomaly(f));
  let drawn = "";
  if (normal.length > MAX_IMAGES) {
    drawn = `${MAX_IMAGES} images tirées au hasard sur ${normal.length} · `;
    normal = sample(normal, MAX_IMAGES);
  }
  const kept = [...normal, ...anomaly];

  if (kept.length > ZIP_MAX_FILES) {
    throw new Error(`${kept.length} images : au-delà de ${ZIP_MAX_FILES}, zipper le dossier à la main.`);
  }
  const bytes = kept.reduce((n, f) => n + f.size, 0);
  if (bytes > ZIP_MAX_BYTES) {
    throw new Error(`${(bytes / 1024 ** 3).toFixed(1)} Go : au-delà de 3 Go, zipper le dossier à la main.`);
  }
  return buildZip(kept, (done, total) => {
    $("fitstatus").textContent = `${drawn}archive en préparation ${done}/${total}`;
  });
}

fitForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  $("fiterror").textContent = "";
  const layers = selectedLayers();
  if (!layers.length) { $("fiterror").textContent = "Choisir au moins une couche."; return; }

  if (srcMode() === "online") {
    // Le scoring démarre sur la banque qui va être construite, pas sur celle du
    // sélecteur : le vmax du champ est encore celui de l'autre échelle, et
    // personne n'a l'occasion de le corriger entre le fit et la première frame.
    // On le recale ici, avant de lire les réglages — visible dans le champ, donc
    // ajustable ensuite comme n'importe quel réglage à chaud.
    showVmaxHint(layerKey(layers));
    // Un seul appel : la page n'a pas à orchestrer prise, fit et scoring, qui
    // occupent de toute façon le même thread côté serveur.
    $("fitstatus").textContent = "Filmage en cours…";
    const res = await post("/api/online", Object.assign(readParams(), {
      task: $("task").value,
      backbone: $("backbone").value,
      layers: layers,
      coreset_pct: parseFloat($("coreset_pct").value),
      train_subset: parseInt($("train_subset").value || "0", 10),
      dataset: "camera",
      duree_s: parseFloat($("duree_s").value || "20"),
      images_par_s: parseFloat($("images_par_s").value || "5"),
      stocker: $("stocker").checked,
    }));
    if (res.error) { $("fiterror").textContent = res.error; $("fitstatus").textContent = ""; }
    refresh();
    return;
  }

  let body;
  try {
    body = await fitBody();
  } catch (err) {
    $("fiterror").textContent = err.message;
    $("fitstatus").textContent = "";
    return;
  }

  const query = new URLSearchParams({
    task: $("task").value,
    backbone: $("backbone").value,
    layers: layers.join(","),
    coreset_pct: $("coreset_pct").value,
    train_subset: $("train_subset").value || "0",
    dataset: datasetName(),
  });
  $("fitstatus").textContent = "envoi de l'archive…";
  // Corps brut plutôt que FormData : le serveur recopie le flux sur disque sans
  // avoir à découper un multipart de plusieurs gigaoctets.
  let res;
  try {
    const r = await fetch("/api/fit?" + query, { method: "POST", body });
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

// ─── Vidéo envoyée depuis la page ──────────────────────────────────────────
// Corps brut, comme l'archive du fit : le File est passé tel quel à fetch, donc
// rien ne tient en mémoire, et le serveur répond par le chemin à mettre dans
// Source. C'est le seul moyen d'utiliser un fichier local sans taper son chemin.
$("video").addEventListener("change", async () => {
  const file = $("video").files[0];
  if (!file) return;
  const status = $("videostatus");
  status.classList.remove("bad");
  status.textContent = `envoi de ${file.name}…`;
  let res;
  try {
    const r = await fetch("/api/video?name=" + encodeURIComponent(file.name),
                          { method: "POST", body: file });
    res = await readJson(r);
  } catch (err) {
    res = { error: "Envoi interrompu : " + err };
  }
  if (res.error) {
    status.classList.add("bad");
    status.textContent = res.error;
    return;
  }
  $("source").value = res.path;
  status.textContent = `${res.path} · ${(file.size / 1024 ** 2).toFixed(0)} Mo`;
});

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
$("smoothwin").addEventListener("input", () =>
  post("/api/update", { smoothing_seconds: parseFloat($("smoothwin").value || "0.3") }));
$("alpha").addEventListener("input", () => post("/api/update", { alpha: alphaRender() }));

// Une seule case, donc un seul mode : le maximum. La moyenne stabilisait mais
// diluait un objet vu sur une seule frame — l'inverse de ce qu'on cherche ici.
// Rien ne réécrit la case depuis l'état serveur : une case qu'un poll
// repositionne devient impossible à cocher. C'est la ligne d'état, plus bas, qui
// dit ce qui est réellement appliqué.
const smoothingMode = () => ($("smooth_max").checked ? "max" : "none");

$("smooth_max").addEventListener("change", () =>
  post("/api/update", { smoothing: smoothingMode() }));

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
  RUNNING = !!s.running;
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
  $("status").textContent = s.running ? "en cours" : "arrêté";

  // Le lissage est rappelé sous le score : sur une scène stable son effet est
  // sous le niveau de couleur, et sans ce rappel on doute qu'il s'applique.
  const active = (s.live || {}).smoothing === "max";
  // Le nombre de cartes vient de l'état : il suit le stride et le fps de la
  // source, donc l'annoncer de mémoire côté page serait faux dès le premier
  // changement de stride. Zéro = source pas encore ouverte (chargement de la
  // banque), et surtout pas « une seule carte ».
  const cartes = s.smoothing_frames;
  // « maximum » ne disait plus rien depuis que la moyenne a disparu de la page :
  // c'est le seul mode, autant annoncer ce qu'on voit — le nombre de cartes
  // agrégées. Une carte, c'est la dernière inférence seule : à ce stride elle
  // couvre déjà la durée visée, et cocher la case ne change rien à l'image.
  // Tant que la source n'est pas ouverte, le segment saute plutôt que d'afficher
  // un nombre qu'on ignore.
  const lissage = !active ? "sans lissage"
    : !cartes ? null
    : cartes === 1 ? "lissage sans effet à ce stride"
    : `lissage sur ${cartes} cartes`;
  // Le compte vit aussi sous la case, où on le lit sans avoir à cocher : c'est
  // ce que le lissage ferait au stride courant. La ligne d'état, elle, ne parle
  // que de ce qui s'applique vraiment.
  $("smoothnow").textContent = cartes
    ? ` — ${cartes} carte${cartes > 1 ? "s" : ""} en ce moment`
    : "";
  $("meta").textContent = s.running
    ? [`${s.device || "…"}`, `${s.fps.toFixed(1)} fps`,
       `${s.infer_ms.toFixed(0)} ms/inf`, `${s.frames} frames`,
       s.source_fps ? `source ${Math.round(s.source_fps)} fps` : null,
       lissage].filter(Boolean).join(" · ")
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
  // `??` et non `=` : une clé absente — page chargée contre une version
  // antérieure du serveur — laisserait sinon un undefined s'afficher, ou un NaN
  // se propager dans la course de l'alpha.
  ALPHA_MAX = cfg.alpha_max ?? ALPHA_MAX;
  ALPHA_DEFAULT = cfg.alpha_default ?? ALPHA_DEFAULT;
  // Le serveur borne de son côté ; ici c'est pour que le champ refuse la valeur
  // avant l'envoi, une valeur trop haute ne rendant qu'une bouillie floue.
  if (cfg.zoom_max) $("zoom").max = cfg.zoom_max;
  $("faiss_threads").value = cfg.faiss_threads;
  $("faiss_gpu").checked = cfg.faiss_gpu;

  $("backbone").innerHTML = cfg.backbones
    .map((b) => `<option value="${b.name}">${b.label}</option>`).join("");
  $("layers").innerHTML = cfg.layers.map((l) => `
    <label class="check"><input type="checkbox" value="${l}"
      ${cfg.default_layers.includes(l) ? "checked" : ""}><span>${l}</span></label>`).join("");
  // Après le remplissage des cases, sinon il n'y a rien à écouter. Cocher
  // layer4 change l'échelle des scores d'un facteur ~25 : sans ce rappel, le
  // champ garde le vmax de la banque précédente et la heatmap sort uniforme.
  $("layers").addEventListener("change",
                               () => showVmaxHint(layerKey(selectedLayers())));
  $("coreset_pct").value = cfg.default_coreset_pct;
  MAX_IMAGES = cfg.max_images ?? MAX_IMAGES;
  $("train_subset").max = MAX_IMAGES;
  $("maximages").textContent = MAX_IMAGES;
  SMOOTHING_SECONDS = cfg.smoothing_seconds ?? SMOOTHING_SECONDS;
  $("smoothwin").value = SMOOTHING_SECONDS.toFixed(1);
  if (cfg.smoothing_seconds_max) $("smoothwin").max = cfg.smoothing_seconds_max;

  fillBanks(cfg.banks, cfg.default_bank);

  $("alpha").value = Math.sqrt(ALPHA_DEFAULT / ALPHA_MAX);
  alphaRender();

  refresh();
  setInterval(refresh, 300);
})();

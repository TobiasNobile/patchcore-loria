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
    // Le serveur applique la mesure lui-même quand elle arrive après le
    // démarrage (fit online, fin de test) : il lui faut la même marge.
    vmax_coef: vmaxCoef(),
    alpha: alphaExp(),
    smoothing: smoothingMode(),
    smoothing_seconds: parseFloat($("smoothwin").value || "0.3"),
    loop: $("loop").checked,
    device: $("device").value,
    faiss_threads: parseInt($("faiss_threads").value || "1", 10),
    faiss_gpu: $("faiss_gpu").checked,
  };
}

// Repli pour les banques non calibrées. Ordres de grandeur du score de patch
// selon la couche extraite : l'échelle change d'un facteur ~25 entre layer3 et
// layer4, un vmax repris d'une autre couche donne une heatmap uniformément bleue
// ou saturée. Indicatif, et aveugle à la scène : la valeur dépend aussi du
// backbone, de la taille d'image et de ce qu'il y a devant la caméra. Une paire
// [min, max] là où la plage utile est trop large pour se réduire à un point ; le
// champ est pré-rempli avec sa borne basse.
const VMAX_HINT = { "l2": 20, "l3": 10, "l4": 260, "l2-l3": 10, "l3-l4": [150, 200] };

// Le champ accepte le dixième : une échelle mesurée ne tombe pas sur un entier,
// et l'arrondir à l'unité déplacerait la saturation sur les petites échelles.
const fmtVmax = (v) => Math.round(v * 10) / 10;

// Les couches de ce qu'on s'apprête à scorer — celles de la banque chargée, ou
// celles cochées pour le prochain fit. Cf. showVmaxHint.
let VMAX_LAYERS = "";
// L'échelle mesurée que porte la banque chargée, ou null. Cf. renderVmaxHint.
let VMAX_CALIB = null;
// Vrai tant que le champ vmax porte `coefficient × mesure`. Faux dès qu'une
// valeur de table l'a remplacé : le calcul affiché ne décrirait plus ce qui est
// appliqué, et il vaut mieux ne rien écrire que d'écrire une égalité fausse.
let VMAX_FROM_CALIB = false;

// La marge posée sur l'échelle mesurée. Bornée comme le champ, et jamais
// appliquée à un vmax tapé à la main : elle ne multiplie qu'une mesure.
const vmaxCoef = () =>
  Math.min(Math.max(parseFloat($("vmax_coef").value) || 1, 0.1), 2);
// Quantile des scores du test qui fait le vmax auto-calibré, relu de /api/config.
let CALIB_P = 90;

// Vrai pendant le scoring. Le champ vmax ne se laisse pré-remplir qu'à l'arrêt :
// en marche, la valeur affichée est celle que l'utilisateur a réglée à l'œil, et
// la réécrire changerait l'image sous ses yeux. Ce n'est pas `vmax.disabled` qui
// le dit — le champ est .live, donc jamais figé par le scoring, seulement par un
// fit, moment où au contraire il faut le remettre à jour.
let RUNNING = false;

// Les couches, écrites comme la banque les nomme : layer3 -> l3.
const layerKey = (layers) => layers.map((l) => l.replace("layer", "l")).join("-");

// L'aide du champ vmax tient deux choses, qui arrivent séparément : l'échelle
// mesurée de la banque chargée (avec la banque) et le repère de table des
// couches en jeu (avec les cases du fit). Elles sont donc mémorisées, sinon
// chacune effacerait l'autre — les mesures livrées par un fit ou par un test
// arrivent sans couches, et les cases arrivent sans mesure.
function tableRange(layers) {
  const hint = VMAX_HINT[layers];
  return hint === undefined ? null : [].concat(hint);
}

function renderVmaxHint() {
  const range = tableRange(VMAX_LAYERS);
  const table = range ? `${range.join("–")} pour ${VMAX_LAYERS}` : null;
  const c = VMAX_CALIB;
  // Deux mesures possibles, qui ne disent pas la même chose : le plafond du
  // normal (holdout du fit) ou le pic de l'anomalie jouée (phase de test).
  const mesure = !c ? null
    : c.test ? `${fmtVmax(c.vmax)} — p${c.p ?? 90} des scores du test`
    : `mesuré ${fmtVmax(c.vmax)}`
      + (c.n_images ? ` sur ${c.n_images} image${c.n_images > 1 ? "s" : ""} hors banque` : "");
  // « table » n'est écrit que face à une mesure, qu'il sert à situer : seul, le
  // repère de couche est la seule chose affichée, il n'a rien à distinguer.
  $("vmaxhint").textContent = mesure
    ? [mesure, table && "table " + table].filter(Boolean).join(" · ")
    : table || ("inconnu pour " + (VMAX_LAYERS || "?"));
}

// Le calcul, écrit tel qu'il est fait : sans lui le champ affiche un nombre
// dont l'origine est invisible, et on ne sait pas si c'est la mesure, la marge
// ou une saisie. Vide quand aucune mesure ne le nourrit.
function renderVmaxCalc() {
  const c = VMAX_CALIB;
  if (!c || !VMAX_FROM_CALIB) { $("vmaxcalc").textContent = ""; return; }
  const k = vmaxCoef();
  const source = c.test ? `p${c.p ?? 90} du test` : "mesure hors banque";
  $("vmaxcalc").textContent =
    `${k.toFixed(2)} × ${fmtVmax(c.vmax)} (${source}) = ${fmtVmax(c.vmax * k)}`;
}

// L'échelle que porte une banque : proposée dans le champ, sauf en marche — la
// valeur affichée est alors celle qu'on a réglée à l'œil, la réécrire changerait
// l'image sous les yeux. `force` est l'exception : un fit ou un test qui vient
// de livrer son échelle, sur un champ que personne n'a encore touché.
function showVmaxHint(layers, calib, force) {
  if (layers) VMAX_LAYERS = layers;
  VMAX_CALIB = calib && calib.vmax ? calib : null;
  renderVmaxHint();
  if (VMAX_CALIB) {
    if (!RUNNING || force) {
      $("vmax").value = fmtVmax(VMAX_CALIB.vmax * vmaxCoef());
      VMAX_FROM_CALIB = true;
    }
    renderVmaxCalc();
    return;
  }
  const range = tableRange(VMAX_LAYERS);
  if (range && !RUNNING) { $("vmax").value = range[0]; VMAX_FROM_CALIB = false; }
  renderVmaxCalc();
}

// Les cases de couche, elles, sont un geste explicite : elles l'emportent sur
// toute échelle mesurée et s'appliquent même en marche — cocher layer4 change
// l'ordre de grandeur des scores d'un facteur ~25, et c'est le moyen le plus
// direct de reprendre la table sans taper le nombre. La mesure de la banque
// reste affichée à côté, pour qu'on sache ce qu'on vient d'écarter.
function vmaxFromLayers(layers) {
  VMAX_LAYERS = layers;
  renderVmaxHint();
  const range = tableRange(layers);
  if (!range) { renderVmaxCalc(); return; }
  $("vmax").value = range[0];
  VMAX_FROM_CALIB = false;
  renderVmaxCalc();
  // Écrire le champ sans le dire au serveur laisserait la heatmap sur l'ancienne
  // échelle : le champ mentirait sur ce qui est appliqué.
  if (RUNNING) post("/api/update", { vmax: range[0] });
}

// ─── Bandeau de banque ─────────────────────────────────────────────────────
// Ce que le bandeau montre déjà : il est réécrit trois fois par seconde sinon,
// et le sélecteur se disputerait l'affichage avec la banque en cours de scoring.
let bandeau = null;

function showBank(dir) {
  renderBank(BANKS[dir], "sel:" + dir);
}

function renderBank(m, cle) {
  if (cle === bandeau) return;
  bandeau = cle;
  $("bankname").textContent = m
    ? m.name + (m.stored === false ? " · non stockée" : "") : "aucune banque";
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
    // Ce que la banque sait de sa propre échelle : le pire score nominal hors
    // banque, et sur combien d'images il a été pris. Vide pour une banque
    // d'avant la calibration, qui retombe sur la table par couche.
    ["vmax", m.vmax ? fmtVmax(m.vmax) + (m.vmax_images ? ` · ${m.vmax_images} img` : "") : null],
  ];
  $("chips").innerHTML = chips
    .filter(([, v]) => v)
    // Échappé : `dataset` vient d'un nom de fichier, que personne n'a filtré
    // pour du HTML.
    .map(([k, v]) => `<div class="chip"><span>${k}</span><b>${esc(v)}</b></div>`)
    .join("");

  showVmaxHint(m.layers, { vmax: m.vmax, n_images: m.vmax_images });
}

function fillBanks(banks, keep) {
  BANKS = Object.fromEntries(banks.map((b) => [b.dir, b]));
  // Le contenu du sélecteur change : ce que le bandeau montrait n'est plus
  // forcément à jour, on le laisse se redessiner.
  bandeau = null;
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
  $("onlinenote").hidden = mode !== "online";
  // En mode online il n'y a rien à envoyer : la caméra fournit les images, et le
  // scoring démarre tout seul sur la banque obtenue.
  $("dofit").textContent = mode === "online" ? "Filmer, fitter, démarrer" : "Fitter";
  // En online, c'est la banque à venir qui sera scorée : le vmax conseillé est
  // celui de ses couches, et non celui de la banque encore sélectionnée.
  if (mode === "online") vmaxFromLayers(layerKey(selectedLayers()));
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
    vmaxFromLayers(layerKey(layers));
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
      // Coché, une phase de test s'intercale entre la banque et la démo : on y
      // filme l'anomalie, et son pic devient le vmax.
      autocalib: $("selfcalib").checked,
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
    // trained < images : FolderDataset garde 20 % du normal hors banque, et
    // c'est sur ces images-là que l'échelle vient d'être mesurée. Les deux
    // chiffres, sinon l'écart intrigue.
    const n = f.trained && f.images && f.trained !== f.images
      ? ` · ${f.trained}/${f.images} images (20 % gardés hors banque)`
      : f.trained ? ` · ${f.trained} images` : "";
    const v = f.vmax ? ` · vmax ${fmtVmax(f.vmax)}` : "";
    $("fitstatus").textContent = `${f.name}${n}${v} · ${Math.round(f.seconds)} s`;
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
// Clôt la phase de test en gardant sa mesure — l'inverse d'Arrêter, qui
// abandonne la séquence entière sans rien retenir.
$("endtest").addEventListener("click", async () => {
  $("endtest").disabled = true;   // le serveur met une frame à sortir de sa boucle
  await post("/api/end_test");
  refresh();
});
$("bank_dir").addEventListener("change", () => showBank($("bank_dir").value));

["zoom", "vmax", "stride"].forEach((id) => {
  $(id).addEventListener("input", () =>
    post("/api/update", { [id]: parseFloat($(id).value) }));
});
// Le coefficient recalcule le vmax appliqué depuis la mesure, et l'envoie :
// c'est un réglage d'affichage comme un autre, il doit valoir en marche.
$("vmax_coef").addEventListener("input", () => {
  if (!VMAX_CALIB) { renderVmaxCalc(); return; }
  const v = fmtVmax(VMAX_CALIB.vmax * vmaxCoef());
  $("vmax").value = v;
  VMAX_FROM_CALIB = true;
  renderVmaxCalc();
  post("/api/update", { vmax: v });
});

// Un vmax tapé à la main n'est plus le produit affiché : le calcul s'efface.
$("vmax").addEventListener("input", () => {
  VMAX_FROM_CALIB = false;
  renderVmaxCalc();
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
// Dernière échelle livrée par un fit, pour ne l'écrire qu'une fois : la relire à
// chaque poll écraserait le réglage fait à la main juste après. Idem pour celle
// gardée à la fin d'une phase de test.
let lastFitVmax = null;
let lastKept = null;
// Le flux MJPEG est-il clos ? Vrai au départ, et dès que le serveur cesse de
// filmer et de scorer : le <img> doit alors être rebranché sur une URL neuve.
let camCoupe = true;

function apply(s) {
  RUNNING = !!s.running;
  // Le serveur sert le flux tant qu'on filme ou qu'on score, et le coupe entre
  // les deux — pendant le fit. D'où deux états à distinguer : ce qu'on montre, et
  // si la connexion est encore vivante. On garde la dernière image à l'écran
  // pendant le fit, mais on retient qu'il faudra rouvrir, sinon le scoring
  // reprendrait sur un flux clos — le navigateur resservirait la réponse fermée.
  const cam = $("cam");
  const flux = s.running || !!s.filming;
  if (flux && camCoupe) {
    cam.src = "/stream.mjpg?" + Date.now();   // URL unique : pas de cache
    camCoupe = false;
    cam.hidden = false;
    $("placeholder").hidden = true;
  } else if (!flux) {
    camCoupe = true;
    // Ni filmage ni scoring : le carré ne rend la place au message d'attente que
    // si aucun fit online n'est en cours, dont l'image est le dernier état vu.
    if (!(s.fit || {}).running && !cam.hidden) {
      cam.hidden = true;
      cam.removeAttribute("src");
      $("placeholder").hidden = false;
    }
  }

  $("score").textContent =
    s.score === null || s.score === undefined ? "—" : s.score.toFixed(2);
  // « arrêté » sous une image manifestement vivante se lisait comme une panne :
  // depuis que le filmage occupe le grand carré, il lui faut son propre mot.
  $("status").textContent = s.calibrating ? "test de l'anomalie"
    : s.running ? "en cours"
    : s.filming ? "filmage" : "arrêté";

  // Phase de test : le bouton porte la valeur qu'on garde en le pressant, et la
  // voir monter pendant qu'on présente l'anomalie dit quand l'angle est bon.
  $("endtest").hidden = !s.calibrating;
  if (s.calibrating) {
    $("endtest").disabled = false;
    // La valeur gardée est le p90, pas le pic : les deux sont écrits, l'écart
    // entre eux dit à lui seul si l'anomalie a été montrée assez longtemps.
    $("endtest").textContent = s.calib_p90
      ? `Terminer le test · garder ${fmtVmax(s.calib_p90)} `
        + `(p${CALIB_P} de ${s.calib_n} scores · pic ${fmtVmax(s.calib_max)})`
      : "Terminer le test · rien de scoré";
  }

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
       // Le plafond du normal, mesuré au fit, pendant le seul moment où il sert
       // de repère : un test qui ne le dépasse pas n'a rien montré d'anormal.
       s.calibrating && s.calib_bank_vmax
         ? `max normal ${fmtVmax(s.calib_bank_vmax)}` : null,
       lissage].filter(Boolean).join(" · ")
    : "";
  // Le bandeau nomme ce qui est scoré, pas ce que le sélecteur montre : une
  // prise non stockée n'entre dans aucun sélecteur, et le bandeau restait sur
  // la banque d'avant — d'où « des fois ça marche, des fois pas », selon que la
  // case « Stocker » était cochée. À l'arrêt, il revient au sélecteur.
  if ((s.running || s.filming) && s.bank) {
    renderBank(s.bank, "live:" + s.bank.dir);
  } else if (!s.running && !s.filming && String(bandeau).startsWith("live:")) {
    showBank($("bank_dir").value);
  }

  $("error").textContent = s.error || "";
  // Un repli cuda -> cpu est silencieux côté calcul : sans ce bandeau on croit
  // tourner sur GPU alors que non.
  $("notice").textContent = s.device_note || "";
  $("notice").hidden = !s.device_note;

  const fitting = applyFit(s.fit || {});
  // L'échelle mesurée par le fit qui vient de finir. En mode online la banque
  // n'est pas stockée : elle n'entrera jamais dans le sélecteur, donc showBank
  // ne la portera pas, et le scoring a déjà démarré dessus — d'où l'écriture
  // forcée, sur un champ que personne n'a encore eu le temps de toucher. Le
  // serveur applique la même valeur de son côté ; ici c'est pour qu'on la voie.
  const mesure = (s.fit || {}).vmax;
  // Un fit qui démarre remet le compteur : sans ça, un second fit rendant la
  // même valeur au dixième près ne se réécrirait pas dans un champ entre-temps
  // réglé à la main. Pendant le fit, `vmax` est nul de toute façon.
  if ((s.fit || {}).running) lastFitVmax = null;
  if (mesure && mesure !== lastFitVmax) {
    lastFitVmax = mesure;
    showVmaxHint(null, { vmax: mesure, n_images: s.fit.vmax_images }, true);
  }
  // Le test, quand il a eu lieu, l'emporte sur l'échelle du fit : il vient
  // après, et c'est lui que le serveur applique. Le serveur garde la valeur
  // posée pendant tout le scoring, donc rien à surprendre au bon instant — et
  // rien n'est écrit si le test a été abandonné plutôt que terminé.
  if (s.calib_kept && s.calib_kept !== lastKept) {
    lastKept = s.calib_kept;
    showVmaxHint(null, { vmax: s.calib_kept, test: true, p: CALIB_P }, true);
  }
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
    // Les cases de couche font exception : elles ne règlent que le prochain fit,
    // donc rien ne les oblige à être figées pendant le scoring — et c'est par
    // elles qu'on rappelle l'échelle attendue d'une couche, ce qui n'a d'intérêt
    // que si on peut le faire en regardant l'image.
    f.disabled = f.classList.contains("layerbox") ? fitting : fitting || s.running;
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
    <label class="check"><input type="checkbox" class="layerbox" value="${l}"
      ${cfg.default_layers.includes(l) ? "checked" : ""}><span>${l}</span></label>`).join("");
  // Après le remplissage des cases, sinon il n'y a rien à écouter. Cocher
  // layer4 change l'échelle des scores d'un facteur ~25 : sans ce rappel, le
  // champ garde le vmax de la banque précédente et la heatmap sort uniforme.
  $("layers").addEventListener("change",
                               () => vmaxFromLayers(layerKey(selectedLayers())));
  $("coreset_pct").value = cfg.default_coreset_pct;
  MAX_IMAGES = cfg.max_images ?? MAX_IMAGES;
  $("train_subset").max = MAX_IMAGES;
  $("maximages").textContent = MAX_IMAGES;
  CALIB_P = cfg.calib_percentile ?? CALIB_P;
  // Le serveur borne de son côté ; ici c'est pour que le champ refuse la valeur
  // avant l'envoi.
  if (cfg.vmax_coef_min) $("vmax_coef").min = cfg.vmax_coef_min;
  if (cfg.vmax_coef_max) $("vmax_coef").max = cfg.vmax_coef_max;
  SMOOTHING_SECONDS = cfg.smoothing_seconds ?? SMOOTHING_SECONDS;
  $("smoothwin").value = SMOOTHING_SECONDS.toFixed(1);
  if (cfg.smoothing_seconds_max) $("smoothwin").max = cfg.smoothing_seconds_max;

  fillBanks(cfg.banks, cfg.default_bank);

  $("alpha").value = Math.sqrt(ALPHA_DEFAULT / ALPHA_MAX);
  alphaRender();

  refresh();
  setInterval(refresh, 300);
})();

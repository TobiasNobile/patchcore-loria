#!/usr/bin/env bash
#
# fetch_bank.sh — installe la banque de démonstration dans coresets/.
#
#   bash bin/fetch_bank.sh
#   BANK_VERSION=v0.2 bash bin/fetch_bank.sh   # une autre release
#
# Le .pkg pèse 153 Mo, au-delà des 100 Mio par fichier que GitHub accepte au
# push — et un blob de cette taille resterait dans l'historique de tous les
# clones, même supprimé ensuite. Il est donc publié comme asset de release,
# à côté du dépôt et hors de son historique.
set -euo pipefail

REPO="${BANK_REPO:-TobiasNobile/patchcore-loria}"
VERSION="${BANK_VERSION:-v0.1}"
NAME="WideResNet50_DetectionKnife_l3-l4_p0.01_ts20000.pkg"
SHA256="542bd27165b0f1a7d7b1e341c231726cc940aba25ad9294261233fb84fd9fa0a"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/coresets/$NAME"
URL="https://github.com/$REPO/releases/download/$VERSION/$NAME"

somme() {
    if command -v sha256sum > /dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1   # macOS
    fi
}

if [ -f "$DEST" ]; then
    echo "Déjà installée : $DEST"
    exit 0
fi

mkdir -p "$ROOT/coresets"
TMP="$DEST.part"
trap 'rm -f "$TMP"' EXIT

echo "Téléchargement de $NAME (153 Mo, release $VERSION)…"
curl -fL --progress-bar -o "$TMP" "$URL"

obtenu="$(somme "$TMP")"
if [ "$obtenu" != "$SHA256" ]; then
    echo "Somme de contrôle inattendue — rien n'a été installé." >&2
    echo "  attendu : $SHA256" >&2
    echo "  obtenu  : $obtenu" >&2
    exit 1
fi

mv "$TMP" "$DEST"
trap - EXIT
echo "Installée : $DEST"
echo "Lancer « python main.py » : la page la présélectionne à l'ouverture."

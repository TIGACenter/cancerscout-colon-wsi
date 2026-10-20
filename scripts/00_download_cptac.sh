#!/usr/bin/env bash
# Download the CPTAC-COAD tissue slide images (372 SVS files, 66.7 GB) from The Cancer Imaging
# Archive (doi:10.7937/TCIA.YZWQ-ZZ63, CC BY 4.0) and check them against configs/cptac_manifest.tsv.
#
# TCIA distributes these slides only as an IBM Aspera Faspex package. This step uses the Aspera CLI
# (https://github.com/IBM/aspera-cli):  gem install aspera-cli && ascli config transferd install
# Without it, download the package in a browser from the collection page
# (https://www.cancerimagingarchive.net/collection/cptac-coad/, "Tissue Slide Images"), unpack it
# into $CSWSI_SLIDES_CPTAC, and run this script with VERIFY_ONLY=1.
source "$(dirname "$0")/common.sh"
FASPEX_URL="${CPTAC_FASPEX_URL:-https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjU2MiIsInBhc3Njb2RlIjoiODhkZGVmZDAzMGEyODc0ZjRhOWM0OWM0MDhjOTQxM2MxMjIzNThiYiIsInBhY2thZ2VfaWQiOiI1NjIiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=}"
MANIFEST="$REPO/configs/cptac_manifest.tsv"
mkdir -p "$CSWSI_SLIDES_CPTAC"
if [ "${VERIFY_ONLY:-0}" != 1 ]; then
  command -v ascli >/dev/null || { echo "ascli not found: gem install aspera-cli && ascli config transferd install" >&2; exit 1; }
  DL="$CSWSI_SLIDES_CPTAC/.download"; mkdir -p "$DL"
  ascli faspex5 packages receive --url="$FASPEX_URL" --to-folder="$DL"
  # the package may nest the files in folders; keep a flat <slide_id>.svs layout
  find "$DL" -type f -name '*.svs' -exec mv -n {} "$CSWSI_SLIDES_CPTAC/" \;
  rm -rf "$DL"
fi
# check every file of the manifest: present, same size, same MD5
cd "$CSWSI_SLIDES_CPTAC"
missing=0; bad=0
while IFS=$'\t' read -r name size md5; do
  [ "$name" = file ] && continue
  if [ ! -f "$name" ]; then echo "missing: $name"; missing=$((missing + 1)); continue; fi
  [ "$(stat -L -c %s "$name")" = "$size" ] || { echo "size differs: $name"; bad=$((bad + 1)); continue; }
  [ "${SKIP_MD5:-0}" = 1 ] || [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ] || { echo "md5 differs: $name"; bad=$((bad + 1)); }
done < "$MANIFEST"
n=$(($(wc -l < "$MANIFEST") - 1))
echo "CPTAC-COAD: $((n - missing - bad)) of $n slides verified in $CSWSI_SLIDES_CPTAC ($missing missing, $bad differing)"
[ $((missing + bad)) = 0 ]

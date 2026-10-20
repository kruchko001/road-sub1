#!/bin/bash
# curl-based parallel equivalent of ALASKA_v2_JPG_256_QF100_COLOR.sh
DS="ALASKA_v2_JPG_256_QF100_COLOR"
OUT="./$DS"
mkdir -p "$OUT" ./tmp
fetch() {
  n="$1"; f="$n.jpg"
  [ -f "$OUT/$f" ] && return 0
  if curl -sfL --retry 5 --retry-delay 2 -o "./tmp/$f" "https://alaska.utt.fr/DATASETS/$DS/$f"; then
    mv "./tmp/$f" "$OUT/$f"
  else
    echo "FAILED $f" >> ./log_ALASKA_v2_downloads; rm -f "./tmp/$f"
  fi
}
export -f fetch; export OUT DS
seq -f "%05g" 1 80005 | xargs -P 8 -I{} bash -c 'fetch {}'
echo "DONE: $(ls "$OUT" | wc -l) files"

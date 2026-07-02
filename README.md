# BinaryLLM — trained-only dict file format

## What changes

Dictionary files no longer carry IDs and no longer store the
augment-able set. The runtime extractor produces a full dict
(trained + singletons + specials), but the file written by
`save_dict_*` contains **only the trained content**. Loading via
`load_dict_*` deterministically reinserts the singletons + connector
specials so the loaded dict matches the runtime layout.

### Text format (`##` convention + `[delim]` section)

```
the
am
cata##
un##
##stro##
##port##
##phe
##ing
[delim]
\x20
.
\x0A
```

Parse rules above `[delim]`:

| Pattern | Kind |
|---|---|
| no `##` | Whole |
| `value##` | Start |
| `##value` | End |
| `##value##` | Mid |

### Binary format (`BDI2` per-part records)

```
Header (8 bytes):
  4 bytes magic "BDI2"
  uint8  version (=1)
  uint8  reserved (=0)
  uint16 reserved (=0)

Records (until EOF):
  uint8  kind  (0=Start, 1=End, 2=Mid, 4=Whole, 5=Delimiter)
  uint16 length (little-endian, value bytes)
  bytes  value
```

Kind=3 (Letter) is never written. Augment-set parts (single-char
Whole atoms a-z, 0-9, 6 connectors, and connector Delimiters) are
also filtered at save and regenerated at load.

### Augment set

```
42  single-char Whole atoms      a-z, 0-9, 6 connectors (-, ', &, ,, ., $)
126 positional Letter atoms      same set × 3 positions (x##, ##x##, ##x)
6   connector Delimiter atoms    -, ', &, ,, ., $
```

`augment_with_atoms(dict)` adds these in that order. It's
idempotent — safe to call on a dict that already contains them
(relies on `PartDictionary::add()` returning the existing ID for
duplicates).

## Files

```
src/core/parts/include/dict_augment.hpp   ← NEW
src/core/parts/src/dict_augment.cpp       ← NEW
src/core/parts/include/dict_io.hpp        ← NEW (trained-only save/load)
src/core/parts/src/dict_io.cpp            ← NEW
src/core/parts/CMakeLists.txt             ← +dict_augment.cpp, +dict_io.cpp
apps/phase1_extract/main.cpp              ← uses save_dict_text/binary
apps/uniqueness_smoke/main.cpp            ← uses load_dict_text/binary
tests/core/parts/test_dict_io.cpp         ← NEW
tests/core/parts/CMakeLists.txt           ← +test_dict_io.cpp
```

`PartDictionary` and `PartExtractor` are **not** modified. The new
format support lives entirely in new files using the existing public
API. The extractor still adds the augment set inline during training
(needed for peel iterations to have full coverage); `save_dict_*`
filters those parts out, `load_dict_*` reinserts them.

## Apply, build, test

```sh
cd /Users/williambrendel/Development/Projects/BinaryLLM
unzip -o ~/Downloads/BinaryLLM-io.zip
cmake --build build/debug
ctest --preset debug --output-on-failure
```

## Re-train and inspect the new format

```sh
# Text — human-readable
./build/debug/apps/phase1_extract/phase1_extract \
  data/dict_combined.txt \
  data/corpora/alice_en.txt data/corpora/words.txt

head -30 data/dict_combined.txt    # see the ## convention in action
grep '\[delim\]' data/dict_combined.txt

# Binary — compact (BDI2)
./build/debug/apps/phase1_extract/phase1_extract \
  data/dict_combined.bin \
  data/corpora/alice_en.txt data/corpora/words.txt
```

## Run strategies

```sh
./build/debug/apps/uniqueness_smoke/uniqueness_smoke \
  data/dict_combined.bin \
  data/corpora/words.txt \
  --top 10
```

Default `--algorithm=all` runs discovery / type / type-level /
level-type and prints a side-by-side summary.

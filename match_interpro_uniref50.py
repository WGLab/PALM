# save as parse_match_complete.py
import gzip
import sqlite3
from xml.etree.ElementTree import iterparse

DB = "uniref50/uniref50.sqlite"
MATCH_XML_GZ = "InterPro/match_complete.xml.gz"
IPR_VOCAB = "InterPro/ipr_domain_entry.tsv"  # must have ipr in col 1

# ---------- load allowed InterPro domain IDs ----------
allowed = set()
with open(IPR_VOCAB, "r") as f:
    header = next(f, None)
    for line in f:
        ipr = line.split("\t", 1)[0].strip()
        if ipr:
            allowed.add(ipr)

print(f"[INFO] Loaded {len(allowed)} allowed IPR domain IDs")

# ---------- open sqlite ----------
con = sqlite3.connect(DB)
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS domain_match (
  rep_id TEXT NOT NULL,
  ipr TEXT NOT NULL,
  start INTEGER NOT NULL,
  end INTEGER NOT NULL,
  PRIMARY KEY (rep_id, ipr, start, end)
);
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_domain_match_rep ON domain_match(rep_id);")
con.commit()

def has_protein(rep_id: str) -> bool:
    cur.execute("SELECT 1 FROM protein WHERE rep_id=? LIMIT 1", (rep_id,))
    return cur.fetchone() is not None

def strip_ns(tag: str) -> str:
    return tag.split("}")[-1]

def iter_locations(match_elem):
    """
    Yield (start, end) pairs from whatever location tag InterPro uses.
    Common: <location>, <lcn>, sometimes deeper nested.
    """
    # 1) Most common: explicit tags
    for loc_tag in ("location", "lcn"):
        for loc in match_elem.findall(f".//{loc_tag}"):
            s = loc.get("start")
            e = loc.get("end")
            if s is None or e is None:
                continue
            yield int(s), int(e)

    # 2) Fallback: scan for any element with start/end attrs
    # (kept conservative to avoid false positives)
    for loc in match_elem.iter():
        t = strip_ns(loc.tag)
        if t in {"location", "lcn"}:
            continue
        s = loc.get("start")
        e = loc.get("end")
        if s is None or e is None:
            continue
        try:
            yield int(s), int(e)
        except ValueError:
            continue

batch = []
kept_rows = 0
seen_proteins = 0
matched_proteins = 0

# Stream parse, trigger when closing </protein>
ctx = iterparse(gzip.open(MATCH_XML_GZ, "rt"), events=("end",))

for event, elem in ctx:
    tag = strip_ns(elem.tag)

    if tag != "protein":
        continue

    if seen_proteins%10000==0:
        print('|',end='')
    seen_proteins += 1

    # ---- protein id extraction (your InterPro version) ----
    # Example: <protein id="A0A009PAB7" ...>
    rep_id = elem.get("id") or elem.get("accession") or elem.get("ac")

    if not rep_id or not has_protein(rep_id):
        elem.clear()
        continue

    matched_proteins += 1

    # ---- iterate matches within this protein ----
    for m in elem.findall(".//match"):
        # InterPro ID is inside <ipr id="IPRxxxxxx" .../>
        ipr_elem = m.find(".//ipr")
        if ipr_elem is None:
            continue

        ipr_id = ipr_elem.get("id")
        if not ipr_id or ipr_id not in allowed:
            continue

        # Extract coordinates
        for s, e in iter_locations(m):
            if s > e:
                s, e = e, s
            batch.append((rep_id, ipr_id, s, e))

    # flush periodically
    if len(batch) >= 5000:
        cur.executemany(
            "INSERT OR IGNORE INTO domain_match(rep_id, ipr, start, end) VALUES(?,?,?,?)",
            batch
        )
        con.commit()
        kept_rows += len(batch)
        batch.clear()
        print('.',end='')

    elem.clear()

# final flush
if batch:
    cur.executemany(
        "INSERT OR IGNORE INTO domain_match(rep_id, ipr, start, end) VALUES(?,?,?,?)",
        batch
    )
    con.commit()
    kept_rows += len(batch)

con.close()
print(f"[DONE] Seen proteins: {seen_proteins}")
print(f"[DONE] Proteins found in UniRef50 SQLite: {matched_proteins}")
print(f"[DONE] Inserted domain spans (attempted): {kept_rows}")

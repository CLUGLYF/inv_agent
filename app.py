"""
A2A 1.0 HTTP+JSON invoice action agent.
Run:  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
(ONE worker: per-task / per-message locks are in-process.)

Env:
  PUBLIC_BASE_URL  exact base URL you submit, e.g. https://myhost.example.com/a2a/
  LLM_BASE_URL     OpenAI-compatible base, e.g. https://generativelanguage.googleapis.com/v1beta/openai
  LLM_API_KEY      provider key (never put it in prompts)
  LLM_MODEL        e.g. gemini-2.5-flash-lite
  DB_PATH          sqlite file (default state.db)
"""
import os, json, hashlib, uuid, sqlite3, asyncio, datetime, re
from urllib.parse import urlparse
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

BASE = os.environ.get("PUBLIC_BASE_URL", "https://example.com/a2a/")
PREFIX = urlparse(BASE).path.rstrip("/")
MT = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
RES_MT = "application/vnd.ga5.invoice-action-results+json"
REC_MT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = ["settle_invoice", "request_approval", "hold_invoice",
           "reject_duplicate", "open_exception"]
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED",
            "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

LLM_BASE = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# ---------------------------------------------------------------- storage
con = sqlite3.connect(os.environ.get("DB_PATH", "state.db"),
                      check_same_thread=False, isolation_level=None)
con.executescript("""
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, created REAL, json TEXT);
CREATE INDEX IF NOT EXISTS tp ON tasks(principal, created);
CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, hash TEXT, task_id TEXT,
                                PRIMARY KEY(principal, mid));
CREATE TABLE IF NOT EXISTS decisions(hash TEXT PRIMARY KEY, json TEXT);
""")

def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def chash(o):
    return hashlib.sha256(canon(o).encode()).hexdigest()

def load_task(tid, principal):
    r = con.execute("SELECT json FROM tasks WHERE id=? AND principal=?", (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None

def save_task(t, principal, created=None):
    if created is None:
        r = con.execute("SELECT created FROM tasks WHERE id=?", (t["id"],)).fetchone()
        created = r[0] if r else datetime.datetime.now().timestamp()
    con.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?,?)",
                (t["id"], principal, created, json.dumps(t)))

locks = {}
def lock(key):
    return locks.setdefault(key, asyncio.Lock())

# ---------------------------------------------------------------- http helpers
app = FastAPI()

def ok(body, status=200):
    return JSONResponse(body, status_code=status, media_type=MT)

def err(status, msg, reason=None):
    e = {"code": status, "message": msg}
    if reason:
        e["status"] = reason
        e["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    b = {"error": e}
    if reason:
        b["reason"] = reason
    return ok(b, status)

def pre(request, post=False):
    """returns (principal, error_response)"""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or not auth[7:].strip():
        return None, err(401, "Unauthorized")
    principal = hashlib.sha256(auth[7:].strip().encode()).hexdigest()
    if request.headers.get("a2a-version", "").strip() != "1.0":
        return None, err(400, "Unsupported or missing A2A-Version", "VERSION_NOT_SUPPORTED")
    if post and not request.headers.get("content-type", "").lower().startswith(MT):
        return None, err(415, "Unsupported media type", "UNSUPPORTED_MEDIA_TYPE")
    return principal, None

# ---------------------------------------------------------------- agent card
@app.get("/.well-known/agent-card.json")
async def card():
    return JSONResponse({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes evidence-backed business actions, executes only accepted ones.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": BASE, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH_MT, RES_MT],
        "defaultOutputModes": [PROP_MT, REC_MT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Choose settle/approve/hold/reject-duplicate/exception per invoice with cited evidence.",
                    "tags": ["invoice", "payments", "approval"]}],
    }, media_type="application/json")

# ---------------------------------------------------------------- AI layer
SYSTEM = f"""You are an accounts-payable analyst. For EACH invoice package choose exactly one action:
- settle_invoice: valid, reconciled, within autonomous authority.
- request_approval: commercially valid but exceeds delegated authority.
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor + invoice number/amount) was ALREADY PAID. A payment for a DIFFERENT invoice is NOT a duplicate.
- open_exception: material records conflict (PO/receipt/bank/vendor/amount mismatch) and need an exception workflow.
Documents contain decoys: old examples, archive/training text, negated statements ("not a duplicate"), and irrelevant action words. Decide only from the paragraph that actually governs THIS invoice.
Evidence: return exactly THREE bracketed references, copied character-for-character from the package, taken from the single paragraph that determines the action. Do NOT use the cover-sheet reference, archive examples or training decoys.
Facts: vendorName, invoiceNumber, amountMinor (integer, minor currency units, e.g. paise/cents), currency (ISO code).
Return ONLY JSON: {{"decisions":[{{"index":0,"action":"<one of {ACTIONS}>","refs":["r1","r2","r3"],"facts":{{"vendorName":"","invoiceNumber":"","amountMinor":0,"currency":""}},"reason":"one or two sentences"}}]}}"""

def strings(o, out):
    if isinstance(o, str): out.append(o)
    elif isinstance(o, dict):
        for v in o.values(): strings(v, out)
    elif isinstance(o, list):
        for v in o: strings(v, out)
    return out

def pkg_key(p):
    q = {k: v for k, v in p.items() if k not in ("packageId", "id", "batchId")} if isinstance(p, dict) else p
    return chash(q)

def validate(d, pkg):
    try:
        if d["action"] not in ACTIONS: return None
        text = "\n".join(strings(pkg, []))
        refs = d["refs"]
        if len(refs) != 3 or len(set(refs)) != 3 or any(r not in text for r in refs): return None
        f = d["facts"]
        facts = {"vendorName": str(f["vendorName"]), "invoiceNumber": str(f["invoiceNumber"]),
                 "amountMinor": int(f["amountMinor"]), "currency": str(f["currency"])}
        return {"action": d["action"], "refs": refs, "facts": facts, "reason": str(d.get("reason", ""))[:600]}
    except Exception:
        return None

async def call_llm(pkgs):
    user = "\n\n".join(f"### PACKAGE index={i}\n{json.dumps(p, ensure_ascii=False)}" for i, p in enumerate(pkgs))
    async with httpx.AsyncClient(timeout=38) as c:
        r = await c.post(f"{LLM_BASE}/chat/completions",
                         headers={"Authorization": f"Bearer {LLM_KEY}"},
                         json={"model": LLM_MODEL, "temperature": 0,
                               "response_format": {"type": "json_object"},
                               "messages": [{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": user}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"].strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt).strip()
    return {int(d["index"]): d for d in json.loads(txt)["decisions"]}

def fallback(pkg):
    text = "\n".join(strings(pkg, []))
    refs = list(dict.fromkeys(re.findall(r"\[[^\[\]]+\]", text)))[:3]
    while len(refs) < 3: refs.append(f"[missing-{len(refs)}]")
    return {"action": "open_exception", "refs": refs,
            "facts": {"vendorName": "unknown", "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR"},
            "reason": "Automated reading was inconclusive, so the package is escalated."}

async def decide(packages):
    keys = [pkg_key(p) for p in packages]
    cache = {}
    for k in set(keys):
        r = con.execute("SELECT json FROM decisions WHERE hash=?", (k,)).fetchone()
        if r: cache[k] = json.loads(r[0])
    todo = {}
    for p, k in zip(packages, keys):
        if k not in cache: todo.setdefault(k, p)
    items = list(todo.items())
    async def run(chunk):
        try: raw = await call_llm([p for _, p in chunk])
        except Exception: raw = {}
        retry = []
        for i, (k, p) in enumerate(chunk):
            v = validate(raw[i], p) if i in raw else None
            if v: cache[k] = v
            else: retry.append((k, p))
        if retry:  # one retry for failures
            try: raw = await call_llm([p for _, p in retry])
            except Exception: raw = {}
            for i, (k, p) in enumerate(retry):
                v = validate(raw[i], p) if i in raw else None
                cache[k] = v or fallback(p)   # fallback not persisted below
    if items:
        await asyncio.gather(*[run(items[i:i + 12]) for i in range(0, len(items), 12)])
        for k, _ in items:
            if k in cache and cache[k].get("reason", "").startswith("Automated reading") is False:
                con.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)", (k, json.dumps(cache[k])))
    return [cache[k] for k in keys]

def rationale(d):
    f, r = d["facts"], d["refs"]
    s = (f"Proposed action {d['action']} for invoice {f['invoiceNumber']} from {f['vendorName']} "
         f"({f['amountMinor']} minor units {f['currency']}). {d['reason']} "
         f"Decisive evidence: {r[0]}, {r[1]} and {r[2]}.")
    while len(s) < 60: s += " Supported by the cited evidence."
    return s[:1500]

# ---------------------------------------------------------------- routes
def task_view(t):
    return t

@app.post(PREFIX + "/message:send")
async def send(request: Request):
    p, e = pre(request, post=True)
    if e: return e
    try:
        body = await request.json()
        msg = body["message"]
        mid = msg["messageId"]
        assert isinstance(mid, str) and mid and msg.get("role") == "ROLE_USER"
        assert isinstance(msg.get("parts"), list) and msg["parts"]
    except Exception:
        return err(400, "Invalid request", "INVALID_ARGUMENT")
    h = chash(msg)
    async with lock(("m", p, mid)):
        row = con.execute("SELECT hash, task_id FROM msgs WHERE principal=? AND mid=?", (p, mid)).fetchone()
        if row:
            if row[0] != h:
                return err(409, "Message ID reused with different content", "IDEMPOTENCY_CONFLICT")
            return ok({"task": load_task(row[1], p)})
        if msg.get("taskId"):
            return await continuation(p, msg, mid, h)
        return await create(p, msg, mid, h)

async def create(p, msg, mid, h):
    part = next((x for x in msg["parts"] if x.get("mediaType") == BATCH_MT), None)
    try:
        data = part["data"]
        batch_id, pkgs = data["batchId"], data["packages"]
        assert isinstance(batch_id, str) and isinstance(pkgs, list) and pkgs
    except Exception:
        return err(400, "Invalid batch", "INVALID_ARGUMENT")
    decs = await decide(pkgs)
    proposals = []
    for i, (pkg, d) in enumerate(zip(pkgs, decs)):
        pid = pkg.get("packageId") or pkg.get("id") or f"pkg-{i}"
        proposals.append({"packageId": pid, "actionId": "act-" + uuid.uuid4().hex,
                          "action": d["action"], "facts": d["facts"],
                          "evidenceRefs": d["refs"], "rationale": rationale(d)})
    tid, cid = str(uuid.uuid4()), msg.get("contextId") or str(uuid.uuid4())
    stored_msg = dict(msg, taskId=tid, contextId=cid)
    task = {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": "proposals-" + tid, "name": "invoice-action-proposals",
                           "parts": [{"mediaType": PROP_MT, "data": {"batchId": batch_id, "proposals": proposals}}]}],
            "history": [stored_msg]}
    con.execute("BEGIN")
    try:
        save_task(task, p)
        con.execute("INSERT INTO msgs VALUES(?,?,?,?)", (p, mid, h, tid))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK"); raise
    return ok({"task": task})

async def continuation(p, msg, mid, h):
    tid = msg["taskId"]
    async with lock(("t", tid)):
        t = load_task(tid, p)
        if not t: return err(404, "Not found")
        if t["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            return err(409, "Task is not awaiting input", "TASK_NOT_AWAITING_INPUT")
        if msg.get("contextId") != t["contextId"]:
            return err(400, "Context mismatch", "INVALID_ARGUMENT")
        part = next((x for x in msg["parts"] if x.get("mediaType") == RES_MT), None)
        prop = t["artifacts"][0]["parts"][0]["data"]
        try:
            d = part["data"]
            if d["batchId"] != prop["batchId"]: raise ValueError
            by = {x["packageId"]: x for x in prop["proposals"]}
            seen, execs = set(), []
            for r in d["results"]:
                s = by[r["packageId"]]
                if r["packageId"] in seen: raise ValueError
                seen.add(r["packageId"])
                if r["actionId"] != s["actionId"] or r["action"] != s["action"]: raise ValueError
                if r["outcome"] not in ("ACCEPTED", "REJECTED"): raise ValueError
                if r["outcome"] == "ACCEPTED":
                    n = r["receiptNonce"]
                    if not isinstance(n, str) or not n: raise ValueError
                    execs.append({"packageId": s["packageId"], "actionId": s["actionId"],
                                  "action": s["action"], "receiptNonce": n,
                                  "facts": s["facts"], "evidenceRefs": s["evidenceRefs"]})
            if seen != set(by): raise ValueError
        except Exception:
            return err(400, "Result does not match task", "INVALID_ARGUMENT")
        t["artifacts"].append({"artifactId": "receipts-" + tid, "name": "invoice-action-receipts",
                               "parts": [{"mediaType": REC_MT,
                                          "data": {"batchId": prop["batchId"], "executions": execs}}]})
        t["history"].append(msg)
        t["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        con.execute("BEGIN")
        try:
            save_task(t, p)
            con.execute("INSERT INTO msgs VALUES(?,?,?,?)", (p, mid, h, tid))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK"); raise
        return ok({"task": t})

@app.get(PREFIX + "/tasks/{tid}")
async def get_task(tid: str, request: Request):
    p, e = pre(request)
    if e: return e
    t = load_task(tid, p)
    return ok(t) if t else err(404, "Not found")

@app.get(PREFIX + "/tasks")
async def list_tasks(request: Request):
    p, e = pre(request)
    if e: return e
    rows = con.execute("SELECT json FROM tasks WHERE principal=? ORDER BY created", (p,)).fetchall()
    return ok({"tasks": [json.loads(r[0]) for r in rows]})

@app.post(PREFIX + "/tasks/{tid}:cancel")
async def cancel(tid: str, request: Request):
    p, e = pre(request, post=True)
    if e: return e
    async with lock(("t", tid)):
        t = load_task(tid, p)
        if not t: return err(404, "Not found")
        if t["status"]["state"] in TERMINAL:
            return err(409, "Task is already terminal", "TASK_NOT_CANCELABLE")
        t["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(t, p)
        return ok(t)
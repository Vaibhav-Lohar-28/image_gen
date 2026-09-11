'''python send_test.py -e 1zqac9n64wsv2l -k rpa_YR7Z25IG78V9D82C22KHZ9HTEI821KJNVMXUO3ZT1chvag
python send_test.py -e 1zqac9n64wsv2l -k YOUR_API_KEY --source https://drive.google.com/file/d/1ci8nArzbEDou9n5nIagxhu3aRCRVKYJ7/view?usp=drive_link --ref https://drive.google.com/file/d/1ci8nArzbEDou9n5nIagxhu3aRCRVKYJ7/view?usp=drive_link'''




import argparse, base64, json, os, sys, time, urllib.request
API = "https://api.runpod.ai/v2"

def http(url, key, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if payload is not None else "GET",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

def save(entry, out, i):
    val = entry.get("image") if isinstance(entry, dict) else entry
    name = entry.get("name", f"result_{i}.png") if isinstance(entry, dict) else f"result_{i}.png"
    if not val: return
    dest = os.path.join(out, os.path.basename(name))
    if val.startswith("http"):
        with urllib.request.urlopen(val, timeout=180) as r, open(dest, "wb") as f: f.write(r.read())
    else:
        with open(dest, "wb") as f: f.write(base64.b64decode(val))
    print("Saved:", dest)

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--endpoint", required=True)
ap.add_argument("-k", "--api-key", default=os.environ.get("RUNPOD_API_KEY", ""))
ap.add_argument("--body", default="test_input.json")
ap.add_argument("--source", help="body photo to keep (replaces source_image.png)")
ap.add_argument("--ref", help="face/hair photo (replaces reference_image.png)")
ap.add_argument("--out", default="results")
a = ap.parse_args()
if not a.api_key: sys.exit("set --api-key or RUNPOD_API_KEY")

body = json.load(open(a.body, encoding="utf-8"))
for img in body["input"]["images"]:
    if a.source and img["name"] == "source_image.png":
        img["image"] = base64.b64encode(open(a.source, "rb").read()).decode()
    if a.ref and img["name"] == "reference_image.png":
        img["image"] = base64.b64encode(open(a.ref, "rb").read()).decode()

print("Submitting...")
job = http(f"{API}/{a.endpoint}/run", a.api_key, body)
print("Job", job["id"], "- polling...")
deadline = time.time() + 2400
while time.time() < deadline:
    st = http(f"{API}/{a.endpoint}/status/{job['id']}", a.api_key)
    if st.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"): break
    time.sleep(10)
else:
    sys.exit("timed out")
print("Status:", st.get("status"))
os.makedirs(a.out, exist_ok=True)
json.dump(st, open(os.path.join(a.out, "response.json"), "w"), indent=2)
out = st.get("output") or {}
cands = out.get("images") if isinstance(out, dict) else out
if not isinstance(cands, list): cands = [out]
[save(e, a.out, i) for i, e in enumerate(cands)]
if st.get("status") != "COMPLETED": print(json.dumps(out, indent=2)[:3000])
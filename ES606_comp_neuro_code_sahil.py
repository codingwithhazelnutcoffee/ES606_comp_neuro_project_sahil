import subprocess, sys, importlib
for _mod, _pkg in (("boto3", "boto3"), ("sklearn", "scikit-learn"), ("scipy", "scipy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("PIL", "Pillow")):
    try:
        importlib.import_module(_mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])
import os, re, io, json, glob, time, warnings, itertools
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import scipy.io as sio
from scipy import stats
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d, maximum_filter
from scipy.signal import butter, sosfiltfilt, resample_poly, periodogram, argrelextrema
from scipy.stats import spearmanr, kurtosis, norm
from scipy.stats import multivariate_normal as MVN
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
ROOT = Path(os.environ.get("SLEEP_PROJECT_ROOT", "sleep_state_project"))
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PROC = str(ROOT / "processed")
V2 = PROC + "/v2"
V23 = PROC + "/v23"
for _p in (DATA, OUT, Path(PROC), Path(V2), Path(V23), Path(PROC) / "figures"):
    _p.mkdir(parents=True, exist_ok=True)
SEED = 20261101
rng0 = np.random.default_rng(SEED)
BUCKET = "openneuro.org"
DATASETS = {"EESM19": "ds005185", "EESM23": "ds005178"}
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def section(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


def mad(x, axis=None):
    x = np.asarray(x, float)
    if x.ndim <= 1:
        return 1.4826 * np.median(np.abs(x - np.median(x)))
    m = np.median(x, 0, keepdims=True)
    return 1.4826 * np.median(np.abs(x - m), 0, keepdims=True)


def download(key, dest_root=DATA):
    dest = dest_root / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        s3.download_file(BUCKET, key, str(dest))
    return dest


def list_keys(prefix):
    rows = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        rows += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return pd.DataFrame(rows, columns=["key", "size"])


def parse_key(key):
    name = key.split("/")[-1]
    g = lambda pat: (re.search(pat, name) or [None, None])[1]
    return pd.Series({"sub": g(r"(sub-[^_]+)"), "ses": g(r"(ses-[^_]+)"), "task": g(r"task-([^_]+)"),
                      "acq": g(r"acq-([^_]+)"), "ext": "".join(Path(name).suffixes)})


section("PHASE 2  |  data inventory and audit")
inv = {}
for label, ds in DATASETS.items():
    df = list_keys(f"{ds}/")
    df = pd.concat([df, df["key"].apply(parse_key)], axis=1)
    df["dataset"] = label
    inv[label] = df
    print("%s (%s): %d files, %.2f GB, %d subjects" % (label, ds, len(df), df["size"].sum() / 1e9, df["sub"].nunique()))
pd.concat(inv.values(), ignore_index=True).to_csv(OUT / "inventory.csv", index=False)


def nights_table(label):
    df = inv[label]
    sl = df[df["key"].str.contains("task-sleep")]
    rows = []
    for (sub, ses), g in sl.groupby(["sub", "ses"]):
        psg = g[g["acq"] == "PSG"]
        if psg.empty:
            continue
        rows.append({"dataset": label, "sub": sub, "ses": ses,
                     "set_MB": psg.loc[psg.ext == ".set", "size"].sum() / 1e6,
                     "fdt_MB": psg.loc[psg.ext == ".fdt", "size"].sum() / 1e6,
                     "n_scoring": int(g["acq"].isin(["scoring", "scoring1", "scoring2"]).sum()),
                     "has_diary": bool((g["acq"] == "diary").any())})
    t = pd.DataFrame(rows)
    t["storage"] = np.where(t.fdt_MB > 0, "set+fdt", "set-only")
    return t


NIGHTS = pd.concat([nights_table(l) for l in inv], ignore_index=True)
NIGHTS.to_csv(OUT / "sleep_nights.csv", index=False)
for l, t in NIGHTS.groupby("dataset"):
    print(l, "PSG nights:", len(t), "| subjects:", t["sub"].nunique(), "| scoring files per night:", t["n_scoring"].value_counts().to_dict())


def read_set_header(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    E_ = m["EEG"] if "EEG" in m else SimpleNamespace(**{k: v for k, v in m.items() if not k.startswith("__")})
    g = lambda k, d=None: getattr(E_, k, d)
    sr, pn = float(g("srate", np.nan)), int(g("pnts", 0))
    return dict(srate=sr, nbchan=int(g("nbchan", 0)), pnts=pn, dur_h=pn / sr / 3600)


def set_stem(label, sub, ses):
    return f"{DATASETS[label]}/{sub}/{ses}/eeg/{sub}_{ses}_task-sleep_acq-PSG_eeg"


recs = []
for _, r in NIGHTS[(NIGHTS.dataset == "EESM19") & (NIGHTS.storage == "set+fdt")].iterrows():
    try:
        h = read_set_header(download(set_stem("EESM19", r["sub"], r["ses"]) + ".set"))
        h.update(sub=r["sub"], ses=r["ses"], fdt_MB=r["fdt_MB"], expected_fdt_MB=h["nbchan"] * h["pnts"] * 4 / 1e6)
    except Exception as e:
        h = dict(sub=r["sub"], ses=r["ses"], error=repr(e))
    recs.append(h)
H_SET = pd.DataFrame(recs)
H_SET["fdt_ratio"] = H_SET["fdt_MB"] / H_SET["expected_fdt_MB"]
H_SET.to_csv(OUT / "eesm19_set_headers.csv", index=False)
print("EESM19 .set/.fdt nights with |fdt ratio - 1| > 0.001:")
print(H_SET.loc[(H_SET["fdt_ratio"] - 1).abs() > 0.001, ["sub", "ses", "dur_h", "fdt_ratio"]].round(3).to_string(index=False))


def edf_header(key, size):
    get = lambda a, b: s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={a}-{b}")["Body"].read()
    ns = int(get(0, 255)[252:256])
    hl = 256 * (1 + ns)
    h = get(0, hl - 1)
    nrec, rdur = int(h[236:244]), float(h[244:252])
    off = [256]

    def fld(w):
        o = off[0]
        out = [h[o + i * w:o + (i + 1) * w].decode("latin1").strip() for i in range(ns)]
        off[0] += ns * w
        return out

    labels, _, dim, _, _, _, _, pref, nsamp, _ = (fld(16), fld(80), fld(8), fld(8), fld(8), fld(8), fld(8), fld(80),
                                                  [int(x) for x in fld(8)], fld(32))
    start = pd.to_datetime(h[168:176].decode() + " " + h[176:184].decode(), format="%d.%m.%y %H.%M.%S", errors="coerce")
    return dict(key=key, ns=ns, n_rec=nrec, rec_dur=rdur, labels=labels, prefilter=pref,
                fs=sorted({round(n / rdur, 3) for n in nsamp}), dur_h=nrec * rdur / 3600,
                size_ratio=size / (hl + nrec * 2 * sum(nsamp)) if nrec > 0 else np.nan, start=start)


EDFS = pd.concat([inv[l][inv[l].ext == ".edf"].assign(dataset=l) for l in inv], ignore_index=True)
with ThreadPoolExecutor(8) as ex:
    EH = pd.DataFrame(list(ex.map(lambda r: edf_header(r[0], r[1]), zip(EDFS["key"], EDFS["size"]))))
EH["dataset"] = EDFS["dataset"].values
EH[["sub", "ses"]] = EH["key"].str.extract(r"/(sub-\d+)/(ses-\d+)/")
EH["scorer"] = EH["key"].str.extract(r"edf_used_to_score(\d)")[0]
REQ = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]
EH["missing_eeg"] = EH["labels"].apply(lambda L: [c for c in REQ if c not in L])
EH.drop(columns=["labels", "prefilter"]).to_csv(OUT / "edf_headers.csv", index=False)
print("EDFs missing a mastoid-referenced derivation:")
print(EH.loc[EH["missing_eeg"].str.len() > 0, ["key", "missing_eeg"]].to_string(index=False))
print("EDFs longer than 12 h:")
print(EH.loc[EH["dur_h"] > 12, ["key", "dur_h"]].round(2).to_string(index=False))
print("distinct prefilter strings (EESM19):", EH[EH.dataset == "EESM19"]["prefilter"].apply(lambda p: tuple(sorted(set(p)))).value_counts().head(3).to_dict())

DLAB = {"Lights Out": "lights_out_s", "Got up": "got_up_s", "Went to bed": "bed_s"}
rows, pairs = [], []
for _, r in NIGHTS[NIGHTS.dataset == "EESM19"].iterrows():
    k = lambda a: f"ds005185/{r['sub']}/{r['ses']}/eeg/{r['sub']}_{r['ses']}_task-sleep_acq-{a}_events.tsv"
    s1, s2, dr = [pd.read_csv(download(k(a)), sep="\t") for a in ("scoring1", "scoring2", "diary")]
    m = s1[["onset", "Scoring1"]].merge(s2[["onset", "Scoring2"]], on="onset")
    d = dict(sub=r["sub"], ses=r["ses"], n1=len(s1), n2=len(s2), n_common=len(m), first_onset=s1.onset.iloc[0],
             code7_s1=int((s1.Scoring1 == 7).sum()), code7_s2=int((s2.Scoring2 == 7).sum()),
             agree=(m.Scoring1 == m.Scoring2).mean() if len(m) else np.nan,
             kappa=cohen_kappa_score(m.Scoring1, m.Scoring2) if len(m) else np.nan)
    for t, c in DLAB.items():
        d[c] = dr.loc[dr.trial_type == t, "onset"].iloc[0] if (dr.trial_type == t).any() else np.nan
    rows.append(d)
    pairs.append(m.assign(sub=r["sub"], ses=r["ses"]))
S19 = pd.DataFrame(rows)
S19.to_csv(OUT / "eesm19_scoring_audit.csv", index=False)
M_PAIRS = pd.concat(pairs)
print("EESM19 scorer-1 codes:", M_PAIRS.Scoring1.value_counts().sort_index().to_dict())
print("EESM19 agreement:", S19.agree.describe().round(3).to_dict())
print("EESM19 kappa:", S19.kappa.describe().round(3).to_dict())
print("scored span starts before lights-out (min):", ((S19.lights_out_s - S19.first_onset) / 60).describe().round(1).to_dict())

rows23 = []
for _, r in NIGHTS[(NIGHTS.dataset == "EESM23") & (NIGHTS.n_scoring == 1)].iterrows():
    b = f"ds005178/{r['sub']}/{r['ses']}/eeg/{r['sub']}_{r['ses']}_task-sleep_acq-"
    sc = pd.read_csv(download(b + "scoring_events.tsv"), sep="\t")
    js = json.load(open(download(b + "PSG_eeg.json")))
    rows23.append(dict(sub=r["sub"], ses=r["ses"], n=len(sc), first_onset=sc.onset.iloc[0], last_end=sc.onset.iloc[-1] + 30,
                       json_dur=js.get("RecordingDuration"), fs=js.get("SamplingFrequency"), set_MB=r["set_MB"],
                       artefact=int((sc.scoring == "Artefact").sum())))
S23 = pd.DataFrame(rows23)
S23.to_csv(OUT / "eesm23_scoring_audit.csv", index=False)
print("EESM23 scored nights:", len(S23), "| subjects:", S23["sub"].nunique())

a_ = EH[(EH.dataset == "EESM19") & (EH.scorer == "1")].set_index(["sub", "ses"])[["dur_h", "start"]].add_suffix("_e1")
b_ = EH[(EH.dataset == "EESM19") & (EH.scorer == "2")].set_index(["sub", "ses"])[["dur_h", "start"]].add_suffix("_e2")
T = a_.join(b_).reset_index().merge(S19[["sub", "ses", "first_onset", "n1", "n2"]], on=["sub", "ses"])
T = T.merge(H_SET[["sub", "ses", "dur_h", "fdt_ratio"]].rename(columns={"dur_h": "dur_set_h"}), on=["sub", "ses"], how="left")
T["first_onset_s2"] = [pd.read_csv(DATA / f"ds005185/{s}/{e}/eeg/{s}_{e}_task-sleep_acq-scoring2_events.tsv", sep="\t")["onset"].iloc[0]
                       for s, e in zip(T["sub"], T["ses"])]
T["dstart_s"] = (T["start_e2"] - T["start_e1"]).dt.total_seconds()
T["scored_end2_s"] = T["first_onset_s2"] + 30 * T["n2"]
T["scored_minus_edf2_s"] = T["scored_end2_s"] - T["dur_h_e2"] * 3600
T["end2_common_minus_edf2_s"] = T["scored_minus_edf2_s"] - T["dstart_s"]
T.to_csv(OUT / "eesm19_timebase_audit.csv", index=False)
print("EDF2 start minus EDF1 start (s):", T["dstart_s"].describe().round(1).to_dict())
for name, col in [("onsets relative to EDF2's own start", "scored_minus_edf2_s"), ("onsets on the EDF1 clock", "end2_common_minus_edf2_s")]:
    print("scorer-2 labelled end minus EDF2 length, %s: median %.1f s, share > 0 s %.2f" % (name, T[col].median(), (T[col] > 0).mean()))


def edf_read_signal(path, label):
    with open(path, "rb") as f:
        ns = int(f.read(256)[252:256])
        hl = 256 * (1 + ns)
        f.seek(0)
        h = f.read(hl)
    nrec, rdur = int(h[236:244]), float(h[244:252])
    off = [256]

    def fld(w):
        o = off[0]
        out = [h[o + i * w:o + (i + 1) * w].decode("latin1").strip() for i in range(ns)]
        off[0] += ns * w
        return out

    labels = fld(16)
    fld(80)
    fld(8)
    pmin, pmax, dmin, dmax = [[float(x) for x in fld(8)] for _ in range(4)]
    fld(80)
    nsamp = [int(x) for x in fld(8)]
    j = labels.index(label)
    st = np.cumsum([0] + nsamp)
    mm = np.memmap(path, dtype="<i2", mode="r", offset=hl, shape=(nrec, sum(nsamp)))
    x = np.asarray(mm[:, st[j]:st[j + 1]]).reshape(-1).astype(np.float64)
    x = (x - dmin[j]) * (pmax[j] - pmin[j]) / (dmax[j] - dmin[j]) + pmin[j]
    return x, nsamp[j] / rdur


def emg_series(x, fs):
    fs = int(round(fs))
    sos = butter(4, [20, 45], btype="band", fs=fs, output="sos")
    p = sosfiltfilt(sos, x - x.mean()) ** 2
    n = len(p) // fs
    c = np.concatenate([[0], np.cumsum(p[:n * fs].reshape(n, fs).mean(1))])
    return np.log10((c[30:] - c[:-30]) / 30 + 1e-12)


DE = np.arange(-100, 101)


def waso_curve(Mx, on, lab):
    keep = lab != 7
    first_sleep = np.argmax(np.isin(lab, [2, 3, 4, 5]))
    sel_ep = keep & (np.arange(len(lab)) >= first_sleep)
    y = (lab == 1).astype(float)
    n_w = int(y[sel_ep].sum())
    cur = np.full(len(DE), np.nan)
    if n_w < 5:
        return cur, n_w
    for i, d in enumerate(DE):
        t = np.round(on + d).astype(int)
        ok = sel_ep & (t >= 0) & (t < len(Mx))
        if ok.sum() > 100:
            cur[i] = np.corrcoef(Mx[t[ok]], y[ok])[0, 1]
    return cur, n_w


S19c = S19[["sub", "ses", "code7_s1", "code7_s2"]]
pk = T[(T.dur_h_e1 < 12) & (T.dur_h_e2 < 12) & (T.fdt_ratio > 0.99)].merge(S19c, on=["sub", "ses"])
pk = pk[(pk.code7_s1 <= 100) & (pk.code7_s2 <= 100)].copy()
pk["absd"] = pk["dstart_s"].abs()
sel2 = pk.nlargest(8, "absd").reset_index(drop=True)
emg_curves = {}
for i, r in sel2.iterrows():
    for k in (1, 2):
        p = download(f"ds005185/sourcedata/{r['sub']}/{r['ses']}/scoring/edf_used_to_score{k}.edf")
        x, fs = edf_read_signal(p, "EMG_l-c")
        Mx = emg_series(x, fs)
        p.unlink()
        sc = pd.read_csv(DATA / f"ds005185/{r['sub']}/{r['ses']}/eeg/{r['sub']}_{r['ses']}_task-sleep_acq-scoring{k}_events.tsv", sep="\t")
        emg_curves[(r["sub"], r["ses"], k)] = waso_curve(Mx, sc["onset"].to_numpy(float), sc[f"Scoring{k}"].to_numpy())[0]
rho = np.arange(-60, 61)


def pooled_shift(k, frame):
    cs = []
    for (s, e, kk), cur in emg_curves.items():
        if kk != k or not np.isfinite(cur).any():
            continue
        ds = int(sel2.loc[(sel2["sub"] == s) & (sel2["ses"] == e), "dstart_s"].iloc[0])
        cs.append(np.array([cur[(rr - (ds if frame == "common" else 0)) + 100] for rr in rho]))
    return np.array(cs)


for k, frame in [(1, "own"), (2, "own"), (2, "common")]:
    C = pooled_shift(k, frame)
    mcur = np.nanmean(C, 0)
    bs = [rho[np.nanargmax(np.nanmean(C[rng0.integers(0, len(C), len(C))], 0))] for _ in range(500)]
    print("EMG time-base test, scorer %d, frame %s: nights %d, pooled peak %d s, 95%% CI %s" % (k, frame, len(C), rho[np.nanargmax(mcur)], np.percentile(bs, [2.5, 97.5])))


def build_labels_eesm19():
    rows = []
    for _, r in T.iterrows():
        f = lambda a: DATA / f"ds005185/{r['sub']}/{r['ses']}/eeg/{r['sub']}_{r['ses']}_task-sleep_acq-{a}_events.tsv"
        s1, s2 = pd.read_csv(f("scoring1"), sep="\t"), pd.read_csv(f("scoring2"), sep="\t")
        o1, o2 = s1["onset"].to_numpy(float), s2["onset"].to_numpy(float)
        idx = np.abs(o2[None, :] - o1[:, None]).argmin(1)
        d = o2[idx] - o1
        rows.append(pd.DataFrame({"sub": r["sub"], "ses": r["ses"], "onset": o1, "scorer1": s1["Scoring1"].to_numpy(),
                                  "scorer2": np.where(np.abs(d) <= 15, s2["Scoring2"].to_numpy()[idx], np.nan), "grid_diff_s": d}))
    return pd.concat(rows, ignore_index=True)


LAB19 = build_labels_eesm19()
LAB19.to_csv(f"{PROC}/labels_eesm19.csv", index=False)
S19[["sub", "ses", "bed_s", "lights_out_s", "got_up_s"]].to_csv(f"{PROC}/diary_eesm19.csv", index=False)
print("EESM19 labels:", LAB19.shape)


section("PHASE 2  |  preprocessing (loader v2): EESM19 scored EDFs and EESM23 DC-coupled .set files")
FS_OUT = 100
FLAT_SD_UV, OFFSET_UV, SEG_MAX_UV = 0.01, 500.0, 350.0
WANT = {"C3-M2": ["c3m2"], "F3-M2": ["f3m2"], "O1-M2": ["o1m2", "o2m1"], "EMG": ["emglc"], "EOG": ["eoglm2"]}
FALLBACK_PREFIX = {"EMG": "emg", "EOG": "eog"}


def norm_label(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_edf_header(path):
    with open(path, "rb") as f:
        h = f.read(256)
        ns = int(h[252:256])
        nrec = int(h[236:244])
        dur = float(h[244:252])
        sh = f.read(256 * ns)

    def fld(off, w):
        return [sh[off + i * w: off + (i + 1) * w].decode("latin1").strip() for i in range(ns)]

    o = 0
    lab = fld(o, 16); o += 16 * ns; o += 80 * ns
    dim = fld(o, 8); o += 8 * ns
    pmin = np.array(fld(o, 8), float); o += 8 * ns
    pmax = np.array(fld(o, 8), float); o += 8 * ns
    dmin = np.array(fld(o, 8), float); o += 8 * ns
    dmax = np.array(fld(o, 8), float); o += 8 * ns
    o += 80 * ns
    nsamp = np.array(fld(o, 8), int)
    return dict(ns=ns, nrec=nrec, dur=dur, label=lab, dim=dim, pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax,
                nsamp=nsamp, hdr_bytes=256 + 256 * ns)


def pick_channels(H):
    idx = {}
    labs = [norm_label(l) for l in H["label"]]
    for k, alts in WANT.items():
        hit = []
        for a in alts:
            hit = [i for i, l in enumerate(labs) if l.endswith(a) or l == a]
            if hit:
                break
        if not hit and k in FALLBACK_PREFIX:
            hit = [i for i, l in enumerate(labs) if l.startswith(FALLBACK_PREFIX[k])]
        if not hit:
            raise KeyError(f"{k} not found in {H['label']}")
        idx[k] = hit[0]
    return idx


def read_channel_uv(path, H, i):
    nrec, tot = H["nrec"], int(H["nsamp"].sum())
    raw = np.memmap(path, dtype="<i2", mode="r", offset=H["hdr_bytes"], shape=(nrec, tot))
    off = int(H["nsamp"][:i].sum())
    n = int(H["nsamp"][i])
    d = np.asarray(raw[:, off:off + n]).reshape(-1).astype(np.float64)
    g = (H["pmax"][i] - H["pmin"][i]) / (H["dmax"][i] - H["dmin"][i])
    x = (d - H["dmin"][i]) * g + H["pmin"][i]
    dim = H["dim"][i].lower()
    if dim.startswith("mv"):
        x *= 1e3
    elif not dim.startswith(("uv", "µv", "\u03bcv")):
        raise ValueError(f"unexpected unit {H['dim'][i]!r}")
    return x, n / H["dur"]


def process_channel(x, fs):
    fs = int(round(fs))
    nsec = len(x) // fs
    B = x[:nsec * fs].reshape(nsec, fs)
    sd = B.std(1)
    bm = np.median(B, 1)
    flat = sd < FLAT_SD_UV
    ref = float(np.median(bm[~flat])) if (~flat).any() else 0.0
    bad = flat | (np.abs(bm - ref) > OFFSET_UV)
    xc = x[:nsec * fs] - ref
    xc[np.repeat(bad, fs)] = 0.0
    maxabs = np.abs(B - ref).max(1)
    n4 = nsec // 4
    S = xc[:n4 * 4 * fs].reshape(n4, 4 * fs)
    good = ~bad[:n4 * 4].reshape(n4, 4).any(1) & (np.abs(S).max(1) < SEG_MAX_UV)
    if good.sum() >= 20:
        f, P = periodogram(S[good], fs=fs, window="hann", detrend="constant", axis=1)
        psd = np.median(P, 0) / np.log(2)
    else:
        f = np.fft.rfftfreq(4 * fs, 1 / fs)
        psd = np.full(len(f), np.nan)
    y = resample_poly(xc, FS_OUT, fs).astype(np.float32) if fs != FS_OUT else xc.astype(np.float32)
    return dict(y=y, bad=bad, flat=flat, sd=sd.astype(np.float32), maxabs=maxabs.astype(np.float32), ref=ref,
                f=f, psd=psd, n_good_seg=int(good.sum()), nsec=nsec)


def save_night(out_path, out, chans, src, fs_native):
    nsec = min(out[k]["nsec"] for k in chans)
    n100 = nsec * FS_OUT
    np.savez_compressed(out_path, ch=np.array(chans), label=np.array([out[k]["label"] for k in chans]), src=src,
                        fs_native=fs_native, x100=np.stack([out[k]["y"][:n100] for k in chans[:3]]),
                        bad1s=np.stack([out[k]["bad"][:nsec] for k in chans]),
                        flat1s=np.stack([out[k]["flat"][:nsec] for k in chans]),
                        sd1s=np.stack([out[k]["sd"][:nsec] for k in chans]),
                        maxabs1s=np.stack([out[k]["maxabs"][:nsec] for k in chans]),
                        ref=np.array([out[k]["ref"] for k in chans]), psd_f=out[chans[0]]["f"],
                        psd=np.stack([out[k]["psd"] for k in chans]),
                        n_good_seg=np.array([out[k]["n_good_seg"] for k in chans]),
                        emg100=out["EMG"]["y"][:n100], eog100=out["EOG"]["y"][:n100])
    return {k: dict(bad_pct=round(100 * out[k]["bad"][:nsec].mean(), 1), label=out[k]["label"]) for k in chans}


def process_night_v2(edf_path, out_path, src=""):
    H = read_edf_header(edf_path)
    idx = pick_channels(H)
    out = {}
    chans = list(idx)
    for k in chans:
        x, fs = read_channel_uv(edf_path, H, idx[k])
        out[k] = process_channel(x, fs)
        del x
        out[k]["fs"] = fs
        out[k]["label"] = H["label"][idx[k]]
    return save_night(out_path, out, chans, src, out[chans[0]]["fs"])


def v2_night(name, force=False):
    out = f"{V2}/{name}.npz"
    if os.path.exists(out) and not force:
        return "skip"
    sub, ses = name.split("_")
    key = f"ds005185/sourcedata/{sub}/{ses}/scoring/edf_used_to_score1.edf"
    tmp = str(ROOT / f"_tmp_{name}.edf")
    part = f"{V2}/{name}.part.npz"
    s3.download_file(BUCKET, key, tmp)
    try:
        res = process_night_v2(tmp, part, src=key)
        os.replace(part, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return res


edf1 = list_keys("ds005185/sourcedata/")
names19 = sorted(f"{k.split('/')[2]}_{k.split('/')[3]}" for k in edf1["key"] if k.endswith("edf_used_to_score1.edf"))
t_start = time.time()
for i, n in enumerate(names19):
    try:
        r = v2_night(n)
        msg = r if isinstance(r, str) else {k: v["bad_pct"] for k, v in r.items()}
    except Exception as e:
        msg = f"FAIL {type(e).__name__}: {e}"
    print(f"{i + 1}/{len(names19)} {n} {msg} [{time.time() - t_start:.0f}s]")

HP_HZ, LP_HZ, BASE_WIN_S = 0.15, 35.0, 61
DERIV = {"C3-M2": ("C3", "M2"), "F3-M2": ("F3", "M2"), "O1-M2": ("O1", "M2"), "EMG": ("EMGl", "EMGc"), "EOG": ("EOGl", "M2")}


def process_channel23(x, fs):
    fs = int(round(fs))
    nsec = len(x) // fs
    x = x[:nsec * fs]
    B = x.reshape(nsec, fs)
    nanmask = np.isnan(B).any(1)
    with np.errstate(all="ignore"):
        sd = np.nanstd(B, 1)
        bm = np.nanmedian(B, 1)
    flat = (sd < FLAT_SD_UV) & ~nanmask
    valid = ~flat & ~nanmask & np.isfinite(bm)
    if valid.sum() < 10:
        bad = np.ones(nsec, bool)
        base_s = np.zeros(nsec)
    else:
        s = pd.Series(np.where(valid, bm, np.nan)).rolling(BASE_WIN_S, center=True, min_periods=5).median().to_numpy()
        ok = np.isfinite(s)
        base_s = np.interp(np.arange(nsec), np.where(ok)[0], s[ok])
        bad = flat | nanmask | (np.abs(bm - base_s) > OFFSET_UV)
    t_sec = np.arange(nsec) + 0.5
    t_samp = (np.arange(nsec * fs) + 0.5) / fs
    xc = x - np.interp(t_samp, t_sec, base_s)
    xc[np.repeat(bad, fs)] = 0.0
    xc[~np.isfinite(xc)] = 0.0
    with np.errstate(all="ignore"):
        maxabs = np.nanmax(np.abs(B - base_s[:, None]), 1)
    xh = sosfiltfilt(butter(2, HP_HZ, btype="high", fs=fs, output="sos"), xc)
    n4 = nsec // 4
    S = xh[:n4 * 4 * fs].reshape(n4, 4 * fs)
    good = ~bad[:n4 * 4].reshape(n4, 4).any(1) & (np.abs(S).max(1) < SEG_MAX_UV)
    if good.sum() >= 20:
        f, P = periodogram(S[good], fs=fs, window="hann", detrend="constant", axis=1)
        psd = np.median(P, 0) / np.log(2)
    else:
        f = np.fft.rfftfreq(4 * fs, 1 / fs)
        psd = np.full(len(f), np.nan)
    xl = sosfiltfilt(butter(4, LP_HZ, btype="low", fs=fs, output="sos"), xh)
    y = resample_poly(xl, FS_OUT, fs).astype(np.float32)
    return dict(y=y, bad=bad, flat=flat, sd=np.nan_to_num(sd, nan=0).astype(np.float32),
                maxabs=np.nan_to_num(maxabs, nan=0).astype(np.float32), ref=float(np.median(base_s)), f=f, psd=psd,
                n_good_seg=int(good.sum()), nsec=nsec)


def process_night_23(set_path, out_path, src=""):
    m = sio.loadmat(set_path, struct_as_record=False, squeeze_me=True)
    labs = [str(c.labels) for c in np.atleast_1d(m["chanlocs"])]
    fs = float(m["srate"])
    data = m["data"]
    assert data.shape[0] == len(labs) and int(round(fs)) == 250, (data.shape, labs, fs)
    out = {}
    chans = list(DERIV)
    for k, (a, b) in DERIV.items():
        x = data[labs.index(a)].astype(np.float64) - data[labs.index(b)].astype(np.float64)
        out[k] = process_channel23(x, fs)
        out[k]["fs"] = fs
        out[k]["label"] = f"{a}-{b}"
        del x
    return save_night(out_path, out, chans, src, fs)


def v23_night(key, force=False):
    sub, ses = key.split("/")[1], key.split("/")[2]
    name = f"{sub}_{ses}"
    out = f"{V23}/{name}.npz"
    if os.path.exists(out) and not force:
        return "skip"
    tmp = str(ROOT / f"_tmp23_{name}.set")
    part = f"{V23}/{name}.part.npz"
    s3.download_file(BUCKET, key, tmp)
    try:
        res = process_night_23(tmp, part, src=key)
        os.replace(part, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return res


keys23 = list_keys("ds005178/")
psg23 = sorted(k for k in keys23["key"] if k.endswith(".set") and "acq-PSG" in k and "task-sleep" in k)
t_start = time.time()
for i, key in enumerate(psg23):
    name = f"{key.split('/')[1]}_{key.split('/')[2]}"
    try:
        r = v23_night(key)
        msg = r if isinstance(r, str) else {k: v["bad_pct"] for k, v in r.items()}
    except Exception as e:
        msg = f"FAIL {type(e).__name__}: {str(e)[:150]}"
    print(f"{i + 1}/{len(psg23)} {name} {msg} [{time.time() - t_start:.0f}s]")


section("PHASE 2  |  epoch masks, QC rule, labels, time bases, recorder filters")


def epoch_table(lab_night, z, onset_col="onset"):
    rows = []
    nsec = z["bad1s"].shape[1]
    chans = list(z["ch"][:3])
    s0 = np.round(lab_night[onset_col].to_numpy()).astype(int)
    for ci, c in enumerate(chans):
        bad, ma, sd = z["bad1s"][ci], z["maxabs1s"][ci], z["sd1s"][ci]
        for e, s in enumerate(s0):
            if s < 0 or s + 30 > nsec:
                rows.append((e, c, np.nan, np.nan, np.nan, True))
                continue
            b = bad[s:s + 30]
            ok = ~b
            rows.append((e, c, b.mean(), (ma[s:s + 30] > 300).mean(), np.median(sd[s:s + 30][ok]) if ok.any() else np.nan, False))
    return pd.DataFrame(rows, columns=["epoch", "ch", "bad_frac", "hi_amp_frac", "med_sd_uV", "outside"])


MASK_MAX, SD_MAX, MIN_KEEP = 0.10, 150.0, 0.50


def apply_qc(ET, L, cols, sd_max=SD_MAX):
    E_ = ET.merge(L[["night", "epoch"] + cols], on=["night", "epoch"], how="left")
    E_["keep_epoch"] = (E_.bad_frac <= MASK_MAX) & (E_.med_sd_uV <= sd_max) & ~E_.outside.astype(bool)
    cn = E_.groupby(["night", "ch"]).keep_epoch.mean().rename("retained").reset_index()
    cn["channel_ok"] = cn.retained >= MIN_KEEP
    E_ = E_.merge(cn[["night", "ch", "channel_ok"]], on=["night", "ch"])
    E_["include"] = E_.keep_epoch & E_.channel_ok
    return E_, cn


L19 = pd.read_csv(f"{PROC}/labels_eesm19.csv")
L19["night"] = L19["sub"].astype(str) + "_" + L19["ses"].astype(str)
L19["epoch"] = L19.groupby("night").cumcount()
tabs = []
for fn in sorted(glob.glob(f"{V2}/sub-*_ses-*.npz")):
    if ".part" in fn:
        continue
    name = os.path.basename(fn)[:-4]
    z = np.load(fn)
    ln = L19[L19["night"] == name].reset_index(drop=True)
    if len(ln) == 0:
        continue
    t = epoch_table(ln, z)
    t.insert(0, "night", name)
    tabs.append(t)
ET19 = pd.concat(tabs, ignore_index=True)
ET19.to_csv(f"{PROC}/epoch_mask_v2.csv.gz", index=False)
E19, cn19 = apply_qc(ET19, L19, ["scorer1", "scorer2", "onset"])
E19.to_csv(f"{PROC}/epoch_inclusion_v2.csv.gz", index=False)
print("EESM19 excluded night-channels:")
print(cn19[~cn19.channel_ok].round(2).to_string(index=False))
piv19 = cn19.pivot(index="night", columns="ch", values="channel_ok")
print("EESM19 nights with C3 usable:", int(piv19["C3-M2"].sum()), "| C3 or F3:", int((piv19["C3-M2"] | piv19["F3-M2"]).sum()), "| all three:", int(piv19.all(axis=1).sum()))
NM5 = {1: "Wake", 2: "REM", 3: "N1", 4: "N2", 5: "N3"}
print((E19.assign(stage=E19.scorer1.map(NM5)).groupby(["stage", "ch"]).include.mean().unstack() * 100).round(1).to_string())

names23 = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{V23}/sub-*_ses-*.npz") if ".part" not in f)


def load_labels23(names):
    rows = []
    for n in names:
        sub, ses = n.split("_")
        key = f"ds005178/{sub}/{ses}/eeg/{sub}_{ses}_task-sleep_acq-scoring_events.tsv"
        t = pd.read_csv(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()), sep="\t")
        t.insert(0, "ses", ses)
        t.insert(0, "sub", sub)
        rows.append(t.rename(columns={"scoring_idx": "scorer1"}))
    return pd.concat(rows, ignore_index=True)


L23 = load_labels23(names23)
L23.to_csv(f"{PROC}/labels_eesm23.csv", index=False)
L23["night"] = L23["sub"] + "_" + L23["ses"]
L23["epoch"] = L23.groupby("night").cumcount()
tabs = []
for n in names23:
    z = np.load(f"{V23}/{n}.npz")
    ln = L23[L23["night"] == n].reset_index(drop=True)
    t = epoch_table(ln, z)
    t.insert(0, "night", n)
    tabs.append(t)
ET23 = pd.concat(tabs, ignore_index=True)
E23, cn23 = apply_qc(ET23, L23, ["scorer1", "onset"])
E23.to_csv(f"{PROC}/epoch_inclusion_v23.csv.gz", index=False)
print("EESM23 excluded night-channels:")
print(cn23[~cn23.channel_ok].round(2).to_string(index=False))
NM6 = {1: "Wake", 2: "REM", 3: "N1", 4: "N2", 5: "N3", 6: "Artefact"}
print((E23.assign(stage=E23.scorer1.map(NM6)).groupby(["stage", "ch"]).include.mean().unstack() * 100).round(1).to_string())


def emg_env(z):
    e = z["emg100"].astype(np.float64)
    n = len(e) // 100
    b = z["bad1s"][3][:n]
    env = np.log10(np.sqrt((e[:n * 100].reshape(n, 100) ** 2).mean(1)) + 1e-3)
    env[b] = np.nan
    return env


def waso_events(onset, st, min_sleep=2, min_wake=2):
    ev = []
    for i in range(min_sleep, len(st) - min_wake + 1):
        if st[i] == 1 and all(st[i - j] in (2, 3, 4, 5) for j in range(1, min_sleep + 1)) and all(st[i + j] == 1 for j in range(min_wake)):
            ev.append(onset[i])
    return ev


def eventlocked(env, t_ev, half=60):
    segs = []
    for t in t_ev:
        s = int(round(t))
        if s - half < 0 or s + half > len(env):
            continue
        w = env[s - half:s + half].copy()
        if np.isnan(w).mean() > 0.2:
            continue
        segs.append(w - np.nanmedian(w[:30]))
    return np.array(segs)


def changepoint(curve, lo=-30, hi=30, half=60):
    best = (None, -np.inf)
    for t in range(lo, hi + 1):
        i = t + half
        d = np.nanmean(curve[i:i + 30]) - np.nanmean(curve[i - 30:i])
        if d > best[1]:
            best = (t, d)
    return best


def timebase_test(names, labels, v_dir, n_boot=500, seed=0):
    per_night = {}
    for n in names:
        z = np.load(f"{v_dir}/{n}.npz")
        sub, ses = n.split("_")
        L = labels[(labels["sub"] == sub) & (labels["ses"] == ses)]
        segs = eventlocked(emg_env(z), waso_events(L.onset.to_numpy(), L.scorer1.to_numpy()))
        if len(segs):
            per_night[n] = segs
    allseg = np.vstack(list(per_night.values()))
    t_star, d = changepoint(np.nanmean(allseg, 0))
    rng = np.random.default_rng(seed)
    keys = list(per_night)
    boots = [changepoint(np.nanmean(np.vstack([per_night[keys[i]] for i in rng.choice(len(keys), len(keys))]), 0))[0] for _ in range(n_boot)]
    return dict(n_nights=len(per_night), n_events=len(allseg), t_star=t_star, jump=d, ci=np.percentile(boots, [2.5, 97.5]))


for dname, names, labs, vdir in [("EESM19", [n for n in names19 if os.path.exists(f"{V2}/{n}.npz")], L19, V2), ("EESM23", names23, L23, V23)]:
    r = timebase_test(names, labs, vdir)
    print("%s chin-EMG step at sleep-to-wake transitions: %+d s, 95%% CI [%.0f, %.0f] (%d events, %d nights)" % (dname, r["t_star"], *r["ci"], r["n_events"], r["n_nights"]))


def clean_minute_spectrum(fn, ok_channel=True):
    z = np.load(fn)
    x = z["x100"][0].astype(np.float64)
    b = z["bad1s"][0]
    segs = [x[i * 6000:(i + 1) * 6000] for i in range(len(b) // 60) if not b[i * 60:(i + 1) * 60].any() and np.abs(x[i * 6000:(i + 1) * 6000]).max() < 350]
    if len(segs) < 30:
        return None, None
    fq, P = periodogram(np.array(segs), fs=100, window="hann", detrend="constant", axis=1)
    return fq, np.median(P, 0)


files19 = sorted(f for f in glob.glob(f"{V2}/sub-*_ses-*.npz") if ".part" not in f)
good19 = [f for f in files19 if np.load(f)["bad1s"][0].mean() < 0.02][:40]
spec19 = [s for s in (clean_minute_spectrum(f) for f in good19) if s[0] is not None]
FQ = spec19[0][0]
M19 = np.median(np.array([s[1] for s in spec19]), 0)
spec23 = [clean_minute_spectrum(f"{V23}/{n}.npz") for n in names23 if cn23[(cn23.night == n) & (cn23.ch == "C3-M2")].channel_ok.all()]
M23 = np.median(np.array([s[1] for s in spec23 if s[0] is not None]), 0)
r19, r23 = M19[(FQ >= 1) & (FQ <= 2)].mean(), M23[(FQ >= 1) & (FQ <= 2)].mean()
print("C3-M2 spectrum relative to 1-2 Hz (dB): frequency | EESM19 (%d nights) | EESM23 (%d nights)" % (len(spec19), len(spec23)))
for f0 in [0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 35, 38]:
    i = np.argmin(abs(FQ - f0))
    print(" %6.2f Hz | %6.1f | %6.1f" % (FQ[i], 10 * np.log10(M19[i] / r19), 10 * np.log10(M23[i] / r23)))


section("PHASE 4  |  per-epoch spectral features (relative band powers, slope, SEF90, entropy)")
SEG, HOP, NSEG, MIN_SEG = 400, 200, 14, 10
FREQ = np.fft.rfftfreq(SEG, 1 / FS_OUT)
WIN = np.hanning(SEG)
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 25)}
IN = (FREQ >= 0.5) & (FREQ < 25)
FIT = (FREQ >= 2) & (FREQ < 25)
BIN = {k: (FREQ >= lo) & (FREQ < hi) for k, (lo, hi) in BANDS.items()}
FEATS = ["rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "rel_beta", "slope", "sef90", "entropy", "log_total"]
BAND = FEATS[:5]


def epoch_features(P):
    tot = P[:, IN].sum(1)
    out = np.empty((len(P), len(FEATS)))
    for j, k in enumerate(BANDS):
        out[:, j] = P[:, BIN[k]].sum(1) / tot
    lf, lP = np.log10(FREQ[FIT]), np.log10(P[:, FIT])
    lfc = lf - lf.mean()
    out[:, 5] = (lP * lfc).sum(1) / (lfc ** 2).sum()
    c = np.cumsum(P[:, IN], 1) / tot[:, None]
    out[:, 6] = FREQ[IN][(c < 0.9).sum(1)]
    q = P[:, IN] / tot[:, None]
    out[:, 7] = -(q * np.log(q + 1e-300)).sum(1) / np.log(IN.sum())
    out[:, 8] = np.log10(tot)
    return out


def night_channel_features(x, bad1s, onsets):
    st = np.round(onsets[:, None] * FS_OUT).astype(int) + HOP * np.arange(NSEG)[None, :]
    ok = (st >= 0) & (st + SEG <= len(x))
    st = np.where(ok, st, 0)
    sec0 = st // FS_OUT
    sec1 = (st + SEG - 1) // FS_OUT
    cs = np.concatenate([[0], np.cumsum(bad1s.astype(int))])
    nb = cs[np.minimum(sec1, len(bad1s) - 1) + 1] - cs[np.minimum(sec0, len(bad1s) - 1)]
    good = ok & (nb == 0) & (sec1 < len(bad1s))
    seg = x[st[..., None] + np.arange(SEG)]
    seg = (seg - seg.mean(-1, keepdims=True)) * WIN
    P = np.abs(np.fft.rfft(seg, axis=-1)) ** 2
    P = np.where(good[..., None], P, np.nan)
    ngood = good.sum(1)
    with np.errstate(all="ignore"):
        Pm = np.nanmedian(P, axis=1) / np.log(2)
    valid = ngood >= MIN_SEG
    F_ = np.full((len(onsets), len(FEATS)), np.nan)
    if valid.any():
        F_[valid] = epoch_features(Pm[valid])
    return F_, ngood


def build_features(folder, inc):
    parts = []
    for night in sorted(inc["night"].unique()):
        z = np.load(f"{PROC}/{folder}/{night}.npz", allow_pickle=True)
        chans = list(z["ch"])
        x100 = z["x100"]
        bad = z["bad1s"]
        for ci, ch in enumerate(["C3-M2", "F3-M2", "O1-M2"]):
            d = inc[(inc["night"] == night) & (inc["ch"] == ch)].copy()
            if len(d) == 0:
                continue
            F_, ng = night_channel_features(x100[ci].astype(np.float64), bad[chans.index(ch)], d["onset"].to_numpy(float))
            d[FEATS] = F_
            d["n_good_seg"] = ng
            d.loc[~d["include"].to_numpy(bool), FEATS] = np.nan
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


for ds, folder, inc in [("eesm19", "v2", E19), ("eesm23", "v23", E23)]:
    T_ = build_features(folder, inc)
    T_.to_csv(f"{PROC}/features_{ds}_v1.csv.gz", index=False)
    inc_ = T_[T_["include"].astype(bool)]
    print("%s: %d rows, %d included, %.2f%% lost to the clean-segment rule" % (ds, len(T_), len(inc_), 100 * (1 - inc_[FEATS[0]].notna().mean())))
    g = inc_[inc_[FEATS[0]].notna() & (inc_["ch"] == "C3-M2") & inc_["scorer1"].isin([1, 2, 3, 4, 5])]
    print(g.assign(stage=g["scorer1"].map(NM5)).groupby("stage")[FEATS].median().reindex(["Wake", "N1", "N2", "N3", "REM"]).round(3).to_string())


section("PHASE 3  |  S1: Langevin world simulator, validated against exact results")


class MixtureWorld:
    def __init__(self, w, m, s, D=1.0, name=""):
        self.w, self.m, self.s = np.asarray(w, float) / np.sum(w), np.asarray(m, float), np.asarray(s, float)
        self.D, self.name = D, name

    def pdf(self, x):
        x = np.asarray(x, float)[..., None]
        return (self.w * np.exp(-0.5 * ((x - self.m) / self.s) ** 2) / (self.s * np.sqrt(2 * np.pi))).sum(-1)

    def cdf(self, x):
        x = np.asarray(x, float)[..., None]
        return (self.w * stats.norm.cdf((x - self.m) / self.s)).sum(-1)

    def dlogp(self, x):
        x = np.asarray(x, float)[..., None]
        g = self.w * np.exp(-0.5 * ((x - self.m) / self.s) ** 2) / (self.s * np.sqrt(2 * np.pi))
        return (g * (-(x - self.m) / self.s ** 2)).sum(-1) / g.sum(-1)

    def drift(self, x):
        return self.D * self.dlogp(x)

    def sample_stationary(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w)
        return rng.normal(self.m[k], self.s[k])

    def vdr(self, a, b):
        g = np.linspace(a, b, 4001)
        p = self.pdf(g)
        i = np.argmin(p)
        return p[i] / min(self.pdf(a), self.pdf(b)), g[i]

    def mfpt_exact(self, x0, xb, lo=-8.0, n=40001):
        z = np.linspace(lo, xb, n)
        p = self.pdf(z)
        F_ = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(z))])
        f = F_ / p
        j = z >= x0
        return trapezoid(f[j], z[j]) / self.D


def simulate(world, x0, n_epochs, rng, dt=0.05, epoch_min=0.5):
    spe = int(round(epoch_min / dt))
    x = np.array(x0, float)
    out = np.empty((n_epochs, x.size))
    sq = np.sqrt(2 * world.D * dt)
    for e in range(n_epochs):
        out[e] = x
        for _ in range(spe):
            x = x + world.drift(x) * dt + sq * rng.standard_normal(x.shape)
    return out


def calibrate_D(world, target_mfpt, x0, xb):
    world.D = 1.0
    world.D = world.mfpt_exact(x0, xb) / target_mfpt
    return world.D


def iat(x):
    x = x - x.mean(0)
    T_ = x.shape[0]
    f = np.fft.rfft(x, 2 * T_, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:T_] / (np.arange(T_, 0, -1)[:, None])
    ac = (ac / ac[0]).mean(1)
    tau = 1.0
    for k in range(1, T_ - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return tau


def run_checks(seed=1):
    rng = np.random.default_rng(seed)
    ok = True
    W2 = MixtureWorld([0.5, 0.5], [-1, 1], [0.35, 0.35], name="two-well")
    vdr, xv = W2.vdr(-1, 1)
    print("ground-truth valley depth ratio %.4f at x=%.2f (barrier %.2f D)" % (vdr, xv, -np.log(vdr)))
    XB = 0.5
    print("D = %.5f for a 30 min mean passage" % calibrate_D(W2, 30.0, -1.0, XB))
    x0 = W2.sample_stationary(4000, rng)
    for dt in [0.1, 0.05]:
        xT = simulate(W2, x0, 40, rng, dt=dt)[-1]
        ks = stats.kstest(xT, W2.cdf)
        print("CHECK 1 dt=%.2f KS=%.4f p=%.3f" % (dt, ks.statistic, ks.pvalue))
        ok &= ks.pvalue > 0.01
    n = 1500
    x = np.full(n, -1.0)
    t_hit = np.full(n, np.nan)
    alive = np.ones(n, bool)
    dt = 0.05
    sq = np.sqrt(2 * W2.D * dt)
    t = 0.0
    while alive.any() and t < 3000:
        x[alive] += W2.drift(x[alive]) * dt + sq * rng.standard_normal(alive.sum())
        t += dt
        hit = alive & (x >= XB)
        t_hit[hit] = t
        alive &= ~hit
    m, se = np.nanmean(t_hit), np.nanstd(t_hit) / np.sqrt(np.isfinite(t_hit).sum())
    ex = W2.mfpt_exact(-1.0, XB)
    print("CHECK 2 MFPT simulated %.2f +- %.2f | exact %.2f | diff %.1f%%" % (m, se, ex, 100 * (m - ex) / ex))
    ok &= abs(m - ex) < 3 * se + 0.05 * ex
    X = simulate(W2, W2.sample_stationary(300, rng), 400, rng)
    c = (X[:-1] ** 2 * X[1:] - X[:-1] * X[1:] ** 2).mean(0)
    zr = c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))
    print("CHECK 3 reversibility z = %.2f" % zr)
    ok &= abs(zr) < 3
    Nn, T_ = 300, 400
    om = 0.6
    xy = rng.normal(size=(Nn, 2))
    S = np.empty((T_, Nn, 2))
    dtp = 0.05
    for e in range(T_):
        S[e] = xy
        for _ in range(10):
            xy = xy + (np.stack([-xy[:, 0] - om * xy[:, 1], -xy[:, 1] + om * xy[:, 0]], 1)) * dtp + np.sqrt(2 * dtp) * rng.normal(size=(Nn, 2))
    cc = (S[:-1, :, 0] * S[1:, :, 1] - S[:-1, :, 1] * S[1:, :, 0]).mean(0)
    zc = cc.mean() / (cc.std(ddof=1) / np.sqrt(Nn))
    print("CHECK 3b rotating OU z = %.1f" % zc)
    ok &= abs(zc) > 5
    Wc = MixtureWorld([1.0], [0.0], [0.8], name="unimodal")
    calibrate_D(Wc, 30.0, 0.0, 0.8)
    for W in (W2, Wc):
        X = simulate(W, W.sample_stationary(200, rng), 960, rng)
        tau = iat(X)
        print("CHECK 4 %-10s tau %.1f epochs -> ESS %.0f of 960" % (W.name, tau, 960 / tau))
    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    return ok


for sd_ in (1, 2, 3, 4):
    run_checks(seed=sd_)


section("PHASE 3  |  S2: calibration from labels only")
EPOCH_MIN = 0.5
MACRO = {1: "Wake", 2: "REM", 3: "NREM", 4: "NREM", 5: "NREM"}


def sleep_period(st):
    sl = np.isin(st, [2, 3, 4, 5])
    idx = np.where(sl)[0]
    if len(idx) < 3:
        return None
    run = np.convolve(sl.astype(int), np.ones(3, int), "valid")
    st0 = np.where(run == 3)[0]
    return (st0[0], idx[-1]) if len(st0) else None


def bouts(st):
    sp = sleep_period(st)
    if sp is None:
        return []
    a, b = sp
    out = []
    i = a
    while i <= b:
        m = MACRO.get(st[i])
        j = i
        while j + 1 <= b and MACRO.get(st[j + 1]) == m and m is not None:
            j += 1
        left_ok = i == a or MACRO.get(st[i - 1]) is not None
        right_ok = j == b or MACRO.get(st[j + 1]) is not None
        if m is not None and left_ok and right_ok:
            out.append((m, (j - i + 1) * EPOCH_MIN))
        i = j + 1
    return out


def depth_series(st):
    d = np.full(len(st), np.nan)
    for code, v in {1: 0, 3: 1, 4: 2, 5: 3}.items():
        d[st == code] = v
    return d


def pooled_iat(series_list, maxlag=240):
    num = np.zeros(maxlag + 1)
    den = np.zeros(maxlag + 1)
    for d in series_list:
        x = d - np.nanmean(d)
        for k in range(min(maxlag, len(x) - 1) + 1):
            u, v = x[:len(x) - k], x[k:]
            ok = np.isfinite(u) & np.isfinite(v)
            num[k] += (u[ok] * v[ok]).sum()
            den[k] += ok.sum()
    good = den > 0
    ac = np.where(good, num / np.where(good, den, 1), np.nan)
    ac = ac / ac[0]
    tau = 1.0
    for k in range(1, maxlag - 1, 2):
        if not (good[k] and good[k + 1]):
            break
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return tau, ac


def kappa(a, b, cats=(1, 2, 3, 4, 5)):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def summarize_dataset(L, name, scorer_col="scorer1"):
    B_, series, props, n_missing = [], [], [], 0
    for (sb, ss), g in L.groupby(["sub", "ses"], sort=True):
        st = pd.to_numeric(g.sort_values("onset")[scorer_col], errors="coerce").to_numpy()
        st = np.where(np.isin(st, [1, 2, 3, 4, 5]), st, np.nan)
        n_missing += np.isnan(st).sum()
        sp = sleep_period(np.nan_to_num(st, nan=-1))
        if sp is None:
            continue
        B_ += [(sb, ss, m, d) for m, d in bouts(np.nan_to_num(st, nan=-1).astype(int))]
        series.append(depth_series(np.nan_to_num(st, nan=-1).astype(int)[sp[0]:sp[1] + 1]))
        seg = st[sp[0]:sp[1] + 1]
        props.append({c: np.nanmean(seg == c) for c in (1, 2, 3, 4, 5)} | {"spt_h": (sp[1] - sp[0] + 1) * EPOCH_MIN / 60})
    B_ = pd.DataFrame(B_, columns=["sub", "ses", "macro", "dur_min"])
    t = B_.groupby("macro").dur_min.agg(n="size", mean="mean", median="median").round(1)
    P = pd.DataFrame(props)
    tau, ac = pooled_iat(series)
    print("%s: bouts (min)\n%s\nstage shares %%: %s | tau %.1f epochs, ESS %.0f per 960 | missing labels %d" % (
        name, t.to_string(), {nm: round(100 * P[c].mean(), 1) for c, nm in NM5.items()}, tau, 960 / tau, n_missing))
    return dict(iat_epochs=float(tau))


def scorer_agreement(L):
    G = L[(L.grid_diff_s.abs() < 1)].copy()
    a = pd.to_numeric(G.scorer1, errors="coerce")
    b = pd.to_numeric(G.scorer2, errors="coerce")
    ok = a.isin([1, 2, 3, 4, 5]) & b.isin([1, 2, 3, 4, 5])
    G, a, b = G[ok], a[ok].astype(int), b[ok].astype(int)
    ks = [kappa(a[(G["sub"] == s) & (G.ses == e)], b[(G["sub"] == s) & (G.ses == e)]) for s, e in G[["sub", "ses"]].drop_duplicates().itertuples(index=False)]
    cm = pd.crosstab(a, b, normalize="index").mul(100).round(1)
    cm.index = cm.columns = ["Wake", "REM", "N1", "N2", "N3"]
    print("scorer agreement %.1f%%, kappa median %.2f (range %.2f-%.2f)\n%s" % (100 * (a == b).mean(), np.nanmedian(ks), np.nanmin(ks), np.nanmax(ks), cm.to_string()))


summarize_dataset(L19, "EESM19 scorer 1")
summarize_dataset(L19.assign(scorer1=L19.scorer2), "EESM19 scorer 2")
summarize_dataset(L23, "EESM23")
scorer_agreement(L19)
for path, name in [(f"{PROC}/epoch_inclusion_v2.csv.gz", "EESM19"), (f"{PROC}/epoch_inclusion_v23.csv.gz", "EESM23")]:
    c3 = pd.read_csv(path, usecols=["night", "ch", "scorer1", "include"])
    c3 = c3[c3.ch == "C3-M2"]
    print(name, "QC retention C3 by scorer-1 code (%):", c3.groupby("scorer1").include.mean().mul(100).round(1).to_dict())


section("PHASE 3  |  S3: calibrated wake + NREM world, rater model, fast-noise level")
REAL_IAT, REAL_WAKE_MEAN, REAL_WAKE_MEDIAN = 33.8, 1.4, 0.5
SH1 = np.array([3.8, 7.9, 46.9, 19.3]); SH1 = SH1 / SH1.sum()
SH2 = np.array([4.4, 5.5, 45.3, 24.1]); SH2 = SH2 / SH2.sum()
RAW = np.array([[96.3, 2.3, 0.2, 0.0], [8.7, 47.9, 31.8, 0.3], [0.3, 0.8, 86.6, 10.7], [0.0, 0.0, 2.8, 97.2]])
REAL_CM = RAW / RAW.sum(1, keepdims=True)


def build_world(m_w=-5.0, s_w=0.6, w_w=0.049, name="W3"):
    return MixtureWorld([w_w, 1 - w_w], [m_w, 0.0], [s_w, 1.0], name=name)


def observed_world(world, sig_f):
    return MixtureWorld(world.w, world.m, np.sqrt(world.s ** 2 + sig_f ** 2), name=world.name + f"+f{sig_f}")


def valley_info(world, a, b):
    g = np.linspace(a, b, 6001)
    p = world.pdf(g)
    i = int(np.argmin(p))
    has = 0 < i < len(g) - 1 and p[i] < 0.999 * min(p[0], p[-1])
    r = p[i] / min(p[: i + 1].max(), p[i:].max()) if has else 1.0
    return has, r, g[i], (-np.log(r) if has else 0.0)


def thresholds(z, shares):
    return np.quantile(z, np.cumsum(shares)[:-1])


def label_of(z, thr):
    return np.digitize(z, thr)


def sim_labels(world, n, rng, sig_r, sig_f, thr, dt=0.1):
    X = simulate(world, world.sample_stationary(n, rng), 960, rng, dt=dt).T
    z = X + rng.normal(0, sig_f, X.shape) + rng.normal(0, sig_r, X.shape)
    return np.array([1, 3, 4, 5])[label_of(z, thr)]


def iat_of_labels(codes):
    return pooled_iat([depth_series(c.astype(int)) for c in codes])[0]


def wake_bouts(codes):
    return np.array([d for c in codes for m, d in bouts(c.astype(int)) if m == "Wake"])


def study_s3(m_w, sig_f, seed=5):
    W = build_world(m_w=m_w)
    rng = np.random.default_rng(seed)

    def cmf(sr):
        r = np.random.default_rng(3)
        n = 150000
        x = W.sample_stationary(n, r) + r.normal(0, sig_f, n)
        z1, z2 = x + r.normal(0, sr, n), x + r.normal(0, sr, n)
        l1, l2 = label_of(z1, thresholds(z1, SH1)), label_of(z2, thresholds(z2, SH2))
        return np.array([[np.mean(l2[l1 == i] == j) for j in range(4)] for i in range(4)])

    fit = minimize_scalar(lambda s: np.abs(cmf(s) - REAL_CM).sum(), bounds=(0.02, 1.0), method="bounded", options={"xatol": 0.02})
    sr = fit.x
    xs = W.sample_stationary(300000, rng) + rng.normal(0, sig_f, 300000) + rng.normal(0, sr, 300000)
    thr = thresholds(xs, SH1)

    def err(logD):
        W.D = np.exp(logD)
        return (iat_of_labels(sim_labels(W, 100, np.random.default_rng(1), sr, sig_f, thr)) - REAL_IAT) ** 2

    W.D = np.exp(minimize_scalar(err, bounds=(np.log(0.02), np.log(20.0)), method="bounded", options={"xatol": 0.03}).x)
    c = sim_labels(W, 150, np.random.default_rng(11), sr, sig_f, thr)
    wb = wake_bouts(c)
    return dict(sig_r=sr, cm_err=fit.fun, iat=iat_of_labels(c), wake_mean=wb.mean() if len(wb) else np.nan,
                wake_median=np.median(wb) if len(wb) else np.nan, wake_pct=100 * np.mean(c == 1))


for m_w in [-4.5, -5.0, -6.0]:
    print("wake well at %.1f | %s" % (m_w, " | ".join("f=%.1f: %s" % (sf, ("%.3f" % valley_info(observed_world(build_world(m_w=m_w), sf), m_w, 0.0)[1]) if valley_info(observed_world(build_world(m_w=m_w), sf), m_w, 0.0)[0] else "none") for sf in [0.0, 0.4, 0.8, 1.2])))
for m_w in [-6.0, -5.0, -3.0]:
    for sf in [0.0, 0.4, 0.8, 1.2]:
        s = study_s3(m_w, sf)
        print("wake %.1f noise %.1f | rater sd %.2f CM err %.2f | tau %.1f | wake bout mean %.1f median %.1f | wake %.1f%%" % (
            m_w, sf, s["sig_r"], s["cm_err"], s["iat"], s["wake_mean"], s["wake_median"], s["wake_pct"]))


section("PHASE 3  |  S4: statistic A (valley depth) against analytic truth")
NU = 0.025


def kde_grid(x, h, n_grid=2048, pad=4.0):
    lo, hi = x.min() - pad * h, x.max() + pad * h
    edges = np.linspace(lo, hi, n_grid + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    cnt, _ = np.histogram(x, bins=edges)
    p = gaussian_filter1d(cnt.astype(float), h / dx, mode="constant")
    return c, p / (p.sum() * dx), dx


def valleys(c, p, dx, nu=NU):
    i = np.arange(1, len(p) - 1)
    pk = list(i[(p[i] >= p[i - 1]) & (p[i] > p[i + 1])])
    mn = i[(p[i] <= p[i - 1]) & (p[i] < p[i + 1])]
    if len(pk) < 2:
        return []
    vl = [int(min((v for v in mn if pk[k] < v < pk[k + 1]), key=lambda v: p[v], default=-1)) for k in range(len(pk) - 1)]
    if any(v < 0 for v in vl):
        return []
    while len(pk) > 1:
        edges = [0] + vl + [len(p) - 1]
        mass = [p[edges[k]:edges[k + 1] + 1].sum() * dx for k in range(len(pk))]
        k = int(np.argmin(mass))
        if mass[k] >= nu:
            break
        j = 0 if k == 0 else (len(vl) - 1 if k == len(pk) - 1 else (k - 1 if p[vl[k - 1]] >= p[vl[k]] else k))
        pk[j] = pk[j] if p[pk[j]] >= p[pk[j + 1]] else pk[j + 1]
        del pk[j + 1]
        del vl[j]
    return [(c[v], p[v], p[pk[k]], p[pk[k + 1]]) for k, v in enumerate(vl)]


def deepest_valley(x, c_bw=1.0, nu=NU, n_eff=None):
    x = np.asarray(x).ravel()
    n = len(x) if n_eff is None else n_eff
    sd = min(x.std(), (np.quantile(x, .75) - np.quantile(x, .25)) / 1.34)
    h = c_bw * 0.9 * sd * n ** (-0.2)
    c, p, dx = kde_grid(x, h)
    vs = valleys(c, p, dx, nu)
    if not vs:
        return 0.0, np.nan, h
    r = [v / min(a, b) for _, v, a, b in vs]
    k = int(np.argmin(r))
    return 1 - r[k], vs[k][0], h


rng = np.random.default_rng(0)
for m_w in [-3.0, -4.5, -5.0, -6.0]:
    for sf in [0.0, 0.4]:
        Wo = observed_world(build_world(m_w=m_w), sf)
        h_, r_, xv, _ = valley_info(Wo, m_w, 0.0)
        row = []
        for n in [2000, 20000]:
            est = np.array([1 - deepest_valley(Wo.sample_stationary(n, rng))[0] for _ in range(200)])
            row.append("%.3f [%.3f], %3.0f%%" % (est.mean(), est.std(), 100 * np.mean(est < 0.999)))
        print("wake %.1f noise %.1f | truth %s | n=2000 %s | n=20000 %s" % (m_w, sf, ("%.3f (%.2f)" % (r_, xv)) if h_ else "none", row[0], row[1]))


section("PHASE 3  |  S5: statistic A on autocorrelated pooled nights")
T_EP, N_NIGHTS, TAU = 960, 80, 45.0


def iat_g(x):
    x = x - x.mean()
    T_ = x.shape[0]
    f = np.fft.rfft(x, 2 * T_, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:T_] / (np.arange(T_, 0, -1)[:, None])
    ac = (ac / ac[0]).mean(1)
    tau = 1.0
    for k in range(1, T_ - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return tau


def set_D(world, tau_target, rng, n_traj=300, dt=0.1):
    D = 0.05
    for _ in range(4):
        world.D = D
        X = simulate(world, world.sample_stationary(n_traj, rng), T_EP, rng, dt=dt)
        D *= (iat_g(X) - 1) / (tau_target - 1)
    world.D = D
    return D


def replicates(world, sf, R_, rng, dt=0.1):
    X = simulate(world, world.sample_stationary(R_ * N_NIGHTS, rng), T_EP, rng, dt=dt)
    X = X + rng.normal(0, sf, X.shape)
    return X.reshape(T_EP, R_, N_NIGHTS)


def estimates(X, cut=None):
    V, NE, SH = [], [], []
    for r in range(X.shape[1]):
        x = X[:, r, :]
        ne = x.size / iat_g(x)
        V.append(deepest_valley(x, n_eff=ne)[0])
        NE.append(ne)
        SH.append(np.mean(x < cut) if cut is not None else np.nan)
    return np.array(V), np.array(NE), np.array(SH)


NULLS = {"W0 Gaussian": MixtureWorld([1.0], [0.0], [1.0]), "W1a shoulder": build_world(m_w=-3.0, w_w=0.049), "W1b skew": build_world(m_w=-2.5, w_w=0.15)}
ALTS = {"A1 wake -4.5": build_world(m_w=-4.5), "A2 wake -5.0": build_world(m_w=-5.0), "A3 wake -6.0": build_world(m_w=-6.0)}
SFS = [0.0, 0.4, 0.8]
R = 100
rng = np.random.default_rng(2024)
W = NULLS["W1a shoulder"]
set_D(W, TAU, rng)
Xc = replicates(W, 0.0, R, rng)
Vn, NE, _ = estimates(Xc)
Vf = np.array([deepest_valley(Xc[:, r, :])[0] for r in range(R)])
Vi = np.array([deepest_valley(W.sample_stationary(int(NE.mean()), rng))[0] for _ in range(R)])
print("n_eff %.0f | null 95th pct: n_eff bandwidth %.3f, full-n bandwidth %.3f, iid n_eff %.3f" % (NE.mean(), np.quantile(Vn, .95), np.quantile(Vf, .95), np.quantile(Vi, .95)))
q95 = {}
for name, W in NULLS.items():
    set_D(W, TAU, rng)
    for sf in SFS:
        V, NE, _ = estimates(replicates(W, sf, R, rng))
        q95[(name, sf)] = np.quantile(V, .95)
DELTA0 = max(q95.values())
print("detection threshold delta0 = %.3f" % DELTA0)
for group, worlds in [("null", NULLS), ("alt", ALTS)]:
    for name, W in worlds.items():
        set_D(W, TAU, rng)
        for sf in SFS:
            Wo = observed_world(W, sf)
            h_, r_, xv, _b = valley_info(Wo, W.m[0], 0.0) if group == "alt" else (False, np.nan, np.nan, 0)
            V, NE, SH = estimates(replicates(W, sf, R, rng), cut=xv if h_ else None)
            print("[%s] %-14s noise %.1f | P(V>delta0) %.2f | mean V %.3f | true V %s | wake share 5th pct %s" % (
                group, name, sf, np.mean(V > DELTA0), V.mean(), ("%.3f" % (1 - r_)) if h_ else "-", ("%.3f" % np.quantile(SH, .05)) if h_ else "-"))


section("PHASE 3  |  S6-S7: nuisance, alignment, clustered nulls")
N_SUBJ = 20
SUBJ = np.repeat(np.arange(N_SUBJ), N_NIGHTS // N_SUBJ)
SCEN = {"none": (0, 0, 0, 0, 0), "subject-level 0.3/0.3": (0.3, 0, 0.3, 0, 0), "night-level 0.15/0.15": (0, 0.15, 0, 0.15, 0),
        "both moderate": (0.3, 0.15, 0.3, 0.15, 0), "large + drift": (0.6, 0.3, 0.5, 0.25, 0.5)}


def nuisance(X, sa_s, sa_n, sb_s, sb_n, sd, rng):
    T_, R_, N = X.shape
    a = rng.normal(0, sa_s, (1, R_, N_SUBJ))[..., SUBJ] + rng.normal(0, sa_n, (1, R_, N))
    b = np.exp(rng.normal(0, sb_s, (1, R_, N_SUBJ))[..., SUBJ] + rng.normal(0, sb_n, (1, R_, N)))
    d = rng.normal(0, sd, (1, R_, N))
    t = (np.arange(T_)[:, None, None] / (T_ - 1)) - 0.5
    return a + b * X + d * t


def align_night(X):
    return (X - np.median(X, 0, keepdims=True)) / mad(X)


def align_night_median_subject_scale(X):
    Z = X - np.median(X, 0, keepdims=True)
    out = np.empty_like(Z)
    for s in range(N_SUBJ):
        j = SUBJ == s
        sc = mad(Z[:, :, j].transpose(0, 2, 1).reshape(-1, Z.shape[1]))
        out[:, :, j] = Z[:, :, j] / sc[:, :, None]
    return out


def align_subject(X):
    out = np.empty_like(X)
    for s in range(N_SUBJ):
        j = SUBJ == s
        P = X[:, :, j].transpose(0, 2, 1).reshape(-1, X.shape[1])
        out[:, :, j] = (X[:, :, j] - np.median(P, 0)[None, :, None]) / mad(P)[:, :, None]
    return out


ALIGN = {"raw": lambda z: z, "night med+MAD": align_night, "night med, subj scale": align_night_median_subject_scale, "subject med+MAD": align_subject}


def est_only(X):
    return np.array([deepest_valley(X[:, r, :], n_eff=X[:, r, :].size / iat_g(X[:, r, :]))[0] for r in range(X.shape[1])])


rng = np.random.default_rng(606)
WORLDS6 = [("null W0 Gaussian", MixtureWorld([1.0], [0.0], [1.0])), ("null W1a shoulder", build_world(m_w=-3.0, w_w=0.049)),
           ("null W1b skew", build_world(m_w=-2.5, w_w=0.15)), ("alt A2 wake -5.0", build_world(m_w=-5.0))]
OUT6 = {}
for name, W in WORLDS6:
    set_D(W, TAU, rng)
    X0 = replicates(W, 0.4, R, rng)
    for sc_, pars in SCEN.items():
        Xn = nuisance(X0, *pars, rng)
        for al, f in ALIGN.items():
            OUT6[(name, sc_, al)] = est_only(f(Xn))
for sc_ in SCEN:
    print("NULL %-22s | %s" % (sc_, " | ".join("%s %.3f/%.2f" % (al, max(np.quantile(OUT6[(n, sc_, al)], .95) for n, _ in WORLDS6[:3]), max(np.mean(OUT6[(n, sc_, al)] > 0.05) for n, _ in WORLDS6[:3])) for al in ALIGN)))
    print("POWER %-21s | %s" % (sc_, " | ".join("%s %.3f/%.2f" % (al, np.median(OUT6[("alt A2 wake -5.0", sc_, al)]), np.mean(OUT6[("alt A2 wake -5.0", sc_, al)] > 0.05)) for al in ALIGN)))


def cluster_shift(X, level, delta, rng):
    T_, R_, N = X.shape
    s = rng.choice([-0.5, 0.5], size=(1, R_, N_SUBJ))[..., SUBJ] if level == "subject" else rng.choice([-0.5, 0.5], size=(1, R_, N))
    return X + delta * s


SCEN7 = {"none": ("subject", 0.0), "subject sep 2": ("subject", 2.0), "subject sep 3": ("subject", 3.0), "night sep 2": ("night", 2.0), "night sep 3": ("night", 3.0)}
rng = np.random.default_rng(707)
NULLW = [("W0 Gaussian", MixtureWorld([1.0], [0.0], [1.0])), ("W1a shoulder", build_world(m_w=-3.0, w_w=0.049))]
OUT7 = {}
for name, W in NULLW:
    set_D(W, TAU, rng)
    X0 = replicates(W, 0.4, R, rng)
    for sc_, (lev, dl) in SCEN7.items():
        Xn = cluster_shift(X0, lev, dl, rng)
        for al, f in ALIGN.items():
            OUT7[(name, sc_, al)] = est_only(f(Xn))
for sc_ in SCEN7:
    print("clustered null %-14s | %s" % (sc_, " | ".join("%s %.2f" % (al, max(np.mean(OUT7[(n, sc_, al)] > 0.05) for n, _ in NULLW)) for al in ALIGN)))


section("PHASE 3  |  S8: statistic C (H3/H2 local valley depth at rater boundaries)")
THIN, B8, SIG_R = 5, 80, 0.3
W_LOC, DELTA_LOC, EPS = 1.0, 0.05, 0.10


def local_depth(c, p, b, w=W_LOC):
    ib = int(np.clip(np.searchsorted(c, b), 1, len(c) - 2))
    k = int(round(w / (c[1] - c[0])))
    mL = p[max(ib - k, 0): ib + 1].max()
    mR = p[ib: ib + k + 1].max()
    return max(0.0, 1 - p[ib] / min(mL, mR))


def boundary(x, y):
    m, s = x.mean(), x.std()
    z = (x - m) / s
    A_ = np.c_[np.ones_like(z), z]
    beta = np.zeros(2)
    for _ in range(25):
        pr = 1 / (1 + np.exp(-(A_ @ beta)))
        g = A_.T @ (y - pr) - 1e-3 * beta
        Hm = (A_ * (pr * (1 - pr))[:, None]).T @ A_ + 1e-3 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return m + s * (-beta[0] / beta[1])


def make_labels(x, rng, t_wake=None, t_n3=None, sig_r=SIG_R):
    z = x + rng.normal(0, sig_r, x.shape)
    t0 = np.quantile(z, SH1[0]) if t_wake is None else t_wake
    t3 = np.quantile(z, 1 - SH1[3]) if t_n3 is None else t_n3
    mid = z[(z >= t0) & (z < t3)]
    t1 = np.quantile(mid, SH1[1] / (SH1[1] + SH1[2]))
    return np.digitize(z, [t0, t1, t3])


def h3_stats(xs, ys, n_eff):
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    sd = min(x.std(), (np.quantile(x, .75) - np.quantile(x, .25)) / 1.34)
    h = 0.9 * sd * n_eff ** (-0.2)
    c, p, dx = kde_grid(x, h)
    bs = np.array([boundary(x, (y <= k).astype(float)) for k in range(3)])
    return np.array([local_depth(c, p, b) for b in bs]), bs


def study_h3(W, sf, thr, rng, R_):
    X = replicates(W, sf, R_, rng)
    res = []
    for r in range(R_):
        Xr = X[:, r, :]
        ne = Xr.size / iat_g(Xr)
        xs = [Xr[::THIN, SUBJ == s].ravel() for s in range(N_SUBJ)]
        lens = np.cumsum([len(x) for x in xs])[:-1]
        lab = np.split(make_labels(np.concatenate(xs), rng, **thr), lens)
        D_, bs = h3_stats(xs, lab, ne)
        Db = np.array([h3_stats([xs[i] for i in idx], [lab[i] for i in idx], ne)[0] for idx in (rng.integers(0, N_SUBJ, N_SUBJ) for _ in range(B8))])
        res.append((D_, bs, np.quantile(Db, 0.05, axis=0), np.quantile(Db, 0.95, axis=0)))
    return res


rng = np.random.default_rng(808)
R8 = 30
W2s = MixtureWorld([0.049, 0.69, 0.26], [-5.0, 0.0, 3.2], [0.6, 1.0, 0.6])
W3s = build_world(m_w=-5.0)
W1s = build_world(m_w=-3.0, w_w=0.049)
for W in (W1s, W2s, W3s):
    set_D(W, TAU, rng)
xv_w = valley_info(observed_world(W3s, 0.4), -5.0, 0.0)[2]
xv_n = valley_info(observed_world(W2s, 0.4), 0.0, 3.2)[2]
CASES8 = [("A wake valley, boundary offset %+.1f" % o, W3s, dict(t_wake=xv_w + o)) for o in [0.0, 0.5, 1.0, 1.5, -1.0, -1.5]]
CASES8 += [("C no valley, share quantile", W1s, dict()), ("D wake and N2|N3 valleys", W2s, dict(t_wake=xv_w, t_n3=xv_n))]
for name, W, thr in CASES8:
    out = study_h3(W, 0.4, thr, rng, R8)
    D_ = np.array([o[0] for o in out]); bs = np.array([o[1] for o in out]); lo = np.array([o[2] for o in out]); hi = np.array([o[3] for o in out])
    print("%-40s | local depth %s | boundaries %s | P(H3) %.2f | P(H2) %.2f" % (name, np.round(np.median(D_, 0), 3), np.round(np.median(bs, 0), 2), np.mean(lo[:, 0] > DELTA_LOC), np.mean((hi[:, 1] < EPS) & (hi[:, 2] < EPS))))


section("PHASE 3  |  S9: H4 sign-flip test for time asymmetry")


def ou2_nights(theta_ratio, R_, rng, T=T_EP, N=N_NIGHTS, sf=0.4, tau=TAU, trend=None):
    a = 2.0 / tau
    th = theta_ratio * a
    Mr = np.exp(-a) * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    q = np.sqrt(1 - np.exp(-2 * a))
    X = np.empty((T, R_, N, 2))
    X[0] = rng.standard_normal((R_, N, 2))
    for t in range(1, T):
        X[t] = X[t - 1] @ Mr.T + q * rng.standard_normal((R_, N, 2))
    X = X + rng.normal(0, sf, X.shape)
    s = np.arange(T) / (T - 1)
    if trend is not None:
        kind, mx, my = trend
        g = (s - 0.5) if kind == "linear" else (np.exp(-3 * s) - np.exp(-3 * s).mean()) / np.exp(-3 * s).std()
        X[..., 0] += mx * g[:, None, None]
        X[..., 1] += my * g[:, None, None]
    return X


def detrend_poly(X, deg):
    T_ = X.shape[0]
    A_ = np.polynomial.legendre.legvander(np.linspace(-1, 1, T_), deg)
    Y = X.reshape(T_, -1)
    return (Y - A_ @ np.linalg.lstsq(A_, Y, rcond=None)[0]).reshape(X.shape)


def detrend_blocks(X, L=120):
    out = X.copy()
    for k in range(0, X.shape[0], L):
        out[k:k + L] -= X[k:k + L].mean(0, keepdims=True)
    return out


def night_features(X):
    Z = (X - X.mean((0, 2), keepdims=True)) / X.std((0, 2), keepdims=True)
    x, y = Z[..., 0], Z[..., 1]
    f1 = (x[:-1] * y[1:] - y[:-1] * x[1:]).mean(0)
    f2 = (x[:-1] ** 2 * x[1:] - x[:-1] * x[1:] ** 2).mean(0)
    return np.stack([f1, f2], -1)


def signflip_p(F_, rng, n_perm=2000):
    S = F_.reshape(F_.shape[0], N_SUBJ, -1, 2).sum(2)
    sg = rng.choice([-1.0, 1.0], size=(n_perm, 1, N_SUBJ, 1))
    stat = lambda A_: (np.abs(A_.sum(-2)) / np.sqrt((A_ ** 2).sum(-2))).max(-1)
    return (1 + (stat(S[None] * sg) >= stat(S)[None]).sum(0)) / (1 + n_perm)


R9 = 200
rng = np.random.default_rng(909)
METHODS = [("raw", lambda X: X), ("linear", lambda X: detrend_poly(X, 1)), ("cubic", lambda X: detrend_poly(X, 3)), ("blocks 60 min", detrend_blocks)]
CASES9 = [("reversible", 0.0, None), ("current 0.05", 0.05, None), ("current 0.1", 0.1, None), ("current 0.25", 0.25, None),
          ("reversible + linear trend", 0.0, ("linear", 2.0, -1.0)), ("reversible + exp trend (1,-0.5)", 0.0, ("exp", 1.0, -0.5)),
          ("reversible + exp trend (2,-1)", 0.0, ("exp", 2.0, -1.0)), ("current 0.25 + exp trend", 0.25, ("exp", 2.0, -1.0))]
for name, thr, tr in CASES9:
    X = ou2_nights(thr, R9, rng, trend=tr)
    print("%-34s | %s" % (name, " | ".join("%s %.2f" % (m, np.mean(signflip_p(night_features(f(X)), rng) < 0.05)) for m, f in METHODS)))


section("PHASE 3  |  S10: statistic B (cross-validated mixture density) and bootstrap coverage")


def mixture_stats(xs, rng, K=5, kmax=3):
    folds = np.array_split(rng.permutation(len(xs)), K)
    ll = np.zeros((K, kmax))
    x_all = np.concatenate(xs)[:, None]
    for f, te in enumerate(folds):
        tr = np.concatenate([xs[i] for i in range(len(xs)) if i not in te])[:, None]
        tst = np.concatenate([xs[i] for i in te])[:, None]
        for k in range(1, kmax + 1):
            ll[f, k - 1] = GaussianMixture(k, n_init=2, random_state=0).fit(tr).score(tst) * len(tst)
    kbest = int(np.argmax(ll.sum(0))) + 1
    g = GaussianMixture(kbest, n_init=3, random_state=0).fit(x_all)
    c = np.linspace(x_all.min() - 1, x_all.max() + 1, 2048)
    p = np.exp(g.score_samples(c[:, None]))
    vs = valleys(c, p, c[1] - c[0], NU)
    return kbest, (1 - min(v / min(a, b) for _, v, a, b in vs)) if vs else 0.0


R10 = 60
rng = np.random.default_rng(1010)
CASES10 = [("null W0 Gaussian", MixtureWorld([1.0], [0.0], [1.0])), ("null W1a shoulder", build_world(m_w=-3.0, w_w=0.049)),
           ("null W1b skew", build_world(m_w=-2.5, w_w=0.15)), ("valley A1 wake -4.5", build_world(m_w=-4.5)), ("valley A2 wake -5.0", build_world(m_w=-5.0))]
for name, W in CASES10:
    set_D(W, TAU, rng)
    X = replicates(W, 0.4, R10, rng)
    res = np.array([mixture_stats([X[:, r, :][::THIN, SUBJ == s].ravel() for s in range(N_SUBJ)], rng) for r in range(R10)])
    print("%-22s | P(k>=2) %.2f | P(V_mix>0) %.2f | median V_mix %.3f" % (name, np.mean(res[:, 0] >= 2), np.mean(res[:, 1] > 0), np.median(res[:, 1])))
for name, W in [("valley A1 wake -4.5", build_world(m_w=-4.5)), ("valley A2 wake -5.0", build_world(m_w=-5.0))]:
    set_D(W, TAU, rng)
    Vtrue = 1 - valley_info(observed_world(W, 0.4), W.m[0], 0.0)[1]
    X = replicates(W, 0.4, 40, rng)
    est, cov, ints = [], [], []
    for r in range(40):
        Xr = X[:, r, :]
        ne = Xr.size / iat_g(Xr)
        xs = [Xr[::THIN, SUBJ == s].ravel() for s in range(N_SUBJ)]
        est.append(deepest_valley(np.concatenate(xs), n_eff=ne)[0])
        vb = np.array([deepest_valley(np.concatenate([xs[i] for i in rng.integers(0, N_SUBJ, N_SUBJ)]), n_eff=ne)[0] for _ in range(100)])
        lo, hi = np.quantile(vb, [0.05, 0.95])
        cov.append(lo <= Vtrue <= hi)
        ints.append((lo, hi))
    est, ints = np.array(est), np.array(ints)
    tgt = np.median(est)
    print("%-22s | true V %.3f | median V_hat %.3f | coverage true %.2f | coverage typical %.2f" % (name, Vtrue, tgt, np.mean(cov), np.mean((ints[:, 0] <= tgt) & (tgt <= ints[:, 1]))))


section("PHASE 4  |  P3: label-free depth axis (PC1), subject alignment, nuisance diagnostics")
SLEEP_CODES = (2, 3, 4, 5)
LONG_WAKE = 120
DEPTH = {1: 0, 3: 1, 4: 2, 5: 3}


def window_epochs(lab):
    sl = np.where(np.isin(lab, SLEEP_CODES))[0]
    if len(sl) == 0:
        return None
    run = np.zeros(len(lab), int)
    for i in range(len(lab)):
        run[i] = (run[i - 1] + 1 if i else 1) if lab[i] == 1 else 0
    cuts = np.where(run == LONG_WAKE)[0]
    grp = np.searchsorted(cuts, sl, side="right")
    s = sl[grp == np.bincount(grp).argmax()]
    return s.min(), s.max()


def prepare(df, name):
    keep = []
    for night, g in df[df["ch"] == "C3-M2"].sort_values(["night", "epoch"]).groupby("night"):
        w = window_epochs(g["scorer1"].to_numpy())
        if w is not None:
            keep.append(g.iloc[w[0]:w[1] + 1])
    d = pd.concat(keep)
    n0 = d["night"].nunique()
    d = d[d["include"].astype(bool) & d[BAND[0]].notna()].copy()
    d["sub"] = d["night"].str.split("_").str[0]
    print("%s: %d nights with a scored sleep window; %d with included C3 epochs; %d epochs" % (name, n0, d["night"].nunique(), len(d)))
    return d


def feature_matrix(d):
    Lg = np.log(d[BAND].to_numpy())
    clr = Lg - Lg.mean(1, keepdims=True)
    return np.c_[clr, np.log(d["sef90"].to_numpy()), d["slope"].to_numpy(), d["entropy"].to_numpy()]


def robust_z(X):
    med = np.median(X, 0)
    sc = np.array([mad(X[:, j]) for j in range(X.shape[1])])
    return (X - med) / sc


def fit_pc1(Z):
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    v = Vt[0]
    return (-v if v[0] < 0 else v), (S ** 2 / (S ** 2).sum())[:3]


def components(v, subj):
    g = pd.Series(np.asarray(v, float)).groupby(np.asarray(subj))
    n = g.size()
    ok = n[n >= 2].index
    if len(ok) == 0:
        return np.nan, np.nan
    within = np.mean([g.get_group(k).var(ddof=1) for k in ok])
    between = max(0.0, g.mean().var(ddof=1) - within / np.mean(n[n >= 2]))
    return np.sqrt(between), np.sqrt(within)


def nrem_unit(d, col="xa"):
    out = []
    for _, g in d.groupby("night"):
        m = g["scorer1"].isin([3, 4, 5])
        if m.sum() > 100 and (g["scorer1"] == 4).sum() > 50:
            out.append((g.loc[m, col] - g.loc[g["scorer1"] == 4, col].median()).std())
    return float(np.median(out))


def diagnostics(d, name, scorers=("scorer1",), B_=100, seed=0):
    rng = np.random.default_rng(seed)
    d = d.sort_values(["night", "epoch"]).copy()
    nrem = d["scorer1"].isin([3, 4, 5])
    print(name)
    for tag, col in [("raw axis", "x"), ("after subject alignment", "xa")]:
        u = nrem_unit(d, col)
        n2 = d[d["scorer1"] == 4].groupby("night")[col].agg(median="median", spread=mad, size="size")
        n2 = n2[n2["size"] >= 100]
        sub = n2.index.str.split("_").str[0]
        sh = components(n2["median"] / u, sub)
        sc = components(np.log(n2["spread"]), sub)
        print(" %-24s N2 shift sd (subject, night) %.2f, %.2f | log-scale sd %.2f, %.2f [unit %.3f]" % (tag, *sh, *sc, u))
    nx = d.groupby("night").shift(-1)
    pair = (nx["epoch"] == d["epoch"] + 1) & nrem & (nx["scorer1"] == d["scorer1"])
    dx = (nx["xa"] - d["xa"])[pair]
    p2 = pair & (d["scorer1"] == 4)
    print(" fast noise sigma_f/NREM sd %.2f (%.2f relative to N2 sd) | lag-1 autocorrelation %.2f" % (
        np.sqrt(0.5 * dx.var()) / mad(d.loc[nrem, "xa"]), np.sqrt(0.5 * (nx["xa"] - d["xa"])[p2].var()) / mad(d.loc[d["scorer1"] == 4, "xa"]),
        d.groupby("night")["xa"].apply(lambda s: s.autocorr(1)).median()))
    mu, sg = d.loc[nrem, "xa"].mean(), d.loc[nrem, "xa"].std()
    subs = d["sub"].unique()
    for sc_ in scorers:
        e = d[d[sc_].isin([1, 3, 4, 5])]
        lab = e[sc_].map(DEPTH).to_numpy()
        x = e["xa"].to_numpy()
        grp = {s: np.where(e["sub"].to_numpy() == s)[0] for s in subs}
        pos = np.array([boundary(x, (lab <= k).astype(float)) for k in range(3)])
        bs = []
        for _ in range(B_):
            idx = np.concatenate([grp[s] for s in rng.choice(subs, len(subs))])
            bs.append([boundary(x[idx], (lab[idx] <= k).astype(float)) for k in range(3)])
        se = np.std(bs, 0)
        print(" rater boundaries (%s), NREM-sd units from NREM mean (SE): %s" % (sc_, " | ".join("%.2f (%.2f)" % ((p - mu) / sg, s_ / sg) for p, s_ in zip(pos, se))))
    sp = d[d["scorer1"].isin(DEPTH)]
    print(" Spearman with stage order: %.3f" % spearmanr(sp["xa"], sp["scorer1"].map(DEPTH))[0])


D19 = prepare(pd.read_csv(f"{PROC}/features_eesm19_v1.csv.gz"), "EESM19")
D23 = prepare(pd.read_csv(f"{PROC}/features_eesm23_v1.csv.gz"), "EESM23")
Z19, Z23 = robust_z(feature_matrix(D19)), robust_z(feature_matrix(D23))
v, ev = fit_pc1(Z19)
print("PC1 explains %.1f%% (PC2 %.1f%%, PC3 %.1f%%); loadings %s" % (100 * ev[0], 100 * ev[1], 100 * ev[2], np.round(v, 3)))
D19["x"], D23["x"] = Z19 @ v, Z23 @ v
for Dd in (D19, D23):
    g = Dd.groupby("sub")["x"]
    Dd["xa"] = (Dd["x"] - g.transform("median")) / g.transform(lambda s: mad(s))
diagnostics(D19, "EESM19 PC1", scorers=("scorer1", "scorer2"))
diagnostics(D23, "EESM23 PC1")
D19[["night", "sub", "epoch", "onset", "scorer1", "scorer2", "x", "xa"]].to_csv(f"{PROC}/axis_eesm19_v1.csv.gz", index=False)
D23[["night", "sub", "epoch", "onset", "scorer1", "x", "xa"]].to_csv(f"{PROC}/axis_eesm23_v1.csv.gz", index=False)


section("PHASE 4  |  P4: sampling-error decomposition, axis benchmark, window rule")
rng4 = np.random.default_rng(4)
BL, NB = 40, 200


def sampling_var(g, col, u):
    x = g.sort_values("epoch")[col].to_numpy()
    n = len(x)
    nb = int(np.ceil(n / BL))
    blocks = [x[i * BL:(i + 1) * BL] for i in range(nb)]
    med, lm = [], []
    for _ in range(NB):
        s = np.concatenate([blocks[j] for j in rng4.integers(0, nb, nb)])[:n]
        med.append(np.median(s) / u)
        lm.append(np.log(mad(s)))
    return np.var(med), np.var(lm)


def decompose(d, name):
    for tag, col in [("raw axis", "x"), ("after subject alignment", "xa")]:
        u = nrem_unit(d, col)
        rows = [(night, g[col].median() / u, np.log(mad(g[col])), *sampling_var(g, col, u)) for night, g in d[d["scorer1"] == 4].groupby("night") if len(g) >= 100]
        Rr = pd.DataFrame(rows, columns=["night", "m", "ls", "vm", "vl"]).set_index("night")
        sub = Rr.index.str.split("_").str[0]
        sh, sc = components(Rr["m"], sub), components(Rr["ls"], sub)
        rem = lambda w, v_: np.sqrt(max(0.0, w ** 2 - v_))
        print("%s %-24s night shift sd observed %.2f, sampling %.2f, remaining %.2f | log-scale observed %.2f, sampling %.2f, remaining %.2f" % (
            name, tag, sh[1], np.sqrt(Rr["vm"].mean()), rem(sh[1], Rr["vm"].mean()), sc[1], np.sqrt(Rr["vl"].mean()), rem(sc[1], Rr["vl"].mean())))


def align_col(d, col):
    g = d.groupby("sub")[col]
    return (d[col] - g.transform("median")) / g.transform(lambda s: mad(s))


def benchmark(Dd, name):
    e = Dd[Dd["scorer1"].isin(DEPTH)].copy()
    y = e["scorer1"].map(DEPTH).to_numpy()
    Z = robust_z(feature_matrix(e))
    out = [("PC1 axis", spearmanr(e["xa"], y)[0])]
    for nm, j in [("relative delta (clr)", 0), ("log SEF90", 5), ("spectral slope", 6), ("spectral entropy", 7)]:
        e["_f"] = Z[:, j]
        out.append((nm, abs(spearmanr(align_col(e, "_f"), y)[0])))
    subs = e["sub"].unique()
    folds = np.array_split(rng4.permutation(subs), 4)
    sc = np.zeros(len(e))
    for te in folds:
        m = e["sub"].isin(te).to_numpy()
        s = LDA(n_components=1).fit(Z[~m], y[~m]).transform(Z[m])[:, 0]
        sc[m] = s if spearmanr(s, y[m])[0] >= 0 else -s
    e["_l"] = sc
    out.append(("supervised linear axis, held-out subjects", spearmanr(align_col(e, "_l"), y)[0]))
    if "scorer2" in e.columns:
        m2 = e["scorer2"].isin(list(DEPTH))
        out.append(("scorer 2 vs scorer 1", spearmanr(e.loc[m2, "scorer1"].map(DEPTH), e.loc[m2, "scorer2"].map(DEPTH))[0]))
    print(name, " | ".join("%s %.3f" % (nm, r) for nm, r in out))


def window_report(path, name):
    c3 = pd.read_csv(path, usecols=["night", "epoch", "ch", "scorer1"])
    c3 = c3[c3["ch"] == "C3-M2"].sort_values(["night", "epoch"])
    rows = []
    for night, g in c3.groupby("night"):
        lab = g["scorer1"].to_numpy()
        w = window_epochs(lab)
        if w is None:
            continue
        sl = np.where(np.isin(lab, SLEEP_CODES))[0]
        rows.append((night, (w[1] - w[0] + 1) / 120, len(lab) / 120, (sl.max() - sl.min() + 1) - (w[1] - w[0] + 1)))
    Rr = pd.DataFrame(rows, columns=["night", "win_h", "rec_h", "removed"])
    print("%s recording h %.1f/%.1f/%.1f | window h %.1f/%.1f/%.1f | nights trimmed by the 60-min rule %d (%d epochs) %s" % (
        name, Rr["rec_h"].min(), Rr["rec_h"].median(), Rr["rec_h"].max(), Rr["win_h"].min(), Rr["win_h"].median(), Rr["win_h"].max(),
        (Rr["removed"] > 0).sum(), Rr["removed"].sum(), Rr.loc[Rr["removed"] > 0, "night"].tolist()))


decompose(D19, "EESM19")
decompose(D23, "EESM23")
benchmark(D19, "EESM19")
benchmark(D23, "EESM23")
window_report(f"{PROC}/features_eesm19_v1.csv.gz", "EESM19")
window_report(f"{PROC}/features_eesm23_v1.csv.gz", "EESM23")


section("PHASE 4  |  S11: power and type-I rate at measured nuisance levels")
WAKE_W = [0.025, 0.049, 0.07, 0.10]


def trim_anchor_align(X, q=0.10):
    out = np.empty_like(X)
    for s in range(N_SUBJ):
        j = SUBJ == s
        P = X[:, :, j].transpose(0, 2, 1).reshape(-1, X.shape[1])
        A_ = np.where(P >= np.quantile(P, q, axis=0), P, np.nan)
        med = np.nanmedian(A_, 0)
        sc = 1.4826 * np.nanmedian(np.abs(A_ - med), 0)
        out[:, :, j] = (X[:, :, j] - med[None, :, None]) / sc[None, :, None]
    return out


def replicates_comp(m_w, weights, sf, R_, rng, dt=0.1):
    X = np.empty((T_EP, R_, N_NIGHTS))
    for k, w in enumerate(weights):
        W = build_world(m_w=m_w, w_w=w)
        set_D(W, TAU, rng)
        nights = np.where(np.isin(SUBJ, [s for s in range(N_SUBJ) if s % len(weights) == k]))[0]
        Xk = simulate(W, W.sample_stationary(R_ * len(nights), rng), T_EP, rng, dt=dt)
        Xk = Xk + rng.normal(0, sf, Xk.shape)
        X[:, :, nights] = Xk.reshape(T_EP, R_, len(nights))
    return X


SC11 = {"no nuisance": (0, 0, 0, 0, 0), "EESM19-like": (0.28, 0.12, 0.14, 0.14, 0.0), "EESM23-like": (0.27, 0.17, 0.38, 0.09, 0.0),
        "twice EESM19 + drift": (0.56, 0.24, 0.28, 0.28, 0.3)}
AL11 = {"raw": lambda z: z, "subject med+MAD": align_subject, "subject trimmed anchor": trim_anchor_align}
WORLDS11 = [("null shoulder -3.0", -3.0, WAKE_W, 0.4), ("null skew -2.5", -2.5, [3 * w for w in WAKE_W], 0.4), ("valley A1 -4.5", -4.5, WAKE_W, 0.4),
            ("valley A2 -5.0", -5.0, WAKE_W, 0.4), ("valley A2 -5.0 noise 0.8", -5.0, WAKE_W, 0.8)]
R11 = 60
rng = np.random.default_rng(1111)
for name, m_w, wts, sf in WORLDS11:
    X0 = replicates_comp(m_w, wts, sf, R11, rng)
    for sc_, pars in SC11.items():
        Xn = nuisance(X0, *pars, rng)
        print("%-26s | %-22s | %s" % (name, sc_, " | ".join("%s %.2f" % (al, np.mean(est_only(f(Xn)) > 0)) for al, f in AL11.items())))


section("PHASE 4  |  P5: stage positions on the aligned PC1 axis")
NAMES = {1: "Wake", 3: "N1", 4: "N2", 5: "N3", 2: "REM"}


def stage_report(d, name):
    u = nrem_unit(d)
    o = d.loc[d["scorer1"] == 4, "xa"].median()
    d = d.assign(z=(d["xa"] - o) / u)
    rows = [[lab, (d["scorer1"] == c).sum(), *d.loc[d["scorer1"] == c, "z"].quantile([.05, .25, .5, .75, .95]).to_list()] for c, lab in NAMES.items()]
    print("%s (NREM unit %.3f)\n%s" % (name, u, pd.DataFrame(rows, columns=["stage", "n", "q05", "q25", "median", "q75", "q95"]).round(2).to_string(index=False)))
    tot = d.groupby("sub").size()
    w = d[d["scorer1"] == 1].groupby("sub")["z"].agg(["size", "median"])
    w["share_%"] = 100 * w["size"] / tot.reindex(w.index)
    print(" wake share %% min/median/max %.1f/%.1f/%.1f | wake position %.2f/%.2f/%.2f" % (w["share_%"].min(), w["share_%"].median(), w["share_%"].max(), w["median"].min(), w["median"].median(), w["median"].max()))
    Lw = d.groupby("night")["epoch"].agg(lambda e: (e.max() - e.min() + 1) * 30 / 3600)
    print(" shortest windows (h):", Lw.nsmallest(5).round(2).to_dict())
    return d


stage_report(D19, "EESM19 PC1")
stage_report(D23, "EESM23 PC1")


section("PHASE 4  |  S12: when does a two-well world have any valley?")
M_GRID = np.round(np.arange(-0.5, -6.01, -0.1), 2)
S_W, W_W, SIG_F = [0.6, 0.83, 1.2], [0.025, 0.06, 0.10], 0.4


def first_valley(w, s, sig):
    for m in M_GRID:
        Wm = observed_world(build_world(m_w=float(m), s_w=s, w_w=w), sig) if sig > 0 else build_world(m_w=float(m), s_w=s, w_w=w)
        has, ratio, pos, barrier = valley_info(Wm, float(m), 0.0)
        if has:
            return m, ratio, barrier
    return None


for s in S_W:
    for w in W_W:
        a, b = first_valley(w, s, 0.0), first_valley(w, s, SIG_F)
        f = lambda r: "none to -6.0" if r is None else "m_w <= %.1f (ratio %.2f)" % (r[0], r[1])
        print("wake sd %.2f weight %.3f | latent %s | noise 0.4 %s" % (s, w, f(a), f(b)))
for s in S_W:
    for w in W_W:
        print("measured position -1.2: wake sd %.2f weight %.3f -> valley %s" % (s, w, valley_info(observed_world(build_world(m_w=-1.2, s_w=s, w_w=w), SIG_F), -1.2, 0.0)[0]))


section("PHASE 4  |  P6-P9: inside vs outside the window, separability ceiling, candidate axes, slope axis")
F19raw = pd.read_csv(f"{PROC}/features_eesm19_v1.csv.gz")
A = F19raw[(F19raw["ch"] == "C3-M2") & F19raw["include"].astype(bool) & F19raw[BAND[0]].notna()].copy()
A["sub"] = A["night"].str.split("_").str[0]
Xw = feature_matrix(D19)
med19 = np.median(Xw, 0)
sc19 = np.array([mad(Xw[:, j]) for j in range(Xw.shape[1])])
vpc1, _ = fit_pc1((Xw - med19) / sc19)
A["x"] = ((feature_matrix(A) - med19) / sc19) @ vpc1
g = D19.groupby("sub")["x"]
ms, ss = g.median(), g.apply(lambda s: mad(s))
A = A[A["sub"].isin(ms.index)].copy()
A["xa"] = (A["x"] - A["sub"].map(ms)) / A["sub"].map(ss)
key = D19.set_index(["night", "epoch"])["xa"]
A["inwin"] = A.set_index(["night", "epoch"]).index.isin(key.index)
chk = A[A["inwin"]].set_index(["night", "epoch"])["xa"].reindex(key.index)
print("check vs P3 axis: max |diff| %.2e" % np.nanmax(np.abs(chk.to_numpy() - key.to_numpy())))
u = nrem_unit(D19, "xa")
o = D19.loc[D19["scorer1"] == 4, "xa"].median()
A["z"] = (A["xa"] - o) / u
for inw in (True, False):
    for c, lab in NAMES.items():
        zz = A.loc[(A["inwin"] == inw) & (A["scorer1"] == c), "z"]
        if len(zz) >= 20:
            print("%s %-5s n=%5d median %+.2f (q05 %+.2f, q95 %+.2f)" % ("in window" if inw else "outside  ", lab, len(zz), zz.median(), zz.quantile(.05), zz.quantile(.95)))

Zs = (feature_matrix(A) - med19) / sc19
Zw = (Xw - med19) / sc19
gm_ = pd.DataFrame(Zw).groupby(D19["sub"].to_numpy())
msub, ssub = gm_.median(), gm_.agg(lambda s: mad(s.to_numpy()))
Zs = (Zs - msub.loc[A["sub"].to_numpy()].to_numpy()) / ssub.loc[A["sub"].to_numpy()].to_numpy()
Bw = A[A["scorer1"].isin([1, 3, 4, 5])].copy()
pos_ = A.index.get_indexer(Bw.index)
yw = (Bw["scorer1"] == 1).astype(int).to_numpy()
Xf = Zs[pos_]
pred = np.zeros(len(Bw))
for tr, te in GroupKFold(5).split(Xf, yw, Bw["sub"].to_numpy()):
    pred[te] = LDA().fit(Xf[tr], yw[tr]).decision_function(Xf[te])
nrem_in = (yw == 0) & Bw["inwin"].to_numpy()
auc_w = lambda mw, s: roc_auc_score(yw[mw | nrem_in], s[mw | nrem_in])
w_in, w_out = (yw == 1) & Bw["inwin"].to_numpy(), (yw == 1) & ~Bw["inwin"].to_numpy()
print("P7 AUC wake vs NREM | in window: PC1 %.3f, supervised %.3f | outside: PC1 %.3f, supervised %.3f" % (
    auc_w(w_in, -Bw["xa"].to_numpy()), auc_w(w_in, pred), auc_w(w_out, -Bw["xa"].to_numpy()), auc_w(w_out, pred)))
cf = LDA().fit(Xf, yw).coef_[0]
print("supervised direction:", np.round(cf / np.linalg.norm(cf), 3))

w_, V_ = np.linalg.eigh(np.cov(Zw.T))
o_ = np.argsort(w_)[::-1]
V_ = V_[:, o_]
ev_ = w_[o_] / w_.sum()
V_ = V_ * np.sign(V_[0:1, :])
print("P8 explained variance PC1-PC4 %s | PC1 equals P3 (|dot| %.4f)" % (np.round(ev_[:4], 3), abs(V_[:, 0] @ vpc1)))
Xall = (feature_matrix(A) - med19) / sc19
sub_all = A["sub"].to_numpy()
win = A["inwin"].to_numpy()
sc1 = A["scorer1"].to_numpy()
cols = {"PC1": Xall @ V_[:, 0], "PC2": Xall @ V_[:, 1], "PC3": Xall @ V_[:, 2], "PC4": Xall @ V_[:, 3], "slope": Xall[:, 6], "entropy": Xall[:, 7], "log SEF90": Xall[:, 5]}
for name, s in cols.items():
    s = pd.Series(s, index=A.index)
    wm = pd.Series(s[win].to_numpy(), index=A.index[win]).groupby(sub_all[win])
    sa = (s.to_numpy() - pd.Series(sub_all).map(wm.median()).to_numpy()) / pd.Series(sub_all).map(wm.apply(lambda z: mad(z.to_numpy()))).to_numpy()
    nrem_w = win & np.isin(sc1, [3, 4, 5])

    def a8(wk):
        m = wk | nrem_w
        r = roc_auc_score((sc1[m] == 1).astype(int), sa[m])
        return max(r, 1 - r)

    md = win & np.isin(sc1, list(DEPTH))
    print("P8 %-10s AUC wake-in %.3f | wake-out %.3f | |Spearman| %.3f" % (name, a8(win & (sc1 == 1)), a8(~win & (sc1 == 1)), abs(spearmanr(sa[md], pd.Series(sc1[md]).map(DEPTH))[0])))

s_raw = -Xall[:, 6]
sw = pd.Series(s_raw[win], index=A.index[win]).groupby(sub_all[win])
A2 = A[["night", "epoch", "sub", "scorer1", "inwin"]].copy()
A2["xa"] = (s_raw - pd.Series(sub_all).map(sw.median()).to_numpy()) / pd.Series(sub_all).map(sw.apply(lambda z: mad(z.to_numpy()))).to_numpy()
W2_ = A2[A2["inwin"]]
u2 = nrem_unit(W2_, "xa")
o2 = W2_.loc[W2_["scorer1"] == 4, "xa"].median()
A2["z"] = (A2["xa"] - o2) / u2
print("P9 slope axis NREM unit %.3f" % u2)
for inw in (True, False):
    for c, lab in NAMES.items():
        zz = A2.loc[(A2["inwin"] == inw) & (A2["scorer1"] == c), "z"]
        if len(zz) >= 20:
            print("%s %-5s n=%5d median %+.2f (q05 %+.2f, q95 %+.2f)" % ("in window" if inw else "outside  ", lab, len(zz), zz.median(), zz.quantile(.05), zz.quantile(.95)))


section("PHASE 4  |  P10: EMG check on the slope axis")
files_v2 = {os.path.basename(f).replace(".npz", ""): f for f in glob.glob(f"{V2}/*.npz") if ".part" not in f}
Aw = A[A["inwin"] & A["scorer1"].isin([1, 3, 4, 5])].copy()
Aw["onset"] = F19raw.loc[Aw.index, "onset"].to_numpy()
Aw["slope_z"] = A2["z"].loc[Aw.index].to_numpy()
emg = np.full(len(Aw), np.nan)
cache = {}
for k, row in enumerate(Aw[["night", "onset"]].itertuples(index=False)):
    if row.night not in cache:
        cache = {row.night: np.load(files_v2[row.night])["emg100"] if row.night in files_v2 else None}
    e = cache[row.night]
    if e is not None:
        i0 = int(round(row.onset * 100))
        seg = e[i0:i0 + 3000]
        if len(seg) == 3000 and np.isfinite(seg).all():
            emg[k] = np.log(np.sqrt(np.mean((seg - np.median(seg)) ** 2)) + 1e-9)
Aw["emg"] = emg
Aw = Aw[Aw["emg"].notna()].copy()
Aw["emg_a"] = Aw["emg"] - Aw.groupby("sub")["emg"].transform("median")
ye = (Aw["scorer1"] == 1).astype(int).to_numpy()
print("AUC wake vs NREM: EMG %.3f | slope %.3f | wake-NREM EMG difference %.2f" % (roc_auc_score(ye, Aw["emg_a"]), roc_auc_score(ye, -Aw["slope_z"]), Aw.loc[ye == 1, "emg_a"].median() - Aw.loc[ye == 0, "emg_a"].median()))
print("rank correlation slope-EMG: wake %.2f, NREM %.2f" % (spearmanr(Aw.loc[ye == 1, "slope_z"], Aw.loc[ye == 1, "emg_a"])[0], spearmanr(Aw.loc[ye == 0, "slope_z"], Aw.loc[ye == 0, "emg_a"])[0]))
for q in (0.9, 0.75, 0.5):
    m = (Aw["emg_a"] <= Aw.loc[ye == 0, "emg_a"].quantile(q)).to_numpy()
    print("EMG <= NREM q%.0f (%d wake): slope AUC %.3f" % (100 * q, (m & (ye == 1)).sum(), roc_auc_score(ye[m], -Aw["slope_z"].to_numpy()[m])))


section("PHASE 4  |  P11-P12: EESM23 replication of the axis checks; nuisance on the slope axis")
Z23f = robust_z(feature_matrix(D23))
sl23 = -Z23f[:, 6]
E = D23[["night", "sub", "epoch", "scorer1", "xa"]].copy().reset_index(drop=True)
E["sl"] = sl23
g = E.groupby("sub")["sl"]
E["xs"] = (E["sl"] - g.transform("median")) / g.transform(lambda s: mad(s.to_numpy()))
u3 = nrem_unit(E, "xs")
o3 = E.loc[E["scorer1"] == 4, "xs"].median()
E["z"] = (E["xs"] - o3) / u3
for c, lab in NAMES.items():
    zz = E.loc[E["scorer1"] == c, "z"]
    print("EESM23 slope %-5s n=%5d median %+.2f (q05 %+.2f, q95 %+.2f)" % (lab, len(zz), zz.median(), zz.quantile(.05), zz.quantile(.95)))
wk = E["scorer1"] == 1
m_ = (wk | E["scorer1"].isin([3, 4, 5])).to_numpy()
dd_ = lambda s: (lambda a: max(a, 1 - a))(roc_auc_score(wk.to_numpy()[m_].astype(int), np.asarray(s)[m_]))
md = E["scorer1"].isin(list(DEPTH))
print("EESM23 AUC wake vs NREM: PC1 %.3f | slope %.3f | |Spearman| PC1 %.3f slope %.3f" % (dd_(E["xa"]), dd_(E["xs"]), abs(spearmanr(E.loc[md, "xa"], E.loc[md, "scorer1"].map(DEPTH))[0]), abs(spearmanr(E.loc[md, "xs"], E.loc[md, "scorer1"].map(DEPTH))[0])))
K_ = ["night", "epoch"]
D19s = D19.drop(columns=["x", "xa"]).merge(A2.loc[A2["inwin"], K_ + ["xa"]].assign(x=s_raw[win]), on=K_)
D23s = D23.reset_index(drop=True).copy()
D23s["x"] = sl23
D23s["xa"] = E["xs"].to_numpy()
assert len(D19s) == len(D19)
diagnostics(D19s, "EESM19 slope", scorers=("scorer1", "scorer2"))
diagnostics(D23s, "EESM23 slope")


section("PHASE 4  |  S13: a world fitted to the real in-window wake distribution")
SF13 = 0.34
wz = A2.loc[A2["inwin"] & (A2["scorer1"] == 1), "z"].to_numpy().reshape(-1, 1)
gm13 = GaussianMixture(2, random_state=0, n_init=5).fit(wz)
wk_w, wk_m, wk_s = gm13.weights_, gm13.means_.ravel(), np.sqrt(gm13.covariances_.ravel())
wk_sl = np.sqrt(np.maximum(wk_s ** 2 - SF13 ** 2, 0.25 ** 2))
print("fitted wake: weights %s means %s sds %s latent sds %s" % (np.round(wk_w, 2), np.round(wk_m, 2), np.round(wk_s, 2), np.round(wk_sl, 2)))
build2 = lambda w: MixtureWorld(np.r_[1 - w, w * wk_w], np.r_[0.0, wk_m], np.r_[1.0, wk_sl], name="fitted wake w=%.3f" % w)
WTS = [0.01, 0.025, 0.04, 0.098]
for w in WTS + [float(np.mean(WTS))]:
    print("weight %.3f -> observed valley %s" % (w, valley_info(observed_world(build2(w), SF13), -7.0, 0.0)[0]))


def replicates2(R_, rng, dt=0.1):
    X = np.empty((T_EP, R_, N_NIGHTS))
    for k, w in enumerate(WTS):
        W = build2(w)
        set_D(W, TAU, rng)
        nights = np.where(np.isin(SUBJ, [s for s in range(N_SUBJ) if s % len(WTS) == k]))[0]
        Xk = simulate(W, W.sample_stationary(R_ * len(nights), rng), T_EP, rng, dt=dt)
        Xk = Xk + rng.normal(0, SF13, Xk.shape)
        X[:, :, nights] = Xk.reshape(T_EP, R_, len(nights))
    return X


SC13 = {"no nuisance": (0, 0, 0, 0, 0), "EESM19 slope-like": (0.32, 0.18, 0.20, 0.16, 0.0), "EESM23 slope-like": (0.25, 0.20, 0.17, 0.07, 0.0)}
rng = np.random.default_rng(1313)
X0_13 = replicates2(60, rng)
for sc_, pars in SC13.items():
    Xn = nuisance(X0_13, *pars, rng)
    print("fitted-wake null %-18s | %s" % (sc_, " | ".join("%s %.2f" % (al, np.mean(est_only(f(Xn)) > 0)) for al, f in AL11.items())))


section("PHASE 4  |  S14: autocorrelation time for ragged nights")


def iat_ragged(nights, min_frac=0.25):
    mu = np.nanmean(np.concatenate(nights))
    L = max(len(a) for a in nights)
    num = np.zeros(L)
    den = np.zeros(L)
    for a in nights:
        n = len(a)
        ok = np.isfinite(a)
        z = np.where(ok, a - mu, 0.0)
        f = np.fft.rfft(z, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        g = np.fft.rfft(ok.astype(float), 2 * n)
        cnt = np.rint(np.fft.irfft(g * np.conj(g))[:n])
        rho_ = (ac / np.maximum(cnt, 1)) / (ac[0] / cnt[0])
        num[:n] += cnt * rho_
        den[:n] += cnt
    ac = num / np.maximum(den, 1)
    Lmax = int(np.max(np.where(den >= min_frac * den[0])[0]))
    tau = 1.0
    for k in range(1, Lmax - 1, 2):
        pair = ac[k] + ac[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    return tau


rng14 = np.random.default_rng(1414)
res14 = {"full": [], "truncated": [], "gaps 3%": [], "truncated + gaps 3%": []}
for r in range(10):
    X = X0_13[:, r, :]
    base = iat_g(X)
    T_, N = X.shape
    full = [X[:, j] for j in range(N)]
    trunc = [X[:int(rng14.integers(300, T_ + 1)), j] for j in range(N)]
    gaps = [np.where(rng14.random(T_) < 0.03, np.nan, X[:, j]) for j in range(N)]
    both = [np.where(rng14.random(len(a)) < 0.03, np.nan, a) for a in trunc]
    for k, nights in zip(res14, (full, trunc, gaps, both)):
        res14[k].append(iat_ragged(nights) / base)
for k, v_ in res14.items():
    print("%-22s ratio %.3f (min %.3f, max %.3f) %s" % (k, np.mean(v_), np.min(v_), np.max(v_), "PASS" if 0.90 <= np.mean(v_) <= 1.10 else "FAIL"))


section("PHASE 4  |  P13: autocorrelation time of the real nights")


def night_arrays(d, col):
    out = []
    for n, g in d.groupby("night"):
        g = g.sort_values("epoch")
        e = g["epoch"].to_numpy().astype(int)
        a = np.full(e.max() - e.min() + 1, np.nan)
        a[e - e.min()] = g[col].to_numpy()
        out.append(a)
    return out


for dname, dpc, dsl in [("EESM19", D19, D19s), ("EESM23", D23, D23s)]:
    for axis, d in [("PC1", dpc), ("slope", dsl)]:
        na = night_arrays(d, "xa")
        tau = iat_ragged(na)
        n = int(sum(np.isfinite(a).sum() for a in na))
        print("%s %-5s nights %d epochs %d tau %.1f n_eff %.0f (%.1f per night)" % (dname, axis, len(na), n, tau, n / tau, n / tau / len(na)))


section("PHASE 4  |  S15: false-positive rate and power at EESM23's size")
SAVE = dict(N_NIGHTS=N_NIGHTS, N_SUBJ=N_SUBJ, SUBJ=SUBJ, T_EP=T_EP, TAU=TAU)
nights23 = D23.groupby("night")["sub"].first()
SUBJ = pd.factorize(nights23.to_numpy())[0]
N_NIGHTS, N_SUBJ, T_EP, TAU, SF15 = len(SUBJ), int(SUBJ.max()) + 1, 808, 50.0, 0.30
SC15 = {"no nuisance": (0, 0, 0, 0, 0), "EESM23 slope-like": (0.25, 0.20, 0.17, 0.07, 0.0)}
rng = np.random.default_rng(1515)
SF13_SAVE = SF13
SF13 = SF15
for name, make in [("null: fitted real wake", lambda: replicates2(60, rng)), ("valley A1 -4.5", lambda: replicates_comp(-4.5, [0.025, 0.049, 0.07, 0.10], SF15, 60, rng)), ("valley A2 -5.0", lambda: replicates_comp(-5.0, [0.025, 0.049, 0.07, 0.10], SF15, 60, rng))]:
    X0_ = make()
    for sc_, pars in SC15.items():
        Xn = nuisance(X0_, *pars, rng)
        print("%-24s | %-18s | %s" % (name, sc_, " | ".join("%s %.2f" % (al, np.mean(est_only(f(Xn)) > 0)) for al, f in AL11.items())))
SF13 = SF13_SAVE
N_NIGHTS, N_SUBJ, SUBJ, T_EP, TAU = SAVE["N_NIGHTS"], SAVE["N_SUBJ"], SAVE["SUBJ"], SAVE["T_EP"], SAVE["TAU"]


section("PHASE 5  |  R1: statistic A on real data (label-free valley depth)")


def statA(d, subs=None):
    if subs is not None:
        d = pd.concat([d[d["sub"] == s].assign(night=d.loc[d["sub"] == s, "night"] + "_b%d" % i) for i, s in enumerate(subs)])
    x = d["xa"].to_numpy()
    tau = iat_ragged(night_arrays(d, "xa"))
    V, pos, h = deepest_valley(x, n_eff=len(x) / tau)
    return V, pos, h, tau, x


SETS = [("EESM19", "slope", D19s), ("EESM19", "PC1", D19), ("EESM23", "slope", D23s), ("EESM23", "PC1", D23)]
rng = np.random.default_rng(2025)
for dname, axn, d in SETS:
    o_ = d.loc[d["scorer1"] == 4, "xa"].median()
    u_ = nrem_unit(d, "xa")
    V, pos, h, tau, x = statA(d)
    c, p, dx = kde_grid(x, h)
    pk = [i for i in argrelextrema(p, np.greater)[0] if p[i] > 0.01 * p.max()]
    subs = d["sub"].unique()
    Vb = np.array([statA(d, rng.choice(subs, len(subs)))[0] for _ in range(200)])
    print("%s %-5s n %d tau %.1f n_eff %.0f h %.3f | V_hat %.3f | peaks %s | bootstrap P(V>0) %.2f, 95th %.3f" % (
        dname, axn, len(x), tau, len(x) / tau, h, V, ", ".join("%.2f (%.2f)" % ((c[i] - o_) / u_, p[i] / p.max()) for i in pk), np.mean(Vb > 0), np.percentile(Vb, 95)))


section("PHASE 5  |  R2: statistic C on real data (H3 and H2 rules)")
THIN = 5
LAB4 = {1: 0, 3: 1, 4: 2, 5: 3}


def parts_of(d, scorer):
    d = d.sort_values(["night", "epoch"])
    o_ = d.loc[d["scorer1"] == 4, "xa"].median()
    u_ = nrem_unit(d, "xa")
    out = []
    for s, g in d.groupby("sub"):
        g = g.iloc[::THIN]
        z = ((g["xa"] - o_) / u_).to_numpy()
        lab = g[scorer].map(LAB4)
        m = lab.notna().to_numpy()
        out.append((z, z[m], lab[m].to_numpy().astype(int)))
    return out


def stats_real(parts, ne):
    x = np.concatenate([p[0] for p in parts])
    xl = np.concatenate([p[1] for p in parts])
    yl = np.concatenate([p[2] for p in parts])
    sd = min(x.std(), (np.quantile(x, .75) - np.quantile(x, .25)) / 1.34)
    h = 0.9 * sd * ne ** (-0.2)
    c, p, dx = kde_grid(x, h)
    bs = np.array([boundary(xl, (yl <= k).astype(float)) for k in range(3)])
    return np.array([local_depth(c, p, b) for b in bs]), bs


def run_statC(d, scorer, B_, rng):
    na = night_arrays(d, "xa")
    ne = sum(np.isfinite(a).sum() for a in na) / iat_ragged(na)
    parts = parts_of(d, scorer)
    D_, bs = stats_real(parts, ne)
    Db = np.array([stats_real([parts[i] for i in rng.integers(0, len(parts), len(parts))], ne)[0] for _ in range(B_)])
    lo, hi = np.quantile(Db, 0.05, 0), np.quantile(Db, 0.95, 0)
    return D_, bs, lo, hi, lo[0] > DELTA_LOC, (hi[1] < EPS) and (hi[2] < EPS)


rng = np.random.default_rng(3030)
for dname, axn, d in SETS:
    for sc_ in (("scorer1", "scorer2") if dname == "EESM19" else ("scorer1",)):
        D_, bs, lo, hi, h3, h2 = run_statC(d, sc_, 400, rng)
        print("%s %-5s %s | boundaries %s | local depths %s | H3 %s | H2 %s" % (dname, axn, sc_, " ".join("%+.2f" % b for b in bs), " ".join("%.3f [%.3f, %.3f]" % (D_[k], lo[k], hi[k]) for k in range(3)), "SUPPORTED" if h3 else "not supported", "SUPPORTED" if h2 else "not supported"))


section("PHASE 5  |  R3 (exploratory): statistic A including out-of-window wake")


def statA_all(d, subs=None, col="z"):
    if subs is not None:
        d = pd.concat([d[d["sub"] == s].assign(night=d.loc[d["sub"] == s, "night"] + "_b%d" % i) for i, s in enumerate(subs)])
    x = d[col].to_numpy()
    tau = iat_ragged(night_arrays(d, col))
    V, pos, h = deepest_valley(x, n_eff=len(x) / tau)
    return V, pos, h, tau, x


AllS = A2[["night", "epoch", "sub", "scorer1", "inwin", "z"]].copy()
AllP = A[["night", "epoch", "sub", "scorer1", "inwin", "z"]].copy()
rng = np.random.default_rng(4040)
for axn, d in [("slope", AllS), ("PC1", AllP)]:
    V, pos, h, tau, x = statA_all(d)
    subs = d["sub"].unique()
    Vb = np.array([statA_all(d, rng.choice(subs, len(subs)))[0] for _ in range(200)])
    print("EESM19 %-5s all epochs n %d (wake %.1f%%) tau %.1f n_eff %.0f | V_hat %.3f | bootstrap P(V>0) %.2f, 95th %.3f" % (axn, len(x), 100 * (d["scorer1"] == 1).mean(), tau, len(x) / tau, V, np.mean(Vb > 0), np.percentile(Vb, 95)))


section("PHASE 5  |  R4: statistic B on real data (cross-validated mixture density)")
rng = np.random.default_rng(5050)
xs_window = lambda d: [p[0] for p in parts_of(d, "scorer1")]
xs_all = lambda d: [g["z"].to_numpy()[::THIN] for _, g in d.sort_values(["night", "epoch"]).groupby("sub")]
for name, xs in [("EESM19 slope, in window", xs_window(D19s)), ("EESM19 PC1, in window", xs_window(D19)), ("EESM23 slope, in window", xs_window(D23s)),
                 ("EESM23 PC1, in window", xs_window(D23)), ("EESM19 slope, all epochs", xs_all(AllS)), ("EESM19 PC1, all epochs", xs_all(AllP))]:
    k0, v0 = mixture_stats(xs, rng)
    bb = np.array([mixture_stats([xs[i] for i in rng.integers(0, len(xs), len(xs))], rng) for _ in range(100)])
    print("%-26s | k %d | V_mix %.3f | P(k>=2) %.2f | P(V_mix>0) %.2f" % (name, k0, v0, np.mean(bb[:, 0] >= 2), np.mean(bb[:, 1] > 0)))


section("PHASE 5  |  R5: H4 time asymmetry on real data (sign-flip test)")


def aligned(vv, subs_, mask):
    s = pd.Series(vv)
    ms_ = s[mask].groupby(subs_[mask]).median()
    ss_ = s[mask].groupby(subs_[mask]).apply(lambda z: mad(z.to_numpy()))
    return (vv - pd.Series(subs_).map(ms_).to_numpy()) / pd.Series(subs_).map(ss_).to_numpy()


sub19 = A2["sub"].to_numpy()
win19 = A2["inwin"].to_numpy().astype(bool)
y19 = aligned(Xall @ V_[:, 1], sub19, win19)
F19 = A2.assign(y=y19)[A2["inwin"]][["night", "epoch", "sub", "scorer1", "z", "y"]].copy()
F23 = E[["night", "epoch", "sub", "scorer1", "z"]].copy()
F23["y"] = aligned(Z23f @ V_[:, 1], E["sub"].to_numpy(), np.ones(len(E), bool))


def movavg(vv, Wn):
    ok = np.isfinite(vv)
    k = np.ones(Wn)
    return np.convolve(np.where(ok, vv, 0.0), k, "same") / np.maximum(np.convolve(ok.astype(float), k, "same"), 1.0)


def h4_features(df, deg=3, nrem_only=False, min_pairs=100, W=None):
    parts = []
    for n, g in df.sort_values(["night", "epoch"]).groupby("night"):
        e = g["epoch"].to_numpy().astype(int)
        t = 2 * (e - e.min()) / max(e.max() - e.min(), 1) - 1
        Aleg = np.polynomial.legendre.legvander(t, deg) if deg is not None else np.ones((len(e), 1))
        res = []
        for col in ("z", "y"):
            vv = g[col].to_numpy()
            r = vv - Aleg @ np.linalg.lstsq(Aleg, vv, rcond=None)[0]
            if W is not None:
                grid = np.full(e.max() - e.min() + 1, np.nan)
                grid[e - e.min()] = r
                r = r - movavg(grid, W)[e - e.min()]
            res.append(r)
        parts.append((n, g["sub"].iloc[0], e, res[0], res[1], g["scorer1"].to_numpy()))
    sx = np.std(np.concatenate([p[3] for p in parts]))
    sy = np.std(np.concatenate([p[4] for p in parts]))
    rows = []
    for n, s, e, x, y, sc in parts:
        x, y = x / sx, y / sy
        ok = (e[1:] == e[:-1] + 1)
        if nrem_only:
            ok &= np.isin(sc[:-1], [3, 4, 5]) & np.isin(sc[1:], [3, 4, 5])
        if ok.sum() < min_pairs:
            continue
        x0, x1, y0, y1 = x[:-1][ok], x[1:][ok], y[:-1][ok], y[1:][ok]
        rows.append((n, s, (x0 * y1 - y0 * x1).mean(), (x0 ** 2 * x1 - x0 * x1 ** 2).mean(), ok.sum()))
    return pd.DataFrame(rows, columns=["night", "sub", "f1", "f2", "pairs"])


def signflip(F_, rng, n_perm=20000):
    S = F_.groupby("sub")[["f1", "f2"]].sum().to_numpy()
    stat = lambda A_: (np.abs(A_.sum(-2)) / np.sqrt((A_ ** 2).sum(-2))).max(-1)
    sg = rng.choice([-1.0, 1.0], size=(n_perm, len(S), 1))
    return (1 + (stat(S[None] * sg) >= stat(S)).sum()) / (1 + n_perm), S


def signflip_each(F_, rng, n_perm=20000):
    S = F_.groupby("sub")[["f1", "f2"]].sum().to_numpy()
    t = lambda M_: np.abs(M_.sum(-2)) / np.sqrt((M_ ** 2).sum(-2))
    sg = rng.choice([-1.0, 1.0], size=(n_perm, len(S), 1))
    return (1 + (t(S[None] * sg) >= t(S)).sum(0)) / (1 + n_perm), S


def h4_line(label, F_, p, S):
    print("%s | subjects %d nights %d | p %.4f | sums f1 %+.3f f2 %+.3f | positive %.2f / %.2f" % (label, F_["sub"].nunique(), len(F_), p, S[:, 0].sum(), S[:, 1].sum(), np.mean(S[:, 0] > 0), np.mean(S[:, 1] > 0)))


rng = np.random.default_rng(6060)
for dname, df in [("EESM19", F19), ("EESM23", F23)]:
    for vname, kw in [("cubic detrend (primary)", dict(deg=3)), ("cubic + NREM-only pairs", dict(deg=3, nrem_only=True)), ("no detrend", dict(deg=None))]:
        F_ = h4_features(df, **kw)
        p, S = signflip(F_, rng)
        h4_line("%s %-26s" % (dname, vname), F_, p, S)


section("PHASE 5  |  R6: where the asymmetry comes from; leave-one-subject-out")
NAMES6 = {1: "Wake", 2: "REM", 3: "N1", 4: "N2", 5: "N3"}


def steps6(df, deg=3):
    parts = []
    for n, g in df.sort_values(["night", "epoch"]).groupby("night"):
        e = g["epoch"].to_numpy().astype(int)
        t = 2 * (e - e.min()) / max(e.max() - e.min(), 1) - 1
        Aleg = np.polynomial.legendre.legvander(t, deg)
        res = [g[col].to_numpy() - Aleg @ np.linalg.lstsq(Aleg, g[col].to_numpy(), rcond=None)[0] for col in ("z", "y")]
        parts.append((n, g["sub"].iloc[0], e, res[0], res[1], g["scorer1"].to_numpy()))
    sx = np.std(np.concatenate([p[3] for p in parts]))
    sy = np.std(np.concatenate([p[4] for p in parts]))
    out = []
    for n, s, e, x, y, sc in parts:
        x, y = x / sx, y / sy
        ok = (e[1:] == e[:-1] + 1)
        out.append(pd.DataFrame({"night": n, "sub": s, "x0": x[:-1][ok], "x1": x[1:][ok], "y0": y[:-1][ok], "y1": y[1:][ok], "s0": sc[:-1][ok], "s1": sc[1:][ok]}))
    D_ = pd.concat(out, ignore_index=True)
    D_["L"] = D_.x0 * D_.y1 - D_.y0 * D_.x1
    D_["dx"] = D_.x1 - D_.x0
    D_["dy"] = D_.y1 - D_.y0
    return D_


rng6 = np.random.default_rng(6061)
for dname, df in [("EESM19", F19), ("EESM23", F23)]:
    D_ = steps6(df)
    for k in (1, 3, 4, 5, 2):
        dk = D_[D_.s0 == k]
        print("%s %-5s steps %6d | mean L %+.4f | mean dx %+.4f | mean dy %+.4f" % (dname, NAMES6[k], len(dk), dk.L.mean(), dk.dx.mean(), dk.dy.mean()))
    D_["xbin"] = pd.qcut(D_.x0, 3, labels=["low x", "mid x", "high x"])
    for b, dk in D_.groupby("xbin", observed=True):
        print("%s %-6s mean dx %+.4f | share dx>0 %.3f" % (dname, b, dk.dx.mean(), (dk.dx > 0).mean()))
    print("%s skewness dx %+.3f | skewness x %+.3f" % (dname, pd.Series(D_.dx).skew(), pd.Series(D_.x0).skew()))
    ps = [signflip(h4_features(df[df["sub"] != s], deg=3), rng6, n_perm=5000)[0] for s in sorted(df["sub"].unique())]
    print("%s leave-one-subject-out p: min %.4f, max %.4f" % (dname, min(ps), max(ps)))
    sizes = df.groupby("night").size().sort_values()
    F_ = h4_features(df[df["night"] != sizes.index[0]], deg=3)
    p, S = signflip(F_, rng6)
    h4_line("%s without shortest night %s" % (dname, sizes.index[0]), F_, p, S)


section("PHASE 5  |  R7: sensitivity of the no-valley results")
FLAG_NIGHTS = ["sub-005_ses-003", "sub-016_ses-003", "sub-011_ses-002"]
for dname, axn, d in SETS:
    Vl = np.array([statA(d[d["sub"] != s])[0] for s in sorted(d["sub"].unique())])
    print("%s %-5s baseline %.3f | without flagged nights %.3f | leave-one-subject-out max %.3f (runs > 0: %d of %d)" % (dname, axn, statA(d)[0], statA(d[~d["night"].isin(FLAG_NIGHTS)])[0], Vl.max(), int((Vl > 0).sum()), len(Vl)))
THIN = 1
rng7 = np.random.default_rng(7070)
for dname, axn, d in SETS:
    D_, bs, lo, hi, h3, h2 = run_statC(d, "scorer1", 150, rng7)
    print("THIN=1 %s %-5s | boundaries %s | depths %s | H3 %s | H2 %s" % (dname, axn, " ".join("%+.2f" % b for b in bs), " ".join("%.3f" % x for x in D_), h3, h2))
THIN = 5


section("PHASE 6  |  R8: stage geometry on candidate axes; R8b plane selection; R8c loadings")


def axes_for(dname):
    if dname == "EESM19":
        ax = {"slope": A2["z"].to_numpy()}
        for k in range(4):
            ax["PC%d" % (k + 1)] = aligned(Xall @ V_[:, k], sub19, win19)
        return ax, A2["scorer1"].to_numpy(), sub19, win19.copy()
    sub23 = E["sub"].to_numpy()
    ones23 = np.ones(len(E), bool)
    ax = {"slope": E["z"].to_numpy()}
    for k in range(4):
        ax["PC%d" % (k + 1)] = aligned(Z23f @ V_[:, k], sub23, ones23)
    return ax, E["scorer1"].to_numpy(), sub23, ones23


ax8, sc8, s8, m8 = axes_for("EESM19")
for nm, vv in ax8.items():
    nr = m8 & np.isin(sc8, [3, 4, 5])
    o_ = np.median(vv[m8 & (sc8 == 4)])
    u_ = np.std(vv[nr] - o_)
    print("%-5s | %s" % (nm, " | ".join("%s %+.2f [%.2f]" % (NAMES[k], (np.median(vv[m8 & (sc8 == k)]) - o_) / u_, mad(vv[m8 & (sc8 == k)]) / u_) for k in (1, 3, 4, 5, 2))))
PAIRS8 = [(1, 3), (1, 4), (3, 2), (4, 2), (5, 2), (1, 2)]
LAB8 = ["W/N1", "W/N2", "N1/REM", "N2/REM", "N3/REM", "W/REM"]
for nm, vv in ax8.items():
    row = []
    for a, b in PAIRS8:
        s = m8 & np.isin(sc8, [a, b])
        r = roc_auc_score(sc8[s] == a, vv[s])
        row.append("%.3f" % max(r, 1 - r))
    print("one-axis AUC %-5s | %s" % (nm, " ".join(row)))
table = {}
for dname in ("EESM19", "EESM23"):
    ax, sc9, sub9, m9 = axes_for(dname)
    rows = []
    for n1, n2 in itertools.combinations(ax.keys(), 2):
        aucs = []
        for a, b in PAIRS8:
            s = m9 & np.isin(sc9, [a, b])
            yy = (sc9[s] == a).astype(int)
            Xp = np.c_[ax[n1][s], ax[n2][s]]
            sc_ = np.zeros(len(yy))
            for tr, te in GroupKFold(5).split(Xp, yy, sub9[s]):
                sc_[te] = LDA().fit(Xp[tr], yy[tr]).decision_function(Xp[te])
            aucs.append(roc_auc_score(yy, sc_))
        rows.append((n1 + "+" + n2, *aucs, float(np.mean(aucs))))
    table[dname] = pd.DataFrame(rows, columns=["plane"] + LAB8 + ["mean"]).sort_values("mean", ascending=False).reset_index(drop=True)
    print(dname)
    print(table[dname].round(3).to_string())
best = table["EESM19"].loc[0, "plane"]
rank23 = int(table["EESM23"].index[table["EESM23"]["plane"] == best][0]) + 1
print("plane rule: best in EESM19 = %s, rank in EESM23 = %d, confirmed %s" % (best, rank23, rank23 <= 3))
print(pd.DataFrame(V_[:, :4], columns=["PC1", "PC2", "PC3", "PC4"], index=["clr_delta", "clr_theta", "clr_alpha", "clr_sigma", "clr_beta", "log_SEF90", "slope", "entropy"]).round(3).to_string())


section("PHASE 6  |  R9: identifiability in the (slope, PC4) plane")
PLANE = ("slope", "PC4")


def class_params(dname):
    ax, sc9, sub9, m9 = axes_for(dname)
    X = np.c_[ax[PLANE[0]], ax[PLANE[1]]]
    return {k: (X[m9 & (sc9 == k)].mean(0), np.cov(X[m9 & (sc9 == k)].T), int((m9 & (sc9 == k)).sum())) for k in NAMES}


def mix_pdf(P, ks, pts):
    w = np.array([P[k][2] for k in ks], float)
    w /= w.sum()
    return sum(wi * MVN(P[k][0], P[k][1]).pdf(pts) for wi, k in zip(w, ks))


def grid_modes(P, ks, n=220):
    mu = np.array([P[k][0] for k in ks])
    sd = np.sqrt(np.array([np.diag(P[k][1]) for k in ks]))
    lo, hi = (mu - 3.5 * sd).min(0), (mu + 3.5 * sd).max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    Zg = mix_pdf(P, ks, np.c_[gx.ravel(), gy.ravel()]).reshape(n, n)
    return int(((Zg == maximum_filter(Zg, size=7)) & (Zg > 0.01 * Zg.max())).sum())


def valley_along(P, ks, a, b, n=600):
    t = np.linspace(-0.25, 1.25, n)
    pts = P[a][0][None] + t[:, None] * (P[b][0] - P[a][0])[None]
    d = mix_pdf(P, ks, pts)
    pk = [i for i in range(1, n - 1) if d[i] > d[i - 1] and d[i] >= d[i + 1] and d[i] > 0.01 * d.max()]
    if len(pk) < 2:
        return 0.0
    i0, i1 = sorted(sorted(pk, key=lambda i: -d[i])[:2])
    return float(1 - d[i0:i1 + 1].min() / min(d[i0], d[i1]))


SUB9 = [("Wake + N1", (1, 3)), ("Wake + N2", (1, 4)), ("N1 + N2", (3, 4)), ("N2 + N3", (4, 5)), ("N1 + REM", (3, 2)), ("N2 + REM", (4, 2)), ("N3 + REM", (5, 2)), ("all five stages", (1, 3, 4, 5, 2))]
for dname in ("EESM19", "EESM23"):
    P = class_params(dname)
    tot = sum(P[k][2] for k in P)
    for k in (1, 3, 4, 5, 2):
        print("%s %-5s weight %.3f mean (%+.2f, %+.2f) sd (%.2f, %.2f) corr %+.2f" % (dname, NAMES[k], P[k][2] / tot, *P[k][0], *np.sqrt(np.diag(P[k][1])), P[k][1][0, 1] / np.sqrt(P[k][1][0, 0] * P[k][1][1, 1])))
    for nm, ks in SUB9:
        print("%s %-16s modes %d | valley %.3f" % (dname, nm, grid_modes(P, ks), valley_along(P, ks, ks[0], ks[1]) if len(ks) == 2 else float("nan")))


section("PHASE 7  |  S16 and S16b: increment-shape surrogate test (calibration fails; not run on real data)")
real_marg = F19["z"].to_numpy()
N_NIGHT16, LEN16, S_SUR, R_REP, ITER, NOISE16 = 20, 700, 49, 30, 30, 0.3


def iaaft(x, rng, n_iter=ITER):
    xs = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(n_iter):
        y = np.fft.irfft(amp * np.exp(1j * np.angle(np.fft.rfft(y))), n=len(x))
        y = xs[np.argsort(np.argsort(y))]
    return y


def night_smooth(rng, tau=45.0):
    phi = np.exp(-1 / tau)
    e = rng.normal(size=LEN16)
    gg = np.empty(LEN16)
    gg[0] = rng.normal()
    for t in range(1, LEN16):
        gg[t] = phi * gg[t - 1] + np.sqrt(1 - phi ** 2) * e[t]
    return np.quantile(real_marg, norm.cdf(gg)) + NOISE16 * rng.normal(size=LEN16)


def night_switch(rng, dwell, levels=(-1.15, 0.0, 1.29), sd=0.6, phi=0.6):
    K = len(levels)
    s = rng.integers(K)
    x = np.empty(LEN16)
    zz = 0.0
    for t in range(LEN16):
        if rng.random() < 1 / dwell:
            s = rng.choice([k for k in range(K) if k != s])
        zz = phi * zz + np.sqrt(1 - phi ** 2) * sd * rng.normal()
        x[t] = levels[s] + zz
    return x + NOISE16 * rng.normal(size=LEN16)


def T_night(x):
    return kurtosis(np.diff(x))


def one_test(make, rng):
    X = [make(rng) for _ in range(N_NIGHT16)]
    Tr = np.mean([T_night(x) for x in X])
    Ts = np.array([np.mean([T_night(iaaft(x, rng)) for x in X]) for _ in range(S_SUR)])
    return (1 + (Ts >= Tr).sum()) / (1 + S_SUR), Tr, Ts.mean()


rng = np.random.default_rng(1616)
for nm, make in [("smooth (real marginal, tau 45)", lambda r: night_smooth(r)), ("switching, dwell 40", lambda r: night_switch(r, 40)), ("switching, dwell 15", lambda r: night_switch(r, 15))]:
    out = np.array([one_test(make, rng) for _ in range(R_REP)])
    print("%-32s | rejection %.2f | mean T %+.3f | mean surrogate T %+.3f" % (nm, np.mean(out[:, 0] < 0.05), out[:, 1].mean(), out[:, 2].mean()))


def gc_null(x, rng):
    n = len(x)
    gg = norm.ppf((np.argsort(np.argsort(x)) + 0.5) / n)
    phi = np.corrcoef(gg[:-1], gg[1:])[0, 1]
    e = rng.normal(size=n)
    s = np.empty(n)
    s[0] = rng.normal()
    for t in range(1, n):
        s[t] = phi * s[t - 1] + np.sqrt(max(1 - phi ** 2, 1e-6)) * e[t]
    return np.sort(x)[np.argsort(np.argsort(s))]


def lag_acf(x, k):
    x = x - x.mean()
    return float((x[:-k] * x[k:]).sum() / (x * x).sum())


rng = np.random.default_rng(1617)
for wname, make in [("smooth", lambda r: night_smooth(r)), ("switching, dwell 40", lambda r: night_switch(r, 40))]:
    X = [make(rng) for _ in range(10)]
    print("%s data: T %+.3f, acf lag1 %.3f lag10 %.3f" % (wname, np.mean([T_night(x) for x in X]), np.mean([lag_acf(x, 1) for x in X]), np.mean([lag_acf(x, 10) for x in X])))
    for nm, f in [("IAAFT 30", lambda x: iaaft(x, rng, 30)), ("IAAFT 100", lambda x: iaaft(x, rng, 100)), ("IAAFT 300", lambda x: iaaft(x, rng, 300)), ("Gaussian-copula AR(1)", lambda x: gc_null(x, rng))]:
        S_ = [[f(x) for x in X] for _ in range(20)]
        print("  %-22s T %+.3f, acf lag1 %.3f lag10 %.3f" % (nm, np.mean([np.mean([T_night(y) for y in s]) for s in S_]), np.mean([np.mean([lag_acf(y, 1) for y in s]) for s in S_]), np.mean([np.mean([lag_acf(y, 10) for y in s]) for s in S_])))


section("ROBUSTNESS  |  R10 and R10b: H4 by variable pair, per feature")


def make_F(dname, xn, yn):
    ax, sc9, sub9, m9 = axes_for(dname)
    if dname == "EESM19":
        df = A2.loc[win19, ["night", "epoch", "sub", "scorer1"]].copy()
        df["z"] = ax[xn][win19]
        df["y"] = ax[yn][win19]
    else:
        df = E[["night", "epoch", "sub", "scorer1"]].copy()
        df["z"] = ax[xn]
        df["y"] = ax[yn]
    return df


def holm(p):
    p = np.asarray(p, float)
    o_ = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    adj[o_] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[o_]))
    return adj


PAIRS10 = [("slope", "PC2"), ("slope", "PC3"), ("slope", "PC4"), ("PC1", "PC2"), ("PC1", "PC4")]
rng10 = np.random.default_rng(6070)
for dname in ("EESM19", "EESM23"):
    for xn, yn in PAIRS10:
        F_ = h4_features(make_F(dname, xn, yn), deg=3)
        p, S = signflip(F_, rng10)
        h4_line("R10 %s %s+%s" % (dname, xn, yn), F_, p, S)
rng10b = np.random.default_rng(6071)
rows10b = []
for dname in ("EESM19", "EESM23"):
    for xn, yn in PAIRS10:
        F_ = h4_features(make_F(dname, xn, yn), deg=3)
        p, S = signflip_each(F_, rng10b)
        rows10b.append([dname, xn, yn, S[:, 0].sum(), np.mean(S[:, 0] > 0), p[0], S[:, 1].sum(), np.mean(S[:, 1] > 0), p[1]])
T10b = pd.DataFrame(rows10b, columns=["dataset", "x", "y", "sum_f1", "share_f1_pos", "p_f1", "sum_f2", "share_f2_pos", "p_f2"])
T10b["p_f1_holm"] = T10b.groupby("dataset")["p_f1"].transform(lambda s: holm(s.to_numpy()))
w19_ = T10b[T10b.dataset == "EESM19"].set_index(["x", "y"])
w23_ = T10b[T10b.dataset == "EESM23"].set_index(["x", "y"])
T10b["rotation replicated"] = [bool(w19_.loc[(r.x, r.y), "p_f1_holm"] < 0.05 and w23_.loc[(r.x, r.y), "p_f1"] < 0.05 and np.sign(w19_.loc[(r.x, r.y), "sum_f1"]) == np.sign(w23_.loc[(r.x, r.y), "sum_f1"])) for r in T10b.itertuples()]
print(T10b.round(4).to_string(index=False))


section("ROBUSTNESS  |  R11, R11b, R11c, R11d: trimmed-anchor alignment and the deep-side PC1 dips")


def trim_align_real(d, q=0.10):
    out = pd.Series(np.nan, index=d.index)
    ratio = []
    for s, g in d.groupby("sub"):
        vv = g["x"].to_numpy()
        a = vv[vv >= np.quantile(vv, q)]
        out.loc[g.index] = (vv - np.median(a)) / mad(a)
        ratio.append(mad(a) / mad(vv))
    return out.to_numpy(), np.array(ratio)


rng11 = np.random.default_rng(1112)
for dname, axn, d in SETS:
    dT = d.copy()
    dT["xa"], ratio = trim_align_real(d)
    V, pos, h, tau, x = statA(dT)
    subs = dT["sub"].unique()
    Vb = np.array([statA(dT, rng11.choice(subs, len(subs)))[0] for _ in range(200)])
    print("R11 A %s %-5s | scale ratio %.2f-%.2f | V_hat %.3f | P(V>0) %.2f | 95th %.3f" % (dname, axn, ratio.min(), ratio.max(), V, np.mean(Vb > 0), np.percentile(Vb, 95)))
for dname, axn, d in SETS:
    dT = d.copy()
    dT["xa"], _ = trim_align_real(d)
    for sc_ in (("scorer1", "scorer2") if dname == "EESM19" else ("scorer1",)):
        D_, bs, lo, hi, h3, h2 = run_statC(dT, sc_, 150, rng11)
        print("R11 C %s %-5s %s | boundaries %s | depths %s | H3 %s | H2 %s" % (dname, axn, sc_, " ".join("%+.2f" % b for b in bs), " ".join("%.3f" % x for x in D_), h3, h2))

rng11b = np.random.default_rng(1113)
for dname, d in [("EESM19", D19), ("EESM23", D23)]:
    for al in ("primary", "trimmed"):
        dd = d.copy()
        if al == "trimmed":
            dd["xa"], _ = trim_align_real(d)
        o_ = dd.loc[dd["scorer1"] == 4, "xa"].median()
        u_ = nrem_unit(dd, "xa")
        subs = dd["sub"].unique()
        Vv, Pp = [], []
        for _ in range(200):
            v_, pos_, *_ = statA(dd, rng11b.choice(subs, len(subs)))
            Vv.append(v_)
            Pp.append((pos_ - o_) / u_ if v_ > 0 else np.nan)
        Vv, Pp = np.array(Vv), np.array(Pp)
        pp = Pp[np.isfinite(Pp)]
        print("R11b %s PC1 %-8s | P(V>0) %.2f | positions %s" % (dname, al, np.mean(Vv > 0), np.round(np.percentile(pp, [5, 50, 95]), 2) if len(pp) else "none"))


def pc1_frame(Dd, med, sc_, rng=None, align="primary"):
    X = feature_matrix(Dd)
    if rng is not None:
        X[:, 5] = np.log(Dd["sef90"].to_numpy() + rng.uniform(-0.125, 0.125, len(Dd)))
    dd = Dd[["night", "sub", "epoch", "scorer1"]].copy()
    dd["x"] = ((X - med) / sc_) @ V_[:, 0]
    if align == "trimmed":
        dd["xa"], _ = trim_align_real(dd)
    else:
        g = dd.groupby("sub")["x"]
        dd["xa"] = (dd["x"] - g.transform("median")) / g.transform(lambda s: mad(s.to_numpy()))
    return dd


rng11c = np.random.default_rng(1114)
for dname, Dd in [("EESM19", D19), ("EESM23", D23)]:
    Xw_ = feature_matrix(Dd)
    med_, sc__ = np.median(Xw_, 0), np.array([mad(Xw_[:, j]) for j in range(Xw_.shape[1])])
    for al in (["primary"] if dname == "EESM19" else ["primary", "trimmed"]):
        Pv = []
        for s in range(5):
            dd = pc1_frame(Dd, med_, sc__, np.random.default_rng(500 + s), al)
            subs = dd["sub"].unique()
            Pv.append(np.mean([statA(dd, rng11c.choice(subs, len(subs)))[0] > 0 for _ in range(100)]))
        print("R11c %s %-8s jittered SEF90: P(V>0) per seed %s, mean %.2f" % (dname, al, np.round(Pv, 2), np.mean(Pv)))

rng11d = np.random.default_rng(1115)
for dname, Dd, aligns in [("EESM19", D19, ["primary"]), ("EESM23", D23, ["primary", "trimmed"])]:
    for al in aligns:
        dd = Dd.copy()
        if al == "trimmed":
            dd["xa"], _ = trim_align_real(Dd)
        o_ = dd.loc[dd["scorer1"] == 4, "xa"].median()
        u_ = nrem_unit(dd, "xa")
        dd["zz"] = (dd["xa"] - o_) / u_
        subs_ = np.sort(dd["sub"].unique())
        C = np.zeros((300, len(subs_)))
        hit = np.zeros(300, bool)
        for b in range(300):
            draw = rng11d.choice(subs_, len(subs_))
            C[b] = [(draw == s).sum() for s in subs_]
            hit[b] = statA(dd, draw)[0] > 0
        rows = []
        for j, s in enumerate(subs_):
            gs = dd[dd["sub"] == s]
            rows.append([s, gs["night"].nunique(), 100 * (gs["scorer1"] == 5).mean(), gs.loc[gs["scorer1"] == 5, "zz"].median(), 100 * (gs["zz"] > 2.0).mean(),
                         C[hit, j].mean() - C[~hit, j].mean() if hit.any() and (~hit).any() else np.nan])
        Td = pd.DataFrame(rows, columns=["sub", "nights", "N3 %", "N3 median", "% > +2.0", "extra copies in dip draws"])
        print("R11d %s PC1 %s: P(V>0) %.2f" % (dname, al, hit.mean()))
        print(Td.sort_values("extra copies in dip draws", ascending=False).round(2).to_string(index=False))


section("ROBUSTNESS  |  R12: QC amplitude threshold 100 / 150 / 200 microvolt")


def as_bool(s):
    return s.map({True: True, False: False, "True": True, "False": False, 1: True, 0: False}).fillna(False).astype(bool)


def c3_features(folder, night, onsets):
    z = np.load(f"{PROC}/{folder}/{night}.npz", allow_pickle=True)
    return night_channel_features(z["x100"][0].astype(np.float64), z["bad1s"][list(z["ch"]).index("C3-M2")], onsets)[0]


RAW_Q = {"EESM19": ("v2", pd.read_csv(f"{PROC}/features_eesm19_v1.csv.gz")), "EESM23": ("v23", pd.read_csv(f"{PROC}/features_eesm23_v1.csv.gz"))}


def variant_q(dname, sd_max):
    folder, F0 = RAW_Q[dname]
    Fq = F0[F0["ch"] == "C3-M2"].copy()
    keep = (Fq["bad_frac"] <= MASK_MAX) & (Fq["med_sd_uV"] <= sd_max) & ~as_bool(Fq["outside"])
    Fq["include"] = keep & (keep.groupby(Fq["night"]).transform("mean") >= MIN_KEEP)
    need = Fq["include"] & Fq[BAND[0]].isna()
    for night, gq in Fq[need].groupby("night"):
        Fq.loc[gq.index, FEATS] = c3_features(folder, night, gq["onset"].to_numpy(float))
    return Fq, int(need.sum())


def axes_q(d, med, sc_):
    Z = (feature_matrix(d) - med) / sc_
    base = d[["night", "sub", "epoch", "scorer1"]].copy()
    out = {}
    for nm, raw in [("PC1", Z @ V_[:, 0]), ("slope", -Z[:, 6]), ("PC2", Z @ V_[:, 1])]:
        gq = pd.Series(raw, index=d.index).groupby(d["sub"])
        out[nm] = base.assign(x=raw, xa=(raw - gq.transform("median").to_numpy()) / gq.transform(lambda s: mad(s.to_numpy())).to_numpy())
    return out


PRIM_Q = {dn: (np.median(feature_matrix(Dd), 0), np.array([mad(feature_matrix(Dd)[:, j]) for j in range(8)])) for dn, Dd in (("EESM19", D19), ("EESM23", D23))}
rng12 = np.random.default_rng(1212)
rows12 = []
for sd_max in (150.0, 100.0, 200.0):
    for dname in ("EESM19", "EESM23"):
        Fq, n_new = variant_q(dname, sd_max)
        d = prepare(Fq, "%s @ %.0f uV" % (dname, sd_max))
        AX = axes_q(d, *PRIM_Q[dname])
        res = {"uV": int(sd_max), "dataset": dname, "nights": d["night"].nunique(), "epochs": len(d), "wake %": 100 * (d["scorer1"] == 1).mean()}
        for axn in ("slope", "PC1"):
            dd = AX[axn]
            res["V " + axn] = statA(dd)[0]
            subs = dd["sub"].unique()
            res["P(V>0) " + axn] = np.mean([statA(dd, rng12.choice(subs, len(subs)))[0] > 0 for _ in range(200)])
            D_, bs, lo, hi, h3, h2 = run_statC(dd, "scorer1", 150, rng12)
            res["bounds " + axn] = " ".join("%+.2f" % b for b in bs)
            res["H3 " + axn], res["H2 " + axn] = h3, h2
        FH = AX["slope"][["night", "epoch", "sub", "scorer1"]].assign(z=AX["slope"]["xa"].to_numpy(), y=AX["PC2"]["xa"].to_numpy())
        p, S = signflip(h4_features(FH, deg=3), rng12)
        res.update({"H4 p": p, "H4 sum f1": S[:, 0].sum(), "H4 sum f2": S[:, 1].sum()})
        rows12.append(res)
print(pd.DataFrame(rows12).round(3).to_string(index=False))


section("ROBUSTNESS  |  S17: validation of the cycle-structure control (zero-phase high-pass)")
N_SUB17, NPS17, T17, NOISE17, TAU17, A_C17 = 20, 4, 860, 0.4, 37.0, 1.0
PHI17 = (TAU17 - 1) / (TAU17 + 1)


def ar1_17(T_, n, rng):
    x = np.empty((T_, n))
    x[0] = rng.normal(0, 1, n)
    s = np.sqrt(1 - PHI17 ** 2)
    for t in range(1, T_):
        x[t] = PHI17 * x[t - 1] + s * rng.normal(0, 1, n)
    return x


def wave_17(T_, ph, period=180, rise=0.7, smooth=5):
    t = (np.arange(T_)[:, None] / period + ph[None, :]) % 1.0
    w = np.where(t < rise, t / rise, (1 - t) / (1 - rise))
    w = gaussian_filter1d(w, smooth, axis=0, mode="nearest")
    return (w - w.mean()) / w.std()


def events_17(T_, n, rng, rate=1 / 60, J=2.0, relax=10.0):
    a = np.zeros((T_, n))
    dk = np.exp(-1 / relax)
    for t in range(1, T_):
        a[t] = dk * a[t - 1] - J * (rng.random(n) < rate)
    return a


def study_17(world, rng):
    n = N_SUB17 * NPS17
    x, y = ar1_17(T17, n, rng), ar1_17(T17, n, rng)
    if world == "smooth cycle":
        ph = rng.uniform(0, 1, n)
        x = x + A_C17 * wave_17(T17, ph)
        y = y + A_C17 * wave_17(T17, ph + 0.25)
    elif world == "fast events":
        x = x + events_17(T17, n, rng)
    x = x + rng.normal(0, NOISE17, x.shape)
    y = y + rng.normal(0, NOISE17, y.shape)
    night = np.repeat(np.arange(n), T17)
    return pd.DataFrame({"night": night, "sub": night // NPS17, "epoch": np.tile(np.arange(T17), n), "scorer1": 4, "z": x.T.ravel(), "y": y.T.ravel()})


WS_17 = [None, 15, 31, 61]
rng_17 = np.random.default_rng(1717)
rej_17 = {}
for world in ("reversible", "smooth cycle", "fast events"):
    hits = {Wn: [] for Wn in WS_17}
    for r in range(150):
        df = study_17(world, rng_17)
        for Wn in WS_17:
            hits[Wn].append(signflip(h4_features(df, W=Wn, deg=3), rng_17, n_perm=2000)[0] < 0.05)
    rej_17[world] = {Wn: float(np.mean(vv)) for Wn, vv in hits.items()}
    print("%-13s | %s" % (world, " | ".join("%s %.3f" % ("no HP" if Wn is None else "W=%d" % Wn, rej_17[world][Wn]) for Wn in WS_17)))
S17_PASS = all(vv <= 0.07 for vv in rej_17["reversible"].values()) and rej_17["smooth cycle"][31] <= 0.10 and rej_17["fast events"][31] >= 0.80
print("S17 overall:", "PASS" if S17_PASS else "FAIL")


section("ROBUSTNESS  |  R13: H4 with the cycle-structure control")
rng_13 = np.random.default_rng(1313)
R13_ROWS = []
if S17_PASS:
    for dname, df in [("EESM19", F19), ("EESM23", F23)]:
        for lab, kw in [("cubic only", dict(W=None)), ("cubic + HP W=31", dict(W=31)), ("cubic + HP W=15", dict(W=15)), ("cubic + HP W=61", dict(W=61)), ("HP W=31 + NREM-only", dict(W=31, nrem_only=True))]:
            F_ = h4_features(df, deg=3, **kw)
            p, S = signflip(F_, rng_13)
            pe, _ = signflip_each(F_, rng_13)
            R13_ROWS.append(dict(dataset=dname, variant=lab, p=p, p_f1=pe[0], p_f2=pe[1], sum_f1=S[:, 0].sum(), sum_f2=S[:, 1].sum(), pos_f1=np.mean(S[:, 0] > 0), pos_f2=np.mean(S[:, 1] > 0)))
    print(pd.DataFrame(R13_ROWS).round(4).to_string(index=False))


section("FIGURES  |  manuscript and supplementary figures")
import glob, os, subprocess
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter1d as _gf
from sklearn.metrics import roc_auc_score, cohen_kappa_score

for _nm in ["D19", "D19s", "D23", "D23s", "A", "A2", "E", "F19", "F23", "statA", "kde_grid", "nrem_unit", "parts_of", "stats_real",
            "night_arrays", "iat_ragged", "steps6", "axes_for", "PROC", "THIN"]:
    assert _nm in globals(), "missing %s: run the analysis cells first" % _nm
assert THIN == 5, "THIN must be 5"
H4F = globals()["h4_features_hp"] if "h4_features_hp" in globals() else globals()["h4_features"]
SFE = globals()["signflip_each_10b"] if "signflip_each_10b" in globals() else globals()["signflip_each"]


def ensure_font():
    names = {f.name for f in fm.fontManager.ttflist}
    if not ({"Arial", "Helvetica"} & names):
        try:
            subprocess.run("echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections; "
                           "apt-get -qq -y install ttf-mscorefonts-installer", shell=True, timeout=900, capture_output=True)
            for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/*.[tT][tT][fF]"):
                fm.fontManager.addfont(f)
        except Exception as e:
            print("Arial installation skipped:", repr(e)[:80])
        names = {f.name for f in fm.fontManager.ttflist}
    for n in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"):
        if n in names:
            return n


FONT = ensure_font()
print("font in use:", FONT)
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 6.5, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2.5,
                     "axes.titlepad": 4, "lines.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.handlelength": 1.4, "legend.borderaxespad": 0.2, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "savefig.dpi": 600, "figure.dpi": 150, "mathtext.default": "regular", "axes.unicode_minus": True})

MM = 1 / 25.4
FIGDIR = os.path.join(str(PROC), "figures", "final")
os.makedirs(FIGDIR, exist_ok=True)
SC = {1: "#D55E00", 3: "#E69F00", 4: "#009E73", 5: "#0072B2", 2: "#CC79A7"}
SN = {1: "Wake", 3: "N1", 4: "N2", 5: "N3", 2: "REM"}
ORDER = (1, 3, 4, 5, 2)
DSC = {"EESM19": "#1f1f1f", "EESM23": "#8a8a8a"}
GREY = "#9a9a9a"


def plab(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, "%s.%s" % (name, ext)), bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print("saved", os.path.join(FIGDIR, name + ".(pdf|png)"))


def to_nrem(d, col="xa"):
    return (d[col].to_numpy() - d.loc[d["scorer1"] == 4, col].median()) / nrem_unit(d, col)


SETS_F = {("EESM19", "slope"): D19s, ("EESM19", "PC1"): D19, ("EESM23", "slope"): D23s, ("EESM23", "PC1"): D23}
POS = {k: (d["scorer1"].to_numpy(), to_nrem(d)) for k, d in SETS_F.items()}
AXNAME = {"slope": "Spectral slope", "PC1": "PC1"}


def boxes(ax, axis):
    for i, k in enumerate(ORDER):
        for j, ds in enumerate(("EESM19", "EESM23")):
            sc_, z = POS[(ds, axis)]
            v = z[sc_ == k]
            q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
            x = i + (-0.19 if j == 0 else 0.19)
            ax.plot([x, x], [q05, q25], color=SC[k], lw=0.6, solid_capstyle="butt")
            ax.plot([x, x], [q75, q95], color=SC[k], lw=0.6, solid_capstyle="butt")
            ax.add_patch(Rectangle((x - 0.15, q25), 0.30, q75 - q25, facecolor=SC[k], alpha=1.0 if j == 0 else 0.35, edgecolor=SC[k], lw=0.6))
            ax.plot([x - 0.15, x + 0.15], [q50, q50], color="white" if j == 0 else SC[k], lw=0.9, solid_capstyle="butt")
    ax.axhline(0, color="0.8", lw=0.4, zorder=0)
    ax.set_xticks(range(5))
    ax.set_xticklabels([SN[k] for k in ORDER])
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylabel("%s position (NREM s.d.)" % AXNAME[axis])
    ax.tick_params(axis="x", length=0)


def figure1():
    d = D19s.assign(z=POS[("EESM19", "slope")][1])
    stats_n = d.groupby("night").agg(n=("epoch", "size"), wake=("scorer1", lambda s: (s == 1).mean()), n3=("scorer1", lambda s: (s == 5).mean()),
                                    rem=("scorer1", lambda s: (s == 2).mean()))
    cand = stats_n[(stats_n.wake.between(0.02, 0.08)) & (stats_n.n3 >= 0.12) & (stats_n.rem >= 0.12)]
    cand = cand if len(cand) else stats_n
    night = (cand["n"] - cand["n"].median()).abs().idxmin()
    g = d[(d["night"] == night) & d["scorer1"].isin([1, 2, 3, 4, 5])].sort_values("epoch")
    t = (g["epoch"].to_numpy() - g["epoch"].min()) * 30 / 3600
    st = g["scorer1"].to_numpy()
    lev = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
    lv = np.array([lev[s] for s in st])
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 1.0, 1.45], hspace=0.62, wspace=0.42, width_ratios=[1, 1, 1.15])
    ax = fig.add_subplot(gs[0, :])
    ax.step(t, lv, where="post", color="0.75", lw=0.4)
    segs = [[(t[i], lv[i]), (t[i] + 30 / 3600, lv[i])] for i in range(len(t))]
    ax.add_collection(LineCollection(segs, colors=[SC[s] for s in st], linewidths=2.4, capstyle="butt"))
    ax.set_yticks(range(5))
    ax.set_yticklabels(["N3", "N2", "N1", "REM", "Wake"])
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlim(0, t.max() + 30 / 3600)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_title("Example night (%s), scorer 1" % night.replace("_", ", "), loc="left", fontsize=6.5, color="0.3")
    plab(ax, "a", x=-0.085)
    ax = fig.add_subplot(gs[1, :])
    ax.scatter(t, g["z"], s=1.4, c=[SC[s] for s in st], linewidths=0, rasterized=True)
    ax.plot(t, pd.Series(g["z"].to_numpy()).rolling(9, center=True, min_periods=3).median(), color="black", lw=0.7)
    ax.set_xlim(0, t.max() + 30 / 3600)
    ax.invert_yaxis()
    ax.set_xlabel("Time from sleep-window start (h)")
    ax.set_ylabel("Slope depth (NREM s.d.)")
    ax.text(0.995, 0.04, "deeper ↓", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="0.4")
    plab(ax, "b", x=-0.085, y=1.1)
    for col, axis, lab in [(0, "slope", "c"), (1, "PC1", "d")]:
        ax = fig.add_subplot(gs[2, col])
        boxes(ax, axis)
        plab(ax, lab)
        if col == 0:
            ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor="0.3", edgecolor="0.3", lw=0.6), Rectangle((0, 0), 1, 1, facecolor="0.3", alpha=0.35, edgecolor="0.3", lw=0.6)],
                      labels=["EESM19", "EESM23"], loc="lower left", ncol=2, handlelength=1.0, columnspacing=0.8)
    ax = fig.add_subplot(gs[2, 2])
    axd, sc8, s8, m8 = axes_for("EESM19")
    pairs = [(1, 3), (1, 4), (3, 2), (4, 2), (5, 2), (1, 2)]
    rows = list(axd.keys())
    Mx = np.zeros((len(rows), len(pairs)))
    for i, nm in enumerate(rows):
        for j, (a, b) in enumerate(pairs):
            s = m8 & np.isin(sc8, [a, b])
            r = roc_auc_score(sc8[s] == a, axd[nm][s])
            Mx[i, j] = max(r, 1 - r)
    im = ax.imshow(Mx, cmap="Greys", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            ax.text(j, i, "%.2f" % Mx[i, j], ha="center", va="center", fontsize=5.2, color="white" if Mx[i, j] > 0.78 else "black")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(["%s/\n%s" % (SN[a], SN[b]) for a, b in pairs], fontsize=5.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["Slope" if r == "slope" else r for r in rows])
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[0.5, 0.75, 1.0])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, width=0.4)
    cb.set_label("AUC (one axis, EESM19)", fontsize=6)
    plab(ax, "e", x=-0.22)
    fig.align_ylabels()
    save(fig, "Figure1_data_and_depth_axes")


def figure2():
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 112 * MM), sharex=True)
    fig.subplots_adjust(hspace=0.42, wspace=0.18, top=0.86)
    for r, ds in enumerate(("EESM19", "EESM23")):
        for c, axis in enumerate(("slope", "PC1")):
            d = SETS_F[(ds, axis)].copy()
            o_ = d.loc[d["scorer1"] == 4, "xa"].median()
            u_ = nrem_unit(d, "xa")
            V, pos, h, tau, x = statA(d)
            cg, pg, _ = kde_grid(x, h)
            cz, pz = (cg - o_) / u_, pg * u_
            zs = (d["xa"].to_numpy() - o_) / u_
            sc_ = d["scorer1"].to_numpy()
            ntot = np.isin(sc_, list(SN)).sum()
            hz = h / u_
            ax = axes[r, c]
            for k in ORDER:
                zk = zs[sc_ == k]
                cnt, _ = np.histogram(zk, bins=np.r_[cz - (cz[1] - cz[0]) / 2, cz[-1] + (cz[1] - cz[0]) / 2])

                yk = _gf(cnt.astype(float), hz / (cz[1] - cz[0]), mode="constant") / (ntot * (cz[1] - cz[0]))
                ax.fill_between(cz, yk, color=SC[k], alpha=0.30, lw=0)
                ax.plot(cz, yk, color=SC[k], lw=0.6)
            ax.plot(cz, pz, color="black", lw=1.1)
            na = night_arrays(d, "xa")
            ne = sum(np.isfinite(a).sum() for a in na) / iat_ragged(na)
            _, bs = stats_real(parts_of(d, "scorer1"), ne)
            ymax = pz.max() * 1.10
            ax.set_ylim(0, ymax)
            ax.set_xlim(-6.5, 5.0)
            for b, lab in zip(bs, ["W|N1", "N1|N2", "N2|N3"]):
                ax.axvline(b, color="0.35", ls=(0, (2.5, 1.5)), lw=0.5)
                ax.text(b - 0.07, ymax * 0.985, lab, ha="right", va="top", rotation=90, fontsize=5.3, color="0.35")
            ax.text(0.985, 0.94, "V = %.3f" % V, transform=ax.transAxes, ha="right", va="top", fontsize=6)
            ax.set_title("%s  ·  %s" % (ds, AXNAME[axis]), loc="left")
            if c == 0:
                ax.set_ylabel("Density (per NREM s.d.)")
            if r == 1:
                ax.set_xlabel("Position (NREM s.d. from N2 median; deeper to the right)")
            plab(ax, "abcd"[2 * r + c], x=-0.11, y=1.04)
    handles = [Line2D([], [], color="black", lw=1.1)] + [Rectangle((0, 0), 1, 1, facecolor=SC[k], alpha=0.45, edgecolor=SC[k], lw=0.6) for k in ORDER] + [Line2D([], [], color="0.35", ls=(0, (2.5, 1.5)), lw=0.5)]
    fig.legend(handles, ["All epochs"] + [SN[k] for k in ORDER] + ["Scorer-1 boundary"], loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.2, columnspacing=1.2)
    save(fig, "Figure2_densities")


def figure3():
    real = {}
    for (ds, axis), (sc_, z) in POS.items():
        real[(ds, axis)] = -np.median(z[sc_ == 1])
    shares = np.array([2.5, 6.0, 10.0])
    spreads = [0.6, 0.83, 1.2]
    if all(n in globals() for n in ("build_world", "observed_world", "valley_info")):
        grid = np.round(np.arange(-0.5, -6.01, -0.1), 2)

        def first(w, s, sig):
            for m in grid:
                Wm = build_world(m_w=float(m), s_w=s, w_w=w)
                Wm = observed_world(Wm, sig) if sig > 0 else Wm
                if valley_info(Wm, float(m), 0.0)[0]:
                    return -m
            return np.nan

        thr = {s: np.array([first(w / 100, s, 0.4) for w in shares]) for s in spreads}
        floor = np.nanmin([first(w / 100, s, 0.0) for s in spreads for w in shares])
        print("Figure 3a thresholds recomputed from the S12 model")
    else:
        thr = {0.6: np.array([3.8, 3.4, 3.2]), 0.83: np.array([4.1, 3.7, 3.5]), 1.2: np.array([4.4, 4.1, 3.9])}
        floor = 2.8
        print("Figure 3a thresholds taken from log 4.4 (S12 functions not in session)")
    fig = plt.figure(figsize=(183 * MM, 66 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.48)
    ax = fig.add_subplot(gs[0])
    ax.axhspan(0, floor, color="0.93", lw=0, zorder=0)
    ax.text(1.7, 0.12, "no valley possible below %.1f s.d.\n(any spread or share, no noise)" % floor, fontsize=5.3, color="0.4", va="bottom")
    for (s, y), mk in zip(thr.items(), ("o", "s", "^")):
        ax.plot(shares, y, ls="none", marker=mk, ms=3.4, mfc="white", mec="black", mew=0.6, label="%.2f" % s)
    cols = {("EESM19", "slope"): "#0072B2", ("EESM23", "slope"): "#56B4E9", ("EESM19", "PC1"): "#D55E00", ("EESM23", "PC1"): "#E69F00"}
    lab_y = {}
    for k in sorted(real, key=lambda k: -real[k]):
        y = real[k]
        ax.axhline(y, color=cols[k], lw=0.9)
        yy = y + 0.13 if not lab_y else min(y + 0.13, min(lab_y.values()) - 0.2)
        lab_y[k] = yy
        ax.text(10.3, yy, "%s, %s (%.2f)" % (k[0], "slope" if k[1] == "slope" else "PC1", y), color=cols[k], fontsize=5.3, ha="right", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85))
    ax.set_xlim(1.5, 10.4)
    ax.set_ylim(0, 4.8)
    ax.set_xticks(shares)
    ax.set_xlabel("Wake share of epochs (%)")
    ax.set_ylabel("Wake distance from NREM (NREM s.d.)")
    ax.legend(title="Wake spread (s.d.)", title_fontsize=5.5, loc="upper right", ncol=3, handletextpad=0.2, columnspacing=0.7, fontsize=5.5)
    plab(ax, "a", x=-0.2)
    P = {}
    axd, sc9, sub9, m9 = axes_for("EESM19")
    X = np.c_[axd["slope"], axd["PC4"]]
    for k in ORDER:
        s = m9 & (sc9 == k)
        P[k] = (X[s].mean(0), np.cov(X[s].T), int(s.sum()))
    w = np.array([P[k][2] for k in ORDER], float)
    w /= w.sum()
    gx, gy = np.meshgrid(np.linspace(-6.5, 4.0, 260), np.linspace(-5.0, 4.0, 240))
    pts = np.c_[gx.ravel(), gy.ravel()]
    Z = sum(wi * multivariate_normal(P[k][0], P[k][1]).pdf(pts) for wi, k in zip(w, ORDER)).reshape(gx.shape)
    ax = fig.add_subplot(gs[1])
    ax.contourf(gx, gy, Z, levels=np.linspace(Z.max() * 0.02, Z.max(), 9), cmap="Greys", alpha=0.75)
    ax.contour(gx, gy, Z, levels=np.linspace(Z.max() * 0.02, Z.max(), 9), colors="0.45", linewidths=0.25)
    LBL = {1: (-1.0, 0.9), 3: (-1.2, -0.9), 4: (-0.3, 1.0), 5: (1.0, 0.5), 2: (0.4, -1.0)}
    for k in ORDER:
        mu, C, _ = P[k]
        vals, vecs = np.linalg.eigh(C)
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(mu, 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]), angle=ang, fill=False, edgecolor=SC[k], lw=0.8))
        ax.plot(*mu, "o", ms=2.8, color=SC[k], mec="white", mew=0.3)
        dvec = np.array(LBL[k], float)
        dvec /= np.linalg.norm(dvec)
        tp = mu + dvec * (np.sqrt(vals[1]) + 0.55)
        ax.annotate(SN[k], xy=mu, xytext=tp, color=SC[k], fontsize=5.5, ha="center", va="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=SC[k], lw=0.4, shrinkA=0, shrinkB=1.5))
    ax.set_xlim(-6.5, 4.0)
    ax.set_ylim(-5.0, 4.0)
    ax.set_xlabel("Spectral slope (s.d.; deeper →)")
    ax.set_ylabel("PC4: sigma vs alpha (s.d.)")
    ax.set_title("Discrete Gaussian stages, EESM19", loc="left", fontsize=6.5, color="0.3")
    plab(ax, "b", x=-0.2)

    def mixp(PP, ks, q):
        ww = np.array([PP[k][2] for k in ks], float)
        ww /= ww.sum()
        return sum(wi * multivariate_normal(PP[k][0], PP[k][1]).pdf(q) for wi, k in zip(ww, ks))

    def vdepth(PP, a, b):
        tt = np.linspace(-0.25, 1.25, 600)
        q = PP[a][0][None] + tt[:, None] * (PP[b][0] - PP[a][0])[None]
        dd = mixp(PP, (a, b), q)
        pk = [i for i in range(1, 599) if dd[i] > dd[i - 1] and dd[i] >= dd[i + 1] and dd[i] > 0.01 * dd.max()]
        if len(pk) < 2:
            return 0.0
        i0, i1 = sorted(sorted(pk, key=lambda i: -dd[i])[:2])
        return float(1 - dd[i0:i1 + 1].min() / min(dd[i0], dd[i1]))

    pairs = [(1, 3), (3, 4), (4, 5), (3, 2), (4, 2), (1, 4), (5, 2)]
    ax = fig.add_subplot(gs[2])
    for j, ds in enumerate(("EESM19", "EESM23")):
        axd, sc9, sub9, m9 = axes_for(ds)
        X = np.c_[axd["slope"], axd["PC4"]]
        PP = {k: (X[m9 & (sc9 == k)].mean(0), np.cov(X[m9 & (sc9 == k)].T), int((m9 & (sc9 == k)).sum())) for k in ORDER}
        vd = [vdepth(PP, a, b) for a, b in pairs]
        ax.plot(np.arange(len(pairs)) + (-0.12 if j == 0 else 0.12), vd, "o", ms=3.2, color=DSC[ds], mfc=DSC[ds] if j == 0 else "white", mew=0.7, label=ds, ls="none")
    ax.axvspan(4.5, 6.5, color="0.94", lw=0, zorder=0)
    ax.text(5.5, 0.97, "non-adjacent\nstages", ha="center", va="top", fontsize=5.3, color="0.4")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(["%s/\n%s" % (SN[a], SN[b]) for a, b in pairs], fontsize=5.5)
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylabel("Valley depth between class means")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", length=0)
    plab(ax, "c", x=-0.2)
    save(fig, "Figure3_resolution_and_identifiability")


def figure4():
    fig = plt.figure(figsize=(183 * MM, 122 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.32)
    steps = {}
    for j, (ds, df) in enumerate((("EESM19", F19), ("EESM23", F23))):
        Dd = steps6(df)
        steps[ds] = Dd
        ax = fig.add_subplot(gs[0, j])
        edges = np.arange(-4.0, 4.01, 0.8)
        gx = np.digitize(Dd.x0, edges) - 1
        gy = np.digitize(Dd.y0, edges) - 1
        ok = (gx >= 0) & (gx < len(edges) - 1) & (gy >= 0) & (gy < len(edges) - 1)
        G = Dd[ok].assign(gx=gx[ok], gy=gy[ok]).groupby(["gx", "gy"]).agg(n=("dx", "size"), dx=("dx", "mean"), dy=("dy", "mean")).reset_index()
        G = G[G.n >= 100]
        cnt = np.full((len(edges) - 1, len(edges) - 1), np.nan)
        cnt[G.gy, G.gx] = np.log10(G.n)
        ax.imshow(cnt, origin="lower", extent=[edges[0], edges[-1], edges[0], edges[-1]], cmap="Greys", vmin=1.5, vmax=4.8, alpha=0.55, aspect="equal", interpolation="nearest")
        xc = (edges[G.gx] + edges[G.gx + 1]) / 2
        yc = (edges[G.gy] + edges[G.gy + 1]) / 2
        q = ax.quiver(xc, yc, G.dx, G.dy, angles="xy", scale_units="xy", scale=1.0, width=0.0045, headwidth=4, headlength=4.5, color="black")
        ax.quiverkey(q, 0.84, 1.03, 0.5, "0.5 s.d.", labelpos="E", fontproperties={"size": 5.5})
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xticks(range(-4, 5, 2))
        ax.set_yticks(range(-4, 5, 2))
        ax.set_xlabel("Spectral slope (s.d.; deeper →)")
        ax.set_ylabel("PC2 (s.d.)")
        ax.set_title(ds, loc="left")
        plab(ax, "ab"[j], x=-0.2)
    sub = gs[1, 0].subgridspec(1, 2, width_ratios=[3.0, 1.0], wspace=0.06)
    axl, axr = fig.add_subplot(sub[0]), fig.add_subplot(sub[1], sharey=None)
    rng_f = np.random.default_rng(4040)
    ypos = {k: i for i, k in enumerate(ORDER[::-1])}
    for j, ds in enumerate(("EESM19", "EESM23")):
        Dd = steps[ds]
        subj = Dd["sub"].to_numpy()
        us = np.unique(subj)
        idx = {s: np.where(subj == s)[0] for s in us}
        for k in ORDER:
            m = (Dd.s0 == k).to_numpy()
            Ls = np.array([Dd.L.to_numpy()[idx[s]][m[idx[s]]].sum() for s in us])
            ns = np.array([m[idx[s]].sum() for s in us])
            est = Ls.sum() / ns.sum()
            bs = []
            for _ in range(2000):
                jj = rng_f.integers(0, len(us), len(us))
                bs.append(Ls[jj].sum() / max(ns[jj].sum(), 1))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            y = ypos[k] + (0.13 if j == 0 else -0.13)
            target = axr if k == 1 else axl
            target.plot([lo, hi], [y, y], color=SC[k], lw=0.8, solid_capstyle="butt")
            target.plot(est, y, "o", ms=3.4, color=SC[k], mfc=SC[k] if j == 0 else "white", mew=0.8)
    axl.axvline(0, color="0.6", lw=0.5)
    axl.set_xlim(-0.065, 0.145)
    axr.set_xlim(0.35, 1.45)
    for a in (axl, axr):
        a.set_ylim(-0.6, 4.6)
    axl.set_yticks([ypos[k] for k in ORDER])
    axl.set_yticklabels([SN[k] for k in ORDER])
    axr.set_yticks([])
    axr.spines["left"].set_visible(False)
    axr.spines["bottom"].set_visible(True)
    axr.tick_params(axis="x", which="both", bottom=True)
    axr.set_xticks([0.5, 1.0])
    axl.set_xticks([-0.05, 0, 0.05, 0.1])
    dkw = dict(transform=axl.transAxes, color="black", clip_on=False, lw=0.5)
    axl.plot([1 - 0.012, 1 + 0.012], [-0.025, 0.025], **dkw)
    axr.plot([-0.04, 0.04], [-0.025, 0.025], transform=axr.transAxes, color="black", clip_on=False, lw=0.5)
    axl.set_xlabel("Mean rotation L (stage at time t)")
    axl.xaxis.set_label_coords(0.67, -0.17)
    axl.legend(handles=[Line2D([], [], marker="o", ls="none", color="0.25", ms=3.4), Line2D([], [], marker="o", ls="none", color="0.25", mfc="white", ms=3.4)],
               labels=["EESM19", "EESM23"], loc="lower right", handletextpad=0.2)
    plab(axl, "c", x=-0.2)
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-6, 6, 81)
    cen = (bins[1:] + bins[:-1]) / 2
    for ds in ("EESM19", "EESM23"):
        dx = steps[ds].dx.to_numpy()
        hp, _ = np.histogram(dx, bins=bins, density=True)
        hm, _ = np.histogram(-dx, bins=bins, density=True)
        ax.plot(cen, np.where(hp > 0, hp, np.nan), color=DSC[ds], lw=0.9, label="%s, Δx" % ds)
        ax.plot(cen, np.where(hm > 0, hm, np.nan), color=DSC[ds], lw=0.7, ls=(0, (2, 1.2)), label="%s, −Δx (mirror)" % ds)
        ax.text(0.98, 0.95 if ds == "EESM19" else 0.86, "skewness %s: %+.2f" % (ds, pd.Series(dx).skew()), transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=DSC[ds])
    ax.set_yscale("log")
    ax.set_xlim(-6, 6)
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("One-epoch change in depth Δx (s.d.)")
    ax.set_ylabel("Probability density")
    ax.legend(loc="lower center", ncol=2, fontsize=5.3, columnspacing=0.8)
    plab(ax, "d", x=-0.2)
    save(fig, "Figure4_time_direction")


def figure5():
    rej = globals().get("rej_17", None)
    if rej is None:
        rej = {"reversible": {None: 0.040, 15: 0.047, 31: 0.047, 61: 0.053}, "smooth cycle": {None: 1.0, 15: 0.080, 31: 0.033, 61: 0.153},
               "fast events": {None: 1.0, 15: 1.0, 31: 1.0, 61: 1.0}}
        print("Figure 5a uses the logged S17 rejection rates (rej_17 not in session)")
    fig = plt.figure(figsize=(183 * MM, 64 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.1], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    ws = [None, 15, 31, 61]
    xs = np.arange(4)
    ax.axhspan(0, 0.07, color="0.93", lw=0, zorder=0)
    ax.axhline(0.05, color="0.5", lw=0.5, ls=(0, (2.5, 1.5)))
    for (wn, lab), col, mk in zip([("reversible", "Reversible (null)"), ("smooth cycle", "Smooth 90-min cycle"), ("fast events", "Fast lightening events")], ["0.55", "#E69F00", "#0072B2"], ["o", "s", "^"]):
        ax.plot(xs, [rej[wn][w] for w in ws], marker=mk, ms=3.4, color=col, lw=0.8, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(["None", "15", "31", "61"])
    ax.set_xlabel("High-pass window (epochs)")
    ax.set_ylabel("Rejection rate (α = 0.05)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center right", fontsize=5.5)
    ax.set_title("Simulated nights (S17)", loc="left", fontsize=6.5, color="0.3")
    plab(ax, "a", x=-0.22)
    ax = fig.add_subplot(gs[1])
    variants = [("Cubic", dict(W=None)), ("HP 31", dict(W=31)), ("HP 15", dict(W=15)), ("HP 61", dict(W=61)), ("HP 31,\nNREM only", dict(W=31, nrem_only=True))]
    rng_5 = np.random.default_rng(1313)
    floor = -np.log10(1 / 20001)
    for j, (ds, df) in enumerate((("EESM19", F19), ("EESM23", F23))):
        for i, (lab, kw) in enumerate(variants):
            Fv = H4F(df, deg=3, **kw)
            pe, _ = SFE(Fv, rng_5, n_perm=20000)
            for f, (mk, dx) in enumerate([("s", -0.22), ("o", -0.07)] if j == 0 else [("s", 0.07), ("o", 0.22)]):
                y = -np.log10(pe[f])
                ax.plot(i + dx, y, marker="^" if y >= floor - 1e-9 else mk, ms=3.4, ls="none", color=DSC[ds], mfc=DSC[ds] if f == 1 else "white", mew=0.7)
    ax.axhline(-np.log10(0.05), color="0.5", lw=0.5, ls=(0, (2.5, 1.5)))
    ax.text(4.45, -np.log10(0.05) + 0.08, "p = 0.05", fontsize=5.3, color="0.4", ha="right", va="bottom")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v[0] for v in variants])
    ax.set_ylabel("−log$_{10}$ p (sign-flip, subjects)")
    ax.set_ylim(0, floor + 0.5)
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color="0.2", ms=3.4), Line2D([], [], marker="s", ls="none", color="0.2", mfc="white", ms=3.4),
                       Line2D([], [], marker="o", ls="none", color=DSC["EESM19"], ms=3.4), Line2D([], [], marker="o", ls="none", color=DSC["EESM23"], ms=3.4),
                       Line2D([], [], marker="^", ls="none", color="0.2", ms=3.4)],
              labels=["f2: depth asymmetry", "f1: rotation", "EESM19", "EESM23", "at permutation floor"], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.5)
    ax.set_title("Real data, cycle-structure control (R13)", loc="left", fontsize=6.5, color="0.3")
    plab(ax, "b", x=-0.14)
    save(fig, "Figure5_cycle_control")


def figureS1():
    inc19 = pd.read_csv(os.path.join(str(PROC), "epoch_inclusion_v2.csv.gz"))
    inc23 = pd.read_csv(os.path.join(str(PROC), "epoch_inclusion_v23.csv.gz"))
    CH = ["C3-M2", "F3-M2", "O1-M2"]
    fig = plt.figure(figsize=(183 * MM, 112 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.0], hspace=0.62, wspace=0.3, width_ratios=[2.6, 1.0])
    for j, (ds, inc) in enumerate((("EESM19", inc19), ("EESM23", inc23))):
        ax = fig.add_subplot(gs[0, j])
        Rr = inc.groupby(["night", "ch"]).keep_epoch.mean().unstack()[CH].sort_index()
        okc = inc.groupby(["night", "ch"]).channel_ok.first().unstack()[CH].reindex(Rr.index)
        im = ax.imshow(Rr.T.values * 100, aspect="auto", cmap="Blues", vmin=50, vmax=100, interpolation="nearest")
        for jj, c in enumerate(CH):
            for i in np.where(~okc[c].values.astype(bool))[0]:
                ax.plot(i, jj, marker="x", color="#D55E00", ms=3.2, mew=0.8)
        subs = [n.split("_")[0].replace("sub-", "") for n in Rr.index]
        tick = [i for i in range(len(subs)) if i == 0 or subs[i] != subs[i - 1]][::1 if ds == "EESM23" else 2]
        ax.set_xticks(tick)
        ax.set_xticklabels([str(int(subs[i])) for i in tick])
        ax.set_yticks(range(3))
        ax.set_yticklabels(CH)
        ax.set_xlabel("Subject (nights in order)")
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(ds, loc="left")
        if j == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04, ticks=[50, 75, 100])
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=1.5, width=0.4)
            cb.set_label("Epochs kept (%)", fontsize=6)
        plab(ax, "ab"[j], x=-0.1 if j == 0 else -0.3, y=1.08)
    ax = fig.add_subplot(gs[1, 0])
    for ds, vdir, inc in (("EESM19", "v2", inc19), ("EESM23", "v23", inc23)):
        okn = inc[(inc.ch == "C3-M2")].groupby("night").channel_ok.first()
        P, f = [], None
        for n in okn[okn.astype(bool)].index:
            fn = os.path.join(str(PROC), vdir, n + ".npz")
            if not os.path.exists(fn):
                continue
            z = np.load(fn)
            f = z["psd_f"]
            p = z["psd"][0]
            if np.isfinite(p).all():
                P.append(p)
        P = np.array(P)
        m = (f >= 0.5) & (f <= 40)
        q1, md, q3 = np.percentile(P[:, m], [25, 50, 75], axis=0)
        ax.fill_between(f[m], q1, q3, color=DSC[ds], alpha=0.18, lw=0)
        ax.plot(f[m], md, color=DSC[ds], lw=0.9, label="%s (%d nights)" % (ds, len(P)))
    ax.axvspan(25, 40, color="0.95", lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("C3-M2 PSD (µV² Hz⁻¹)")
    ax.legend(loc="lower left")
    ax.set_title("Median and IQR over included nights; shaded: outside the 0.5–25 Hz feature band", loc="left", fontsize=5.8, color="0.3")
    plab(ax, "c", x=-0.1)
    ax = fig.add_subplot(gs[1, 1])
    for ds, inc in (("EESM19", inc19), ("EESM23", inc23)):
        U = inc[(inc.ch == "C3-M2") & (inc.bad_frac <= 0.10) & ~inc.outside.astype(bool) & inc.med_sd_uV.notna()]
        vv = np.sort(U.med_sd_uV.values)
        ax.plot(vv, np.arange(1, len(vv) + 1) / len(vv), color=DSC[ds], lw=0.9, label=ds)
    for thr_, ls_ in ((100, (0, (1, 1.2))), (150, "-"), (200, (0, (1, 1.2)))):
        ax.axvline(thr_, color="#D55E00", lw=0.6 if thr_ == 150 else 0.5, ls=ls_)
    ax.text(150, 0.06, "150 µV", rotation=90, fontsize=5.3, color="#D55E00", ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Median per-second s.d. (µV)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="upper left")
    plab(ax, "d", x=-0.3)
    save(fig, "FigureS1_data_quality")


def figureS2():
    L = pd.read_csv(os.path.join(str(PROC), "labels_eesm19.csv"))
    G = L[L.grid_diff_s.abs() < 1]
    a = pd.to_numeric(G.scorer1, errors="coerce")
    b = pd.to_numeric(G.scorer2, errors="coerce")
    ok = a.isin([1, 2, 3, 4, 5]) & b.isin([1, 2, 3, 4, 5])
    G, a, b = G[ok], a[ok].astype(int), b[ok].astype(int)
    order = [1, 3, 4, 5, 2]
    C = pd.crosstab(a, b).reindex(index=order, columns=order, fill_value=0).to_numpy().astype(float)
    Pm = C / C.sum(1, keepdims=True)
    fig = plt.figure(figsize=(120 * MM, 52 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(Pm * 100, cmap="Greys", vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "%.0f" % (Pm[i, j] * 100), ha="center", va="center", fontsize=5.5, color="white" if Pm[i, j] > 0.55 else "black")
    ax.set_xticks(range(5))
    ax.set_xticklabels([SN[k] for k in order])
    ax.set_yticks(range(5))
    ax.set_yticklabels([SN[k] for k in order])
    ax.set_xlabel("Scorer 2")
    ax.set_ylabel("Scorer 1")
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    plab(ax, "a", x=-0.42)
    ax = fig.add_subplot(gs[1])
    kap = []
    for (s, e), gq in G.groupby(["sub", "ses"]):
        kap.append((int(str(s)[-3:]), cohen_kappa_score(a.loc[gq.index], b.loc[gq.index])))
    kap = np.array(kap)
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(kap))
    ax.scatter(kap[:, 0] + jit, kap[:, 1], s=6, color="0.25", linewidths=0)
    ax.axhline(np.median(kap[:, 1]), color="#D55E00", lw=0.6, ls=(0, (2.5, 1.5)))
    ax.text(20.6, np.median(kap[:, 1]), "median %.2f" % np.median(kap[:, 1]), fontsize=5.3, color="#D55E00", va="center")
    ax.set_xticks(range(1, 21, 2))
    ax.set_xlim(0.3, 20.7)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Subject (EESM19)")
    ax.set_ylabel("Cohen's κ, scorer 1 vs 2")
    plab(ax, "b", x=-0.25)
    save(fig, "FigureS2_scorer_agreement")


for fn in (figure1, figure2, figure3, figure4, figure5, figureS1, figureS2):
    print("\n" + fn.__name__)
    fn()

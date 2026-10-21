"""
skill_gap.py
============
Skill-gap analysis for a TARGET JOB ROLE, driven entirely by YOUR skills_registry.py
(SKILLS, ROLE_PROFILES, AMBIGUOUS_TITLES, KITS) and employability_core.

    target role  ->  required skills (curated profile, or a job description you paste)
                 ->  skills the student has (resume detections + confirmations)
                 ->  gap table + readiness score
                 ->  prerequisite-ordered learning plan from KITS (project, resource, milestone, resume wording)
                 ->  profile advice  ->  (optional) what-if with the saved model

Nothing here fits or retrains a model. Curated ROLE_PROFILES are a fallback that is NOT derived from
job postings; a pasted job description (verified=True) takes priority, as the registry specifies.
"""
import re
import difflib

import numpy as np
import pandas as pd

from skills_registry import (SKILLS, ROLE_PROFILES, AMBIGUOUS_TITLES, KITS,
                             MODEL_TECH_SKILL_KEYS, display_name, domain_of)

TIERS = ("core", "important", "complementary")
TIER_LABEL = {"core": "Core", "important": "Important", "complementary": "Complementary"}
TIER_WEIGHT = {"core": 3, "important": 2, "complementary": 1}


# =============================================================================
# 1. NAME -> REGISTRY KEY
# =============================================================================
def _alias_map():
    m = {}
    for k, s in SKILLS.items():                               # exact names first
        for n in (k, k.replace("_", " "), s["display"]):
            m.setdefault(n.strip().lower(), k)
    for k, s in SKILLS.items():                               # then aliases
        for a in s["aliases"]:
            m.setdefault(a.strip().lower(), k)
    return m


_ALIASES = _alias_map()


def resolve_skill_names(names):
    """['python', 'Power BI', 'foo'] -> (['Python', 'PowerBI'], ['foo'])"""
    keys, unknown = [], []
    for n in names:
        k = _ALIASES.get(str(n).strip().lower())
        (keys if k else unknown).append(k or str(n).strip())
    return list(dict.fromkeys(keys)), unknown


# =============================================================================
# 2. ROLES
# =============================================================================
def _req(item, tier):
    """Registry item (key | (label, [keys])) -> normalised requirement."""
    if isinstance(item, (tuple, list)):
        label, options = item
        return dict(label=label, options=list(options), tier=tier)
    return dict(label=display_name(item), options=[item], tier=tier)


def _role_from_profile(rkey):
    p = ROLE_PROFILES[rkey]
    return dict(key=rkey, title=p["title"], verified=False, source="Curated reference profile (not derived from job postings)",
                reqs=[_req(i, t) for t in TIERS for i in p[t]], notes=[])


def list_roles():
    return sorted(p["title"] for p in ROLE_PROFILES.values())


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9+# ]+", " ", str(s).lower()).split())


def resolve_role(query, top=4):
    """Ranked [(role_key, score)] for free text such as 'ml engg', 'sde', 'react developer'."""
    q = _norm(query)
    if not q:
        return []
    if q in AMBIGUOUS_TITLES:
        return [(k, 1.0) for k in AMBIGUOUS_TITLES[q]]
    scores = {}
    for rk, p in ROLE_PROFILES.items():
        best = 0.0
        for name in [p["title"]] + p["aliases"]:
            n = _norm(name)
            if q == n:
                best = 1.0
                break
            if min(len(q), len(n)) >= 4 and (n in q or q in n):
                best = max(best, 0.9)
            best = max(best, difflib.SequenceMatcher(None, q, n).ratio())
            qt, nt = set(q.split()), set(n.split())
            if qt and nt:
                best = max(best, 0.85 * len(qt & nt) / len(qt | nt))
        scores[rk] = best
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(r, round(s, 2)) for r, s in ranked[:top] if s >= 0.45]


def role_from_skills(title, skill_names, important=()):
    """A role you define by typing skills (names, keys or aliases). Unknown names are reported, never scored."""
    core, unk1 = resolve_skill_names(skill_names)
    imp, unk2 = resolve_skill_names(important)
    imp = [k for k in imp if k not in core]
    return dict(key="custom", title=title, verified=False, source="Skills typed by you",
                reqs=[_req(k, "core") for k in core] + [_req(k, "important") for k in imp],
                notes=([f"Not in your registry, so they cannot be checked automatically: {unk1 + unk2}"] if unk1 + unk2 else []))


_PREF_RX = re.compile(r"(?i)\b(preferred|nice[- ]to[- ]have|good[- ]to[- ]have|bonus|desirable|a plus|added advantage|optional)\b")
_REQ_RX = re.compile(r"(?i)\b(required|must[- ]have|requirements?|qualifications?|responsibilit\w+|what you.ll need|mandatory)\b")


def role_from_job_description(title, jd_text, ec):
    """
    Required skills = registry skills found in the job description (verified=True).
    Lines under / containing 'preferred / nice to have / bonus ...' become Complementary; everything else Core.
    Ambiguous one-letter tokens (R, Go, C ...) are skipped unless you add them by hand.
    """
    tier_of, mode, skipped = {}, "core", set()
    for line in jd_text.splitlines():
        if not line.strip():
            continue
        short = len(line.strip()) <= 70
        if _PREF_RX.search(line) and short:
            mode = "complementary"
        elif _REQ_RX.search(line) and short:
            mode = "core"
        line_mode = "complementary" if _PREF_RX.search(line) else mode
        for k, d in ec.detect_skills(line).items():
            if d["confidence"] != "high":
                skipped.add(display_name(k))
                continue
            if tier_of.get(k) != "core":                       # a skill seen as required anywhere stays Core
                tier_of[k] = line_mode
    notes = []
    if skipped:
        notes.append(f"Ambiguous tokens skipped (add by hand if needed): {sorted(skipped)}")
    if not tier_of:
        notes.append("No registry skills were found in the job description.")
    return dict(key="jd", title=title, verified=True, source="Your job description",
                reqs=[_req(k, t) for k, t in tier_of.items()], notes=notes)


def choose_role(query, interactive=True):
    """Returns a normalised role dict. Asks a short question if the title is generic/ambiguous or unknown."""
    cands = resolve_role(query)
    generic = _norm(query) in AMBIGUOUS_TITLES
    if cands and not generic and (cands[0][1] == 1.0 or len(cands) == 1 or cands[0][1] - cands[1][1] >= 0.1):
        return _role_from_profile(cands[0][0])
    if not interactive:
        if cands:
            print(f"'{query}' is ambiguous; using '{ROLE_PROFILES[cands[0][0]]['title']}'. "
                  f"Other options: {[ROLE_PROFILES[c[0]]['title'] for c in cands[1:]]}")
            return _role_from_profile(cands[0][0])
        raise ValueError(f"Role '{query}' is not in ROLE_PROFILES. Built-in roles: {list_roles()}. "
                         "Use role_from_skills(...) or set a job description.")
    if cands:
        print(f"'{query}' could mean several roles:")
        for i, (r, _) in enumerate(cands, 1):
            print(f"  {i}. {ROLE_PROFILES[r]['title']}")
        print("  0. None of these (define my own)")
        pick = input("Choose a number: ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(cands):
            return _role_from_profile(cands[int(pick) - 1][0])
    else:
        print(f"'{query}' is not a built-in role. Built-in roles: {', '.join(list_roles())}")
    print("Define it yourself: type the skills it needs (or stop and set JOB_DESCRIPTION to paste a posting).")
    raw = input("Skills, comma-separated (or press Enter to stop): ").strip()
    if not raw:
        raise ValueError("No role selected. Set JOB_DESCRIPTION in the notebook to paste a posting instead.")
    return role_from_skills(query.strip().title(), [s.strip() for s in raw.split(",") if s.strip()])


# =============================================================================
# 3. SKILLS THE STUDENT HAS
# =============================================================================
def collect_skills(ex=None, profile=None, manual=(), not_have=()):
    """
    Returns (have, unconfirmed, unresolved)
      have        {key: dict(source, confidence, evidence)}
      unconfirmed low-confidence resume tokens (R, Go, C, Excel...) NOT yet counted
      unresolved  manual names that are not in the registry
    With a StudentProfile (step 6) its present/absent ticks are authoritative.
    """
    det = dict((ex or {}).get("skill_detections", {}) or {})
    ev = lambda k: list(det.get(k, {}).get("evidence", []))
    have, unconfirmed = {}, []
    if profile is not None:
        absent = {k for k, v in profile.skill_state.items() if v == "absent"}
        for k in profile.present_skills():
            have[k] = dict(source="confirmed in your profile", confidence=det.get(k, {}).get("confidence", "confirmed"), evidence=ev(k))
        for k, d in det.items():
            if k in have or k in absent:
                continue
            if d["confidence"] == "high":
                have[k] = dict(source="resume", confidence="high", evidence=ev(k))
            else:
                unconfirmed.append(k)
    else:
        for k, d in det.items():
            if d["confidence"] == "high":
                have[k] = dict(source="resume", confidence="high", evidence=ev(k))
            else:
                unconfirmed.append(k)
    mkeys, unresolved = resolve_skill_names(manual)
    for k in mkeys:
        have[k] = dict(source="added by you", confidence="confirmed", evidence=[display_name(k)])
        if k in unconfirmed:
            unconfirmed.remove(k)
    nkeys, _ = resolve_skill_names(not_have)
    for k in nkeys:
        have.pop(k, None)
        if k in unconfirmed:
            unconfirmed.remove(k)
    return have, sorted(unconfirmed), unresolved


def skill_inventory(have):
    """{domain: [display names]} for every skill the student has."""
    g = {}
    for k in have:
        g.setdefault(domain_of(k), []).append(display_name(k))
    return {d: sorted(v) for d, v in sorted(g.items())}


# =============================================================================
# 4. GAP ANALYSIS
# =============================================================================
def missing_prereqs(key, have, _seen=None):
    """Registry prerequisites (KITS) the student still lacks, deepest first."""
    _seen = _seen if _seen is not None else set()
    out = []
    for p in KITS.get(key, {}).get("prereq", []):
        if p in have or p in _seen:
            continue
        _seen.add(p)
        out += missing_prereqs(p, have, _seen) + [p]
    return out


def analyze_gap(role, have):
    rows = []
    for r in role["reqs"]:
        hit = [o for o in r["options"] if o in have]
        if hit:
            suggest = None
        else:   # any-of group: start with the option that needs the fewest missing prerequisites
            suggest = min(r["options"], key=lambda o: (len(missing_prereqs(o, have)), r["options"].index(o)))
        rows.append(dict(requirement=r["label"], tier=r["tier"], status="Have" if hit else "Missing",
                         matched=", ".join(display_name(o) for o in hit),
                         evidence=", ".join(have[hit[0]]["evidence"]) if hit else "",
                         learn_key=suggest, learn_first=display_name(suggest) if suggest else None,
                         options=r["options"], weight=TIER_WEIGHT[r["tier"]]))
    t = pd.DataFrame(rows, columns=["requirement", "tier", "status", "matched", "evidence", "learn_key",
                                    "learn_first", "options", "weight"])
    if t.empty:
        return dict(table=t, score=0.0, label="No requirements to compare.", pct={}, missing=t, have_rows=t)
    pct = {tier: (round(100 * (d["status"] == "Have").mean(), 1) if len(d) else None)
           for tier in TIERS for d in [t[t.tier == tier]]}
    score = round(100 * float((t.weight * (t.status == "Have")).sum()) / float(t.weight.sum()), 1)
    label = ("Strong match: focus on projects, depth and interview practice" if score >= 75 else
             "Good foundation: close the remaining gaps before applying widely" if score >= 50 else
             "Early stage: build the core skills first (see the plan)" if score >= 25 else
             "Beginner for this role: start with the fundamentals in the plan")
    return dict(table=t, score=score, label=label, pct=pct,
                missing=t[t.status == "Missing"], have_rows=t[t.status == "Have"])


# =============================================================================
# 5. LEARNING PLAN (KITS, prerequisites first)
# =============================================================================
def _resource_text(kit):
    name, url = kit["resource"]
    return f"{name}: {url}" if url else f"{name} (no link stored: search the official documentation)"


def build_plan(gap, have):
    """Prerequisite-ordered steps for every missing requirement, Core -> Important -> Complementary."""
    order, seen = [], set(have)
    phase_name = {"core": "Phase 1: Core", "important": "Phase 2: Important", "complementary": "Phase 3: Complementary"}

    def add(key, tier, reason):
        if key in seen:
            return
        for p in KITS.get(key, {}).get("prereq", []):
            add(p, tier, f"prerequisite for {display_name(key)}")
        seen.add(key)
        kit = KITS.get(key)
        order.append(dict(
            step=len(order) + 1, phase=phase_name[tier], skill=display_name(key), key=key, tier=TIER_LABEL[tier], why=reason,
            project=kit["project"] if kit else f"Build a small project using {display_name(key)}.",
            resource=_resource_text(kit) if kit else "Search the official documentation.",
            milestone=kit["milestone"] if kit else "Use it in one working project and publish it.",
            resume_wording=kit["resume"] if kit else f"Used {display_name(key)} in a project"))

    miss = gap["missing"]
    for tier in TIERS:
        for r in miss[miss.tier == tier].itertuples():
            add(r.learn_key, tier, f"{TIER_LABEL[tier].lower()} requirement: {r.requirement}")
    return pd.DataFrame(order, columns=["step", "phase", "skill", "key", "tier", "why", "project", "resource", "milestone", "resume_wording"])


# =============================================================================
# 6. PROFILE ADVICE
# =============================================================================
def profile_advice(role, gap, plan, pv=None, ex=None):
    pv, ex, out = pv or {}, ex or {}, []
    if pv.get("Relevant_Project_Count") is not None and pv["Relevant_Project_Count"] < 2:
        out.append(f"Role-relevant projects: you report {int(pv['Relevant_Project_Count'])}. Aim for 2-3 that clearly target "
                   f"{role['title']}; the project in each learning step below is a ready-made choice.")
    elif pv.get("Project_Count") is not None and pv["Project_Count"] < 2:
        out.append(f"Projects: you report {int(pv['Project_Count'])}. Aim for at least 2-3, using the projects in the plan.")
    if pv.get("Internship") == 0:
        out.append("No internship: look for internships or virtual internships and keep a written record of what you shipped.")
    if pv.get("Certification_Count") == 0 or (pv.get("Certification_Count") is None and ex.get("certification_count_estimate") == 0):
        out.append("No certification listed: completing a plan milestone from an official course and listing it is a practical start.")
    if pv.get("Hackathon_Participation") == 0 or (pv.get("Hackathon_Participation") is None and ex.get("hackathon_mention") is False):
        out.append("No hackathon or competition: one team event shows applied skill and collaboration.")
    if pv.get("Open_Source_Contribution") == 0:
        out.append("No open-source contribution: a small documentation or bug-fix pull request is an achievable first one.")
    if pv.get("Backlogs") not in (None, 0):
        out.append("Backlogs: clear them early; many recruiters require zero active backlogs.")
    if pv.get("CGPA") is not None and pv["CGPA"] < 6.5:
        out.append("CGPA is on the lower side: some recruiters use eligibility cut-offs, so offset it with strong, visible projects.")
    soft = {c: pv[c] for c in ("Communication", "Teamwork", "Leadership", "Problem_Solving", "Presentation", "Time_Management")
            if pv.get(c) is not None}
    if soft:
        low = sorted(soft, key=soft.get)[:2]
        out.append(f"Soft skills: your lowest self-ratings are {', '.join(c.replace('_', ' ') for c in low)}; "
                   "practise them deliberately (e.g. present one project in 2 minutes; use STAR stories for interviews).")
    else:
        out.append("Soft skills: prepare a 2-minute project pitch and 3 STAR-format stories; interviews test communication and teamwork.")
    if len(plan):
        out.append("Resume: add each skill only after you reach its milestone, using the suggested wording in the plan.")
    return out


# =============================================================================
# 7. OPTIONAL: WHAT-IF WITH THE SAVED MODEL
# =============================================================================
def what_if_model(ec, model, pre, X_raw, gap):
    """
    Re-score the SAME student row after switching individual missing skills to 'present'. Only skills that
    are model input columns can be tested. The model was trained on synthetic data: this illustrates how the
    model responds, not what real recruiters will do.
    """
    base = ec.predict_employability(model, pre, X_raw)["p_placed"]
    rows, keys_all, keys_core = [], [], []
    for r in gap["missing"].itertuples():
        k = r.learn_key
        if k not in X_raw.columns or k not in MODEL_TECH_SKILL_KEYS or X_raw.iloc[0][k] == 1:
            continue
        X2 = X_raw.copy()
        X2.loc[X2.index[0], k] = 1.0
        try:
            p2 = ec.predict_employability(model, pre, X2)["p_placed"]
        except Exception:
            continue
        rows.append(dict(skill=display_name(k), tier=TIER_LABEL[r.tier], p_placed_now=base, p_placed_if_added=p2, change=p2 - base))
        keys_all.append(k)
        if r.tier == "core":
            keys_core.append(k)

    def combined(keys):
        if not keys:
            return None
        X3 = X_raw.copy()
        for k in keys:
            X3.loc[X3.index[0], k] = 1.0
        return ec.predict_employability(model, pre, X3)["p_placed"]

    tbl = pd.DataFrame(rows, columns=["skill", "tier", "p_placed_now", "p_placed_if_added", "change"])
    return dict(base=base, table=tbl.sort_values("change", ascending=False).reset_index(drop=True),
                all_added=combined(keys_all), core_added=combined(keys_core))


# =============================================================================
# 8. ORCHESTRATION, PRINTING, CHARTS, REPORT
# =============================================================================
def run_skill_gap(role_query, ec=None, ex=None, profile=None, manual_skills=(), not_have=(),
                  job_description=None, interactive=True, profile_values=None):
    if job_description and job_description.strip():
        if ec is None:
            raise ValueError("Pass ec (employability_core) to parse a job description.")
        role = role_from_job_description(role_query or "Target role (from job description)", job_description, ec)
    else:
        role = choose_role(role_query, interactive=interactive)
    have, unconfirmed, unresolved = collect_skills(ex, profile, manual_skills, not_have)
    gap = analyze_gap(role, have)
    plan = build_plan(gap, have) if len(gap["table"]) else pd.DataFrame()
    pv = dict(profile_values or {})
    if profile is not None:
        for c in ("CGPA", "Backlogs", "Internship", "Project_Count", "Relevant_Project_Count", "Certification_Count",
                  "Hackathon_Participation", "Open_Source_Contribution",
                  "Communication", "Teamwork", "Leadership", "Problem_Solving", "Presentation", "Time_Management"):
            if profile.value(c) is not None:
                pv.setdefault(c, profile.value(c))
    advice = profile_advice(role, gap, plan, pv, ex)
    return dict(role=role, have=have, unconfirmed=unconfirmed, unresolved=unresolved,
                inventory=skill_inventory(have), gap=gap, plan=plan, advice=advice)


def print_report(res):
    role, g, plan = res["role"], res["gap"], res["plan"]
    bar = lambda p: "#" * int(round((p or 0) / 5)) + "-" * (20 - int(round((p or 0) / 5)))
    print("=" * 74)
    print(f"SKILL-GAP REPORT  |  Target role: {role['title']}")
    print(f"Requirements source: {role['source']}" + ("" if role["verified"] else "  [verified=False]"))
    print("=" * 74)
    for n in role["notes"]:
        print("NOTE:", n)
    if g["table"].empty:
        print("No requirements were found for this role, so no gap analysis is possible.")
        return
    print(f"Role readiness score : {g['score']:.0f}/100  [{bar(g['score'])}]")
    for t in TIERS:
        if g["pct"].get(t) is not None:
            print(f"  {TIER_LABEL[t]:<14}: {g['pct'][t]}% covered")
    print(f"  Verdict        : {g['label']}")
    print("The score = share of this role's skills found in your resume/confirmations (core x3, important x2, "
          "complementary x1). It is a planning aid, not a hiring probability.")
    if not role["verified"]:
        print("The role list is a curated fallback, not taken from real postings: paste a job description "
              "(JOB_DESCRIPTION) for a requirement list you can verify.")

    print("\n--- 1. SKILLS YOU HAVE (by domain) ---")
    for d, items in res["inventory"].items():
        print(f"  {d}: {', '.join(items)}")
    if not res["inventory"]:
        print("  (none detected yet: add them with MANUAL_SKILLS)")
    low = [display_name(k) for k, v in res["have"].items() if v["confidence"] == "low"]
    if low:
        print(f"  Low-confidence matches counted (please verify): {', '.join(low)}")
    if res["unconfirmed"]:
        print(f"  Possible but NOT counted until you confirm (add to MANUAL_SKILLS): "
              f"{', '.join(display_name(k) for k in res['unconfirmed'])}")
    if res["unresolved"]:
        print(f"  Typed skills not in the registry (ignored): {', '.join(res['unresolved'])}")

    print("\n--- 2. WHAT YOU ALREADY COVER FOR THIS ROLE ---")
    for r in g["have_rows"].itertuples():
        print(f"  [OK] {r.requirement} ({TIER_LABEL[r.tier]})" + (f"  <- {r.evidence}" if r.evidence else ""))
    if g["have_rows"].empty:
        print("  (nothing yet)")

    print("\n--- 3. YOUR GAPS ---")
    for t in TIERS:
        d = g["missing"][g["missing"].tier == t]
        if d.empty:
            continue
        print(f"  {TIER_LABEL[t]}:")
        for r in d.itertuples():
            extra = f"  (suggested start: {r.learn_first})" if len(r.options) > 1 else ""
            print(f"    [GAP] {r.requirement}{extra}")
    if g["missing"].empty:
        print("  None: every requirement for this role is covered.")

    print("\n--- 4. LEARNING PLAN (prerequisites first) ---")
    if plan.empty:
        print("  No gaps to plan for. Go deeper: ship role-relevant projects and rehearse interviews.")
    for phase, d in plan.groupby("phase", sort=False):
        print(f"  {phase}")
        for r in d.itertuples():
            print(f"   {r.step}. {r.skill}  ({r.why})")
            print(f"      Project   : {r.project}")
            print(f"      Resource  : {r.resource}")
            print(f"      Milestone : {r.milestone}")
            print(f"      Resume    : \"{r.resume_wording}\"  (add only after the milestone)")

    print("\n--- 5. PROFILE & RESUME ADVICE ---")
    for a in res["advice"]:
        print(f"  - {a}")
    known = [r.requirement for r in g["have_rows"].itertuples() if r.tier == "core"]
    if known:
        print("\n--- 6. INTERVIEW PREPARATION ---")
        print("  Be ready to explain and demonstrate, with a project example each: " + ", ".join(known) + ".")


def plot_gap(res):
    import matplotlib.pyplot as plt
    g = res["gap"]
    if g["table"].empty:
        return None
    t = g["table"].copy()
    t["o"] = t["tier"].map({"core": 0, "important": 1, "complementary": 2})
    t = t.sort_values(["o", "status"]).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, max(3.5, 0.34 * len(t) + 1.2)), gridspec_kw=dict(width_ratios=[1, 2]))
    names = ["Overall score"] + [TIER_LABEL[x] for x in TIERS if g["pct"].get(x) is not None]
    vals = [g["score"]] + [g["pct"][x] for x in TIERS if g["pct"].get(x) is not None]
    pal = {"Overall score": "#8ab17d", "Core": "#264653", "Important": "#e9c46a", "Complementary": "#e76f51"}
    ax[0].barh(names[::-1], vals[::-1], color=[pal[n] for n in names[::-1]])
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("% covered")
    ax[0].set_title(f"Readiness: {res['role']['title']}", fontsize=10)
    for i, v in enumerate(vals[::-1]):
        ax[0].text(min(v + 2, 86), i, f"{v:.0f}%", va="center", fontsize=9)
    ax[1].barh([f"{r.requirement[:48]} ({ {'core': 'core', 'important': 'imp.', 'complementary': 'comp.'}[r.tier] })" for r in t.itertuples()], [1] * len(t),
               color=["#2a9d8f" if s == "Have" else "#e76f51" for s in t["status"]])
    ax[1].set_xticks([])
    ax[1].set_title("Green = you have it   Red = gap", fontsize=9)
    fig.tight_layout()
    return fig


def plot_what_if(wi):
    import matplotlib.pyplot as plt
    t = wi["table"].head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.3))
    ax.barh(t["skill"], t["change"] * 100, color=["#2a9d8f" if v >= 0 else "#e76f51" for v in t["change"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Change in model P(Placed), percentage points")
    ax.set_title("What-if: adding one missing skill (saved model, synthetic data)")
    return fig


def to_markdown(res, what_if=None):
    role, g, plan = res["role"], res["gap"], res["plan"]
    L = [f"# Skill-gap report: {role['title']}", "", f"_Requirements source: {role['source']}"
         + ("" if role["verified"] else " (verified=False)") + "_", ""]
    if g["table"].empty:
        return "\n".join(L + ["No requirements found; no analysis possible."])
    L += [f"**Readiness score:** {g['score']:.0f}/100  ", f"**Verdict:** {g['label']}", "",
          "_Share of this role's skills found in your resume/confirmations (core x3, important x2, complementary x1). "
          "A planning aid, not a hiring probability._", "", "## Skills you have"]
    L += [f"- **{d}:** {', '.join(v)}" for d, v in res["inventory"].items()]
    L += ["", "## Gaps"]
    for t in TIERS:
        d = g["missing"][g["missing"].tier == t]
        if len(d):
            L.append(f"- **{TIER_LABEL[t]}:** " + ", ".join(d["requirement"]))
    if g["missing"].empty:
        L.append("- None")
    L += ["", "## Learning plan (prerequisites first)"]
    for r in plan.itertuples():
        L += [f"{r.step}. **{r.skill}** ({r.phase}; {r.why})",
              f"   - Project: {r.project}", f"   - Resource: {r.resource}",
              f"   - Milestone: {r.milestone}", f"   - Resume wording (after milestone): \"{r.resume_wording}\""]
    L += ["", "## Profile and resume advice"] + [f"- {a}" for a in res["advice"]]
    if what_if is not None and len(what_if["table"]):
        L += ["", "## Model what-if (synthetic-data model; illustration only)", f"Current model P(Placed): {what_if['base']:.1%}"]
        L += [f"- Adding {r.skill}: {r.p_placed_if_added:.1%} ({r.change * 100:+.1f} pts)" for r in what_if["table"].itertuples()]
    return "\n".join(L)

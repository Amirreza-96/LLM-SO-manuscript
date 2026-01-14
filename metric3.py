import re
from typing import Dict, Tuple, List, Set, Optional

from sympy import sympify, simplify
from sympy.core.sympify import SympifyError


def eval1(true_model: str, generated_model: str):
   

    def split_top_level_args(s: str) -> List[str]:
        args, buf = [], []
        dp = db = dc = 0
        in_str: Optional[str] = None
        esc = False
        for ch in s:
            if in_str is not None:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue

            if ch in ("'", '"'):
                in_str = ch
                buf.append(ch)
                continue

            if ch == "(":
                dp += 1
            elif ch == ")":
                dp -= 1
            elif ch == "[":
                db += 1
            elif ch == "]":
                db -= 1
            elif ch == "{":
                dc += 1
            elif ch == "}":
                dc -= 1

            if ch == "," and dp == 0 and db == 0 and dc == 0:
                a = "".join(buf).strip()
                if a:
                    args.append(a)
                buf = []
            else:
                buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            args.append(tail)
        return args

    def find_calls(code: str, method_names: Tuple[str, ...]) -> List[str]:
        out = []
        pat = re.compile(r"\.\s*(" + "|".join(map(re.escape, method_names)) + r")\s*\(")
        for m in pat.finditer(code):
            i = m.end()
            depth = 1
            in_str: Optional[str] = None
            esc = False
            start = i
            while i < len(code) and depth > 0:
                ch = code[i]
                if in_str is not None:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == in_str:
                        in_str = None
                else:
                    if ch in ("'", '"'):
                        in_str = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                i += 1
            if depth == 0:
                out.append(code[start : i - 1])
        return out

    def strip_generator(expr: str) -> str:
        s = expr.strip()
        for _ in range(3):
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1].strip()
            else:
                break

        dp = db = dc = 0
        in_str: Optional[str] = None
        esc = False
        i = 0
        while i < len(s):
            ch = s[i]
            if in_str is not None:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                i += 1
                continue

            if ch in ("'", '"'):
                in_str = ch
                i += 1
                continue

            if ch == "(":
                dp += 1
            elif ch == ")":
                dp -= 1
            elif ch == "[":
                db += 1
            elif ch == "]":
                db -= 1
            elif ch == "{":
                dc += 1
            elif ch == "}":
                dc -= 1

            if dp == 0 and db == 0 and dc == 0:
                if s.startswith(" for ", i):
                    return s[:i].strip()
                if s.startswith("for ", i):
                    return s[:i].strip()
            i += 1

        return s.strip()

    def parse_addvars(code: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        assign_pat = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*.*?\.addVars\s*\(")
        for m in assign_pat.finditer(code):
            v = m.group(1)
            i = m.end()
            depth = 1
            in_str: Optional[str] = None
            esc = False
            start = i
            while i < len(code) and depth > 0:
                ch = code[i]
                if in_str is not None:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == in_str:
                        in_str = None
                else:
                    if ch in ("'", '"'):
                        in_str = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                i += 1
            if depth == 0:
                args = code[start : i - 1].replace("\n", " ").strip()
                out[v] = args
        return out

    def parse_constraints(code: str) -> List[str]:
        cons = []
        for inside in find_calls(code, ("addConstr", "addConstrs")):
            args = split_top_level_args(inside)
            if args:
                expr0 = strip_generator(args[0].strip())
                cons.append(expr0)
        return cons

    def build_assignment_map(code: str) -> Dict[str, str]:
        amap: Dict[str, str] = {}
        for line in code.splitlines():
            line = line.split("#", 1)[0].strip()
            m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", line)
            if m:
                amap[m.group(1)] = m.group(2).strip()
        return amap

    def strip_outer_sums(expr: str) -> str:
        e = expr.strip()
        for _ in range(3):
            m = re.match(r"^(?:gp\.)?(?:quicksum|sum)\s*\(\s*(.*)\s*\)\s*$", e, flags=re.DOTALL)
            if not m:
                break
            e = m.group(1).strip()
        return e

    def parse_objective(code: str) -> str:
        amap = build_assignment_map(code)
        for inside in find_calls(code, ("setObjective",)):
            args = split_top_level_args(inside)
            if not args:
                continue
            if len(args) >= 2:
                sense = args[1].strip()
                ok = ("MINIMIZE" in sense) or ("MAXIMIZE" in sense)
                if not ok:
                    continue
            obj = args[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", obj) and obj in amap:
                obj = amap[obj]
            return strip_outer_sums(obj)
        return ""

    def normalize_indexing(s: str) -> str:
        out = re.sub(r"\[[^\]]*\]", "[idx]", s)
        out = re.sub(r"\bfor\s+\w+\s+in\b", "for idx in", out)
        out = re.sub(r"\bfor\s+\([^\)]*\)\s+in\b", "for idx in", out)
        return out

    token_re = re.compile(r"([A-Za-z_]\w*|\d+\.\d+|\d+|\[[^\]]*\])", re.DOTALL)

    def normalize_structure(expr: str, target: str) -> str:
        s = " ".join(expr.replace("\n", " ").split())
        s = normalize_indexing(s)

        y_map: Dict[str, str] = {}
        yk = 0
        out_parts: List[str] = []

        i = 0
        while i < len(s):
            m = token_re.match(s, i)
            if not m:
                out_parts.append(s[i])
                i += 1
                continue
            tok = m.group(1)
            if re.fullmatch(r"[A-Za-z_]\w*", tok):
                if tok == target:
                    out_parts.append("_X_")
                else:
                    if tok not in y_map:
                        y_map[tok] = f"_Y{yk}_"
                        yk += 1
                    out_parts.append(y_map[tok])
            else:
                out_parts.append(tok)
            i = m.end()

        return "".join(out_parts)

    def rmsp(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "", s)

    def residualize_relation(expr: str) -> str:
        s = expr.strip()
        dp = db = dc = 0
        in_str: Optional[str] = None
        esc = False
        for i, ch in enumerate(s):
            if in_str is not None:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                continue
            if ch == "(":
                dp += 1
            elif ch == ")":
                dp -= 1
            elif ch == "[":
                db += 1
            elif ch == "]":
                db -= 1
            elif ch == "{":
                dc += 1
            elif ch == "}":
                dc -= 1

            if dp == 0 and db == 0 and dc == 0:
                if s.startswith("<=", i):
                    lhs, rhs = s[:i].strip(), s[i + 2 :].strip()
                    return f"({lhs})-({rhs})"
                if s.startswith(">=", i):
                    lhs, rhs = s[:i].strip(), s[i + 2 :].strip()
                    return f"({rhs})-({lhs})"
                if s.startswith("==", i):
                    lhs, rhs = s[:i].strip(), s[i + 2 :].strip()
                    return f"({lhs})-({rhs})"
        return s

    def normalize_ws(s: str) -> str:
        return " ".join(s.replace("\n", " ").split())

    def split_generator_full(gen: str) -> Tuple[str, List[str]]:
        s = gen.strip()
        dp = db = dc = 0
        in_str: Optional[str] = None
        esc = False
        for_pos: List[int] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if in_str is not None:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in ("'", '"'):
                in_str = ch
                i += 1
                continue
            if ch == "(":
                dp += 1
            elif ch == ")":
                dp -= 1
            elif ch == "[":
                db += 1
            elif ch == "]":
                db -= 1
            elif ch == "{":
                dc += 1
            elif ch == "}":
                dc -= 1

            if dp == 0 and db == 0 and dc == 0:
                if s.startswith(" for ", i):
                    for_pos.append(i)
                    i += 5
                    continue
                if i == 0 and s.startswith("for ", i):
                    for_pos.append(i)
                    i += 4
                    continue
            i += 1

        if not for_pos:
            return s.strip(), []

        first = for_pos[0]
        body = s[:first].strip()
        rest = s[first:].strip()
        parts = []
        dp = db = dc = 0
        in_str = None
        esc = False
        start = 0
        i = 0
        while i < len(rest):
            ch = rest[i]
            if in_str is not None:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                i += 1
                continue
            if ch in ("'", '"'):
                in_str = ch
                i += 1
                continue
            if ch == "(":
                dp += 1
            elif ch == ")":
                dp -= 1
            elif ch == "[":
                db += 1
            elif ch == "]":
                db -= 1
            elif ch == "{":
                dc += 1
            elif ch == "}":
                dc -= 1

            if dp == 0 and db == 0 and dc == 0 and rest.startswith(" for ", i):
                seg = rest[start:i].strip()
                if seg:
                    parts.append(seg)
                i += 5
                start = i
                continue
            i += 1
        tail = rest[start:].strip()
        if tail:
            parts.append(tail)

        cleaned = []
        for seg in parts:
            seg2 = seg.strip()
            if seg2.startswith("for "):
                seg2 = seg2[4:].strip()
            cleaned.append(seg2)
        return body, cleaned

    def canonicalize_sum_call(content: str) -> str:
        body, clauses = split_generator_full(content)
        body = normalize_ws(normalize_indexing(body))
        canon_clauses = []
        for cl in clauses:
            cln = normalize_ws(normalize_indexing(cl))
            canon_clauses.append(cln)
        canon_clauses = sorted(canon_clauses)
        return "SUM(" + body + (";" + ";".join(canon_clauses) if canon_clauses else "") + ")"

    def preprocess_for_sympy(expr: str) -> str:
        s = normalize_ws(expr)
        sum_map: Dict[str, str] = {}
        next_id = 0

        def get_symbol(canon: str) -> str:
            nonlocal next_id
            if canon not in sum_map:
                sum_map[canon] = f"QS{next_id}"
                next_id += 1
            return sum_map[canon]

        i = 0
        out: List[str] = []
        in_str: Optional[str] = None
        esc = False

        pat = re.compile(r"(?:gp\.)?(?:quicksum|sum)\s*\(")

        while i < len(s):
            ch = s[i]
            if in_str is not None:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == in_str:
                    in_str = None
                i += 1
                continue

            if ch in ("'", '"'):
                in_str = ch
                out.append(ch)
                i += 1
                continue

            m = pat.match(s, i)
            if not m:
                out.append(ch)
                i += 1
                continue

            j = m.end()
            depth = 1
            in_str2: Optional[str] = None
            esc2 = False
            start = j
            while j < len(s) and depth > 0:
                cj = s[j]
                if in_str2 is not None:
                    if esc2:
                        esc2 = False
                    elif cj == "\\":
                        esc2 = True
                    elif cj == in_str2:
                        in_str2 = None
                else:
                    if cj in ("'", '"'):
                        in_str2 = cj
                    elif cj == "(":
                        depth += 1
                    elif cj == ")":
                        depth -= 1
                j += 1

            if depth != 0:
                out.append(s[i])
                i += 1
                continue

            inside = s[start : j - 1].strip()
            canon = canonicalize_sum_call(inside)
            sym = get_symbol(canon)
            out.append(sym)
            i = j

        return "".join(out)

    def sympy_eq(a: str, b: str) -> bool:
        try:
            aa = sympify(preprocess_for_sympy(a))
            bb = sympify(preprocess_for_sympy(b))
            return simplify(aa - bb) == 0
        except (SympifyError, TypeError, ValueError):
            return False

    def score_match(true_args: str, gen_args: str) -> int:
        s = 0
        for k in ("lb", "ub", "vtype"):
            m1 = re.search(rf"\b{k}\s*=\s*([^,)\]]+)", true_args)
            m2 = re.search(rf"\b{k}\s*=\s*([^,)\]]+)", gen_args)
            if m1 and m2 and rmsp(m1.group(0)) == rmsp(m2.group(0)):
                s += 1
        return s

    def count_occ(expr: str, var: str) -> int:
        bare = re.findall(rf"(?<![A-Za-z0-9_]){re.escape(var)}(?![A-Za-z0-9_])", expr)
        idxd = re.findall(rf"(?<![A-Za-z0-9_]){re.escape(var)}\s*\[", expr)
        return len(bare) + len(idxd)

    def score_constraints(true_cons: List[str], gen_cons: List[str], t: str, g: str) -> int:
        sc = 0
        for ct in true_cons:
            if count_occ(ct, t) == 0:
                continue
            for cg in gen_cons:
                if count_occ(cg, g) == 0:
                    continue
                nt = normalize_structure(ct, t)
                ng = normalize_structure(cg, g)
                if nt == ng:
                    sc += 2
                if sympy_eq(residualize_relation(nt), residualize_relation(ng)):
                    sc += 2
                if rmsp(nt) == rmsp(ng):
                    sc += 2
        return sc

    def score_objective(true_obj: str, gen_obj: str, t: str, g: str) -> int:
        if not true_obj or not gen_obj:
            return 0
        if count_occ(true_obj, t) == 0 or count_occ(gen_obj, g) == 0:
            return 0
        nt = normalize_structure(true_obj, t)
        ng = normalize_structure(gen_obj, g)
        sc = 0
        if nt == ng:
            sc += 2
        if sympy_eq(residualize_relation(nt), residualize_relation(ng)):
            sc += 2
        if rmsp(nt) == rmsp(ng):
            sc += 2
        return sc

    def score_usage_frequency(true_cons: List[str], gen_cons: List[str], true_obj: str, gen_obj: str, t: str, g: str) -> int:
        tot_t = sum(count_occ(s, t) for s in true_cons) + (count_occ(true_obj, t) if true_obj else 0)
        tot_g = sum(count_occ(s, g) for s in gen_cons) + (count_occ(gen_obj, g) if gen_obj else 0)
        return min(tot_t, tot_g)

    def extract_terms(obj: str, var: str) -> Set[str]:
        s = normalize_indexing(obj)
        pat = re.compile(rf"((?:[A-Za-z_]\w*|\d+(?:\.\d+)?)\s*)\*\s*{re.escape(var)}\s*\[idx\]")
        return {f"{m.group(1).strip()}*{var}[idx]" for m in pat.finditer(s)}

    def score_term_alignment(true_obj: str, gen_obj: str, t: str, g: str) -> int:
        if not true_obj or not gen_obj:
            return 0
        tt = extract_terms(true_obj, t)
        tg = extract_terms(gen_obj, g)
        return 2 * len(tt & tg)

    def replace_whole_identifier(code: str, old: str, new: str) -> str:
        if old == new:
            return code
        pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])")
        return pat.sub(new, code)

    def short(s: str, n: int = 160) -> str:
        ss = " ".join((s or "").replace("\n", " ").split())
        return ss if len(ss) <= n else ss[: n - 3] + "..."

    true_vars = parse_addvars(true_model)
    gen_vars = parse_addvars(generated_model)

    true_cons = parse_constraints(true_model)
    gen_cons = parse_constraints(generated_model)

    true_obj = parse_objective(true_model)
    gen_obj = parse_objective(generated_model)

    all_scores: List[Tuple[int, str, str]] = []
    for g, g_args in gen_vars.items():
        for t, t_args in true_vars.items():
            s = 0
            s += score_match(t_args, g_args)
            s += score_constraints(true_cons, gen_cons, t, g)
            s += score_objective(true_obj, gen_obj, t, g)
            s += score_usage_frequency(true_cons, gen_cons, true_obj, gen_obj, t, g)
            s += score_term_alignment(true_obj, gen_obj, t, g)
            if s > 0:
                all_scores.append((s, g, t))

    all_scores.sort(key=lambda x: -x[0])

    mapping: Dict[str, str] = {}
    used_g: Set[str] = set()
    used_t: Set[str] = set()
    for s, g, t in all_scores:
        if g in used_g or t in used_t:
            continue
        mapping[g] = t
        used_g.add(g)
        used_t.add(t)

    aligned_code = generated_model
    for g in sorted(mapping.keys(), key=len, reverse=True):
        aligned_code = replace_whole_identifier(aligned_code, g, mapping[g])

    gen_vars_hat = parse_addvars(aligned_code)
    gen_cons_hat = parse_constraints(aligned_code)
    gen_obj_hat = parse_objective(aligned_code)

    true_obj_set: Set[str] = {true_obj} if true_obj.strip() else set()
    gen_obj_set: Set[str] = {gen_obj_hat} if gen_obj_hat.strip() else set()

    true_components = {
        "variables": set(true_vars.keys()),
        "constraints": set(true_cons),
        "objective": true_obj_set,
    }
    gen_components = {
        "variables": set(gen_vars_hat.keys()),
        "constraints": set(gen_cons_hat),
        "objective": gen_obj_set,
    }

    def constraint_or_obj_match(true_expr: str, gen_expr: str, target_vars: List[str]) -> bool:
        if normalize_indexing(true_expr) == normalize_indexing(gen_expr):
            return True
        for t in target_vars:
            nt = normalize_structure(true_expr, t)
            ng = normalize_structure(gen_expr, t)
            if nt == ng:
                return True
            if rmsp(nt) == rmsp(ng):
                return True
            if sympy_eq(residualize_relation(nt), residualize_relation(ng)):
                return True
        return False

    target_vars = sorted(list(true_components["variables"]))

    T_vars = true_components["variables"]
    G_vars = gen_components["variables"]
    matched_vars = T_vars.intersection(G_vars)
    matched_vars_count = len(matched_vars)
    extra_gen_vars = G_vars - matched_vars

    percent_matched_variables = (matched_vars_count / len(T_vars) * 100.0) if len(T_vars) > 0 else 0.0

    T_cons = true_components["constraints"]
    G_cons = gen_components["constraints"]

    matched_constraint_pairs: List[Tuple[str, str]] = []
    matched_true_constraints: Set[str] = set()
    matched_gen_constraints: Set[str] = set()
    used_gen_constraints: Set[str] = set()

    for te in T_cons:
        for ge in G_cons:
            if ge in used_gen_constraints:
                continue
            if constraint_or_obj_match(te, ge, target_vars):
                matched_true_constraints.add(te)
                matched_gen_constraints.add(ge)
                used_gen_constraints.add(ge)
                matched_constraint_pairs.append((te, ge))
                break

    matched_cons_count = len(matched_true_constraints)
    percent_matched_constraints = (matched_cons_count / len(T_cons) * 100.0) if len(T_cons) > 0 else 0.0

    unmatched_true_constraints = [c for c in list(T_cons) if c not in matched_true_constraints]
    unmatched_gen_constraints = [c for c in list(G_cons) if c not in matched_gen_constraints]

    matched_obj_pairs: List[Tuple[str, str]] = []
    matched_true_obj: Set[str] = set()
    matched_gen_obj: Set[str] = set()
    used_gen_obj: Set[str] = set()

    T_obj = true_components["objective"]
    G_obj = gen_components["objective"]

    for te in T_obj:
        for ge in G_obj:
            if ge in used_gen_obj:
                continue
            if constraint_or_obj_match(te, ge, target_vars):
                matched_true_obj.add(te)
                matched_gen_obj.add(ge)
                used_gen_obj.add(ge)
                matched_obj_pairs.append((te, ge))
                break

    matched_obj_count = len(matched_true_obj)
    percent_matched_objective = (matched_obj_count / len(T_obj) * 100.0) if len(T_obj) > 0 else 0.0

    unmatched_true_obj = [o for o in list(T_obj) if o not in matched_true_obj]
    unmatched_gen_obj = [o for o in list(G_obj) if o not in matched_gen_obj]

    total_true = len(T_vars) + len(T_cons) + len(T_obj)
    total_generated = len(G_vars) + len(G_cons) + len(G_obj)

    total_matches = matched_vars_count + matched_cons_count + matched_obj_count
    total_extra_in_generated = len(extra_gen_vars) + len(unmatched_gen_constraints) + len(unmatched_gen_obj)

    if total_matches > total_true:
        raise RuntimeError(f"BUG: total_matches={total_matches} > total_true={total_true}")

    percent_matched_total = (total_matches / total_true * 100.0) if total_true > 0 else 0.0
    percent_extra_in_generated = (total_extra_in_generated / total_generated * 100.0) if total_generated > 0 else 0.0

    print("=== eval1 debug ===")
    print("Mapping (generated -> true):", mapping)

    print(f"Matched variables:   {matched_vars_count}/{len(T_vars)} -> {sorted(list(matched_vars))}")
    print(f"Extra generated vars ({len(extra_gen_vars)}): {sorted(list(extra_gen_vars))}")

    print(f"Matched constraints: {matched_cons_count}/{len(T_cons)}")
    if matched_constraint_pairs:
        print("Constraint matches (true -> gen):")
        for i, (te, ge) in enumerate(matched_constraint_pairs, 1):
            print(f"  [{i}] TRUE: {short(te)}")
            print(f"      GEN : {short(ge)}")
    else:
        print("Constraint matches (true -> gen): (none)")

    print(f"TRUE constraints ({len(T_cons)}):")
    for i, te in enumerate(list(T_cons), 1):
        tag = "MATCHED" if te in matched_true_constraints else "UNMATCHED"
        print(f"  [T{i}] {tag}: {short(te)}")

    print(f"GEN constraints ({len(G_cons)}):")
    for i, ge in enumerate(list(G_cons), 1):
        tag = "MATCHED" if ge in matched_gen_constraints else "UNMATCHED"
        print(f"  [G{i}] {tag}: {short(ge)}")

    if len(T_obj) == 0:
        print("Matched objective:   N/A (no objective extracted in TRUE model)")
    else:
        print(f"Matched objective:   {matched_obj_count}/{len(T_obj)}")

    if matched_obj_pairs:
        for te, ge in matched_obj_pairs:
            print("Objective match (true -> gen):")
            print("  TRUE:", short(te, 260))
            print("  GEN :", short(ge, 260))
    else:
        if len(T_obj) > 0:
            print("Objective match (true -> gen): (none)")

    if unmatched_true_obj:
        print("Unmatched TRUE objective(s):")
        for te in unmatched_true_obj:
            print("  TRUE:", short(te, 260))
    if unmatched_gen_obj:
        print("Unmatched GEN objective(s):")
        for ge in unmatched_gen_obj:
            print("  GEN :", short(ge, 260))

    print(f"Partial score (eval1): {percent_matched_total:.2f}%")
    print(
        "Scores:",
        f"P_total={percent_matched_total:.2f}%",
        f"P_extra={percent_extra_in_generated:.2f}%",
        f"P_var={percent_matched_variables:.2f}%",
        f"P_cons={percent_matched_constraints:.2f}%",
        f"P_obj={percent_matched_objective:.2f}%",
    )

    return (
        float(percent_matched_total),
        float(percent_extra_in_generated),
        float(percent_matched_variables),
        float(percent_matched_constraints),
        float(percent_matched_objective),
    )

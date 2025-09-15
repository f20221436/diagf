# diagf/log/logparser/logparser/Drain/Drain.py
import os
import re
import csv
import hashlib
import pandas as pd
from tqdm import tqdm


# -----------------------
# Helper structures
# -----------------------
class Node:
    def __init__(self, parent=None, token=None):
        self.parent = parent
        self.token = token
        self.child = dict()
        self.template = None  # Logcluster at leaf


class Logcluster:
    def __init__(self, logTemplate=None, logIDL=None):
        self.logTemplate = logTemplate if logTemplate is not None else []
        self.logIDL = logIDL if logIDL is not None else []
        self.count = len(self.logIDL)


# -----------------------
# Similarity utilities
# -----------------------
def lcs_len(a, b):
    """Longest Common Subsequence length (memory optimized)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(m):
        cur = [0] * (n + 1)
        ai = a[i]
        for j in range(n):
            if ai == b[j]:
                cur[j + 1] = prev[j] + 1
            else:
                cur[j + 1] = prev[j + 1] if prev[j + 1] >= cur[j] else cur[j]
        prev = cur
    return prev[n]


def seq_similarity(a, b):
    """Normalized LCS similarity between token sequences."""
    if not a or not b:
        return 0.0
    lcs = lcs_len(a, b)
    return float(lcs) / max(len(a), len(b))


# -----------------------
# Main Drain Parser
# -----------------------
class LogParser:
    def __init__(self, log_format, indir="./", outdir="./result/",
                 depth=4, st=0.5, rex=None, keep_para=True):
        self.path = indir
        self.savePath = outdir
        self.depth = depth
        self.st = st
        self.rex = rex if rex is not None else []
        self.log_format = log_format
        self.keep_para = keep_para

        # clustering structures
        self.root = Node()
        self.clusters = []
        self.line_counter = 0

        # parse log_format
        self.headers, self._logformat_regex = self.generate_logformat_regex(self.log_format)
        if "Content" in self.headers:
            self.content_header = "Content"
        else:
            self.content_header = self.headers[-1]

        os.makedirs(self.savePath, exist_ok=True)

    # -----------------------
    # format regex utilities
    # -----------------------
    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        regex = ""
        for k in range(len(splitters)):
            if k % 2 == 0:
                literal = splitters[k]
                if literal:
                    literal = re.sub(r"\s+", r"\\s+", re.escape(literal))
                    regex += literal
            else:
                header = splitters[k][1:-1].strip()
                headers.append(header)
                regex += r"(?P<%s>.*?)" % header
        try:
            comp = re.compile("^" + regex + "$")
        except Exception:
            headers = ["Content"]
            comp = re.compile(r"^(?P<Content>.*)$")
        return headers, comp

    # -----------------------
    # high-level parse loop
    # -----------------------
    def parse(self, logName, chunk_size=200000):
        infile = os.path.join(self.path, logName)
        base_out = os.path.join(self.savePath, logName.replace(".prepped", ""))
        structured_out = base_out + "_structured.csv"
        templates_out = base_out + "_templates.csv"

        # prepare structured CSV header
        header_row = ["LineId"] + self.headers + ["EventId", "EventTemplate"]
        if self.keep_para:
            header_row.append("ParameterList")
        with open(structured_out, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(header_row)

        # count total lines
        total_lines = self._count_lines(infile)
        if total_lines == 0:
            pd.DataFrame(columns=["EventId", "EventTemplate", "Occurrences"]).to_csv(templates_out, index=False)
            print("⚠ Empty file:", infile)
            return

        print(f"Processing {infile} ({total_lines} lines) in chunks of {chunk_size} ...")
        with open(infile, "r", encoding="utf-8", errors="ignore") as fin:
            chunk = []
            pbar = tqdm(fin, total=total_lines, desc=f"Parsing {os.path.basename(infile)}",
                        unit="lines", ncols=100)
            for raw in pbar:
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                chunk.append(raw)
                if len(chunk) >= chunk_size:
                    self._process_chunk(chunk, structured_out)
                    chunk = []
            if chunk:
                self._process_chunk(chunk, structured_out)

        # final templates
        self._write_templates(templates_out)
        print(f"✔ Finished: structured -> {structured_out} ; templates -> {templates_out}")

    # -----------------------
    # utilities
    # -----------------------
    def _count_lines(self, filepath):
        cnt = 0
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                cnt += 1
        return cnt

    def _apply_format_extract(self, raw_line):
        m = self._logformat_regex.match(raw_line)
        if not m:
            return None
        return {h: m.group(h) for h in self.headers}

    def _tokenize(self, content):
        return content.split()

    def _normalize_token_for_tree(self, token):
        if any(ch.isdigit() for ch in token):
            return "<*>"
        return token

    # -----------------------
    # chunk processing
    # -----------------------
    def _process_chunk(self, lines, structured_out_path):
        rows_to_write = []
        for raw in lines:
            extracted = self._apply_format_extract(raw)
            if extracted is None:
                extracted = {h: "" for h in self.headers}
                extracted[self.content_header] = raw
            content = extracted.get(self.content_header, raw)

            tokens = self._tokenize(content)
            candidate_node = self._search_tree(tokens)
            assigned_cluster = None
            if candidate_node is not None and candidate_node.template is not None:
                sim = seq_similarity(candidate_node.template.logTemplate, tokens)
                if sim >= self.st:
                    assigned_cluster = candidate_node.template

            if assigned_cluster is None:
                template_tokens = [self._normalize_token_for_tree(t) for t in tokens]
                new_cluster = Logcluster(template_tokens, [self.line_counter + 1])
                new_cluster.count = 1
                self.clusters.append(new_cluster)
                self._insert_into_tree(template_tokens, new_cluster)
                assigned_cluster = new_cluster
            else:
                assigned_cluster.logIDL.append(self.line_counter + 1)
                assigned_cluster.count = len(assigned_cluster.logIDL)

            self.line_counter += 1
            event_id = hashlib.md5(" ".join(assigned_cluster.logTemplate).encode("utf-8")).hexdigest()[0:8]
            event_template_str = " ".join(assigned_cluster.logTemplate)
            row = [self.line_counter] + [extracted.get(h, "") for h in self.headers] + [event_id, event_template_str]
            if self.keep_para:
                param_list = self._extract_parameters(assigned_cluster.logTemplate, tokens)
                row.append(";".join(param_list) if param_list else "")
            rows_to_write.append(row)

        with open(structured_out_path, "a", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerows(rows_to_write)

    # -----------------------
    # prefix tree ops
    # -----------------------
    def _search_tree(self, tokens):
        seq_len = len(tokens)
        if seq_len not in self.root.child:
            return None
        node = self.root.child[seq_len]
        for t in tokens:
            key = self._normalize_token_for_tree(t)
            if key in node.child:
                node = node.child[key]
            elif "<*>" in node.child:
                node = node.child["<*>"]
            else:
                return None
        return node

    def _insert_into_tree(self, template_tokens, logcluster):
        seq_len = len(template_tokens)
        if seq_len not in self.root.child:
            self.root.child[seq_len] = Node(parent=self.root, token=seq_len)
        node = self.root.child[seq_len]
        for t in template_tokens:
            key = t
            if key not in node.child:
                node.child[key] = Node(parent=node, token=key)
            node = node.child[key]
        node.template = logcluster

    # -----------------------
    # parameters + templates
    # -----------------------
    def _extract_parameters(self, template_tokens, content_tokens):
        params = []
        for t_tok, c_tok in zip(template_tokens, content_tokens):
            if t_tok == "<*>":
                params.append(c_tok)
        return params

    def _write_templates(self, templates_out_path):
        rows = []
        for cluster in self.clusters:
            template_str = " ".join(cluster.logTemplate)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
            occurrence = cluster.count if hasattr(cluster, "count") else len(cluster.logIDL)
            rows.append([template_id, template_str, occurrence])
        df = pd.DataFrame(rows, columns=["EventId", "EventTemplate", "Occurrences"])
        df.to_csv(templates_out_path, index=False)

    



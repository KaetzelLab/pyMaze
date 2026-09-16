"""
Behavior Experiment Analyzer (PyQt6)
------------------------------------
Drop one or more experiment .txt files (or click "Add files…") and the
app extracts metadata, parses trial outcomes, and shows a summary table.

Supports TWO file formats automatically:

  Format A — "SA:" tokens grouped by State (e.g. spontAlt tasks)
      P <ts> State: <name>
      P <ts> SA:C
      P <ts> SA:W
      ...
  -> one row per State that contains C/W trials.

  Format B — Trial-numbered with Correct_choice/Incorrect_choice
  (e.g. SpontMaze_RightStart, T-Maze tasks)
      P <ts> Trial number is :N
      P <ts> Correct_choice: ...   (or)  P <ts> Incorrect_choice: ...
      ...
  Each Correct_choice line counts as one C trial, each Incorrect_choice
  line counts as one W trial. The number after the colon is ignored.
  -> one row per file, with State = "<TaskName>".

  Format C — New tab-separated log (columns: time/type/subtype/content)
  (e.g. 282-Box1-2026-07-04-132310.tsv)
      <time>  info    subject_id  282
      <time>  info    start_time  2026-07-04 13:23:10
      <time>  print   task        Correct R (n_correct=1)
      <time>  print   task        Incorrect R (n_incorrect=1)
      ...
      Each "Correct <arm>" print line counts as one C trial, each
      "Incorrect <arm>" print line counts as one W trial.
  -> one row per file, with State = "<TaskName>".

Run:
    pip install PyQt6 pandas openpyxl
    python app.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter, QStatusBar,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

CHOICE_CODES = {"C", "W", "I"}
NUM_CHOICE_COLS = 14


def _parse_metadata(lines):
    meta = {
        "MouseID": "",
        "Group": "",
        "Subgroup": "",
        "ExptDate": "",
        "StartTime": "",
        "TaskName": "",
        "ExperimentName": "",
    }
    for line in lines:
        if not line.startswith("I "):
            continue
        m = re.match(r"I\s+(.+?)\s*:\s*(.*)", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if "Subject ID" in key:
            meta["MouseID"] = val
        elif "Subject Group" in key:
            meta["Group"] = val
        elif "Subject Subgroup" in key:
            meta["Subgroup"] = val
        elif "Start date" in key:
            for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(val, fmt)
                    meta["ExptDate"] = dt.strftime("%Y-%m-%d")
                    meta["StartTime"] = dt.strftime("%H:%M:%S")
                    break
                except ValueError:
                    continue
            else:
                meta["ExptDate"] = val
        elif "Task name" in key:
            meta["TaskName"] = val
        elif "Experiment name" in key:
            meta["ExperimentName"] = val
    return meta


# ---- Format A: SA: codes grouped by State ----------------------------------

def _parse_format_a(lines):
    """Parse SA-token format. Returns list of {label, choices} dicts."""
    sa_events_by_state = {}
    state_order = []
    current_state = None

    for line in lines:
        if not line.startswith("P "):
            continue
        m = re.match(r"P\s+\d+\s+(.+)", line)
        if not m:
            continue
        event = m.group(1).strip()

        if event.startswith("State:"):
            current_state = event[len("State:"):].strip()
            if current_state not in sa_events_by_state:
                sa_events_by_state[current_state] = []
                state_order.append(current_state)
        elif event.startswith("SA:") and current_state is not None:
            code = event[len("SA:"):].strip().upper()
            if code in CHOICE_CODES:
                sa_events_by_state[current_state].append(code)

    results = []
    for state in state_order:
        choices = sa_events_by_state.get(state, [])
        if choices:
            results.append({"label": f"State: {state}", "choices": choices})
    return results


# ---- Format B: Trial-numbered with Correct_choice / Incorrect_choice -------

def _parse_format_b(lines):
    """Each Correct_choice line = one C trial, each Incorrect_choice = one W.

    The number after the colon is the running cumulative count for that type
    (correct: 1,2,3,...; incorrect: 1,2,3,...). A genuine trial always
    increments it. The end-of-task summary block re-prints the final
    Correct_choice / Incorrect_choice totals at the same timestamp, so those
    lines repeat the previous value instead of incrementing it. We skip any
    line whose counter does not exceed the last one seen for its type, which
    drops the summary repeats without counting them as extra trials."""
    cells = []              # ordered list of "C" / "W" strings (one per trial)
    last_correct = 0
    last_incorrect = 0

    for line in lines:
        if not line.startswith("P "):
            continue
        m = re.match(r"P\s+\d+\s+(.+)", line)
        if not m:
            continue
        event = m.group(1).strip()

        if "Correct_choice" in event and "Incorrect_choice" not in event:
            n = _trailing_int(event)
            if n is None or n > last_correct:
                cells.append("C")
                if n is not None:
                    last_correct = n
        elif "Incorrect_choice" in event:
            n = _trailing_int(event)
            if n is None or n > last_incorrect:
                cells.append("W")
                if n is not None:
                    last_incorrect = n

    return cells


def _trailing_int(event):
    """Return the integer after the final ':' in an event string, or None."""
    m = re.search(r":\s*(\d+)\s*$", event)
    return int(m.group(1)) if m else None


# ---- Format C: New tab-separated log (time/type/subtype/content) -----------

def _split_tsv_rows(raw_lines):
    """Turn raw lines into (time, type, subtype, content) tuples.

    Only lines that look like the new tab-separated log are kept. The
    content column may itself contain tabs (e.g. embedded JSON), so we
    limit the split to 4 fields.
    """
    rows = []
    for ln in raw_lines:
        if "\t" not in ln:
            continue
        parts = ln.split("\t", 3)
        while len(parts) < 4:
            parts.append("")
        t, typ, sub, content = (p.strip() for p in parts[:4])
        rows.append((t, typ, sub, content))
    return rows


def _is_new_format(raw_lines):
    """Detect the new TSV log by its header row."""
    for ln in raw_lines[:10]:
        parts = [p.strip().lower() for p in ln.split("\t")]
        if len(parts) >= 4 and parts[0] == "time" and parts[1] == "type" \
                and parts[2] == "subtype" and parts[3] == "content":
            return True
    return False


def _parse_metadata_new(tsv_rows):
    meta = {
        "MouseID": "",
        "Group": "",
        "Subgroup": "",
        "ExptDate": "",
        "StartTime": "",
        "TaskName": "",
        "ExperimentName": "",
    }
    for _t, typ, sub, content in tsv_rows:
        if typ == "info":
            if sub == "subject_id":
                meta["MouseID"] = content
            elif sub == "task_name":
                meta["TaskName"] = content
            elif sub == "project":
                meta["ExperimentName"] = content
            elif sub == "start_time":
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                    try:
                        dt = datetime.strptime(content, fmt)
                        meta["ExptDate"] = dt.strftime("%Y-%m-%d")
                        meta["StartTime"] = dt.strftime("%H:%M:%S")
                        break
                    except ValueError:
                        continue
                else:
                    meta["ExptDate"] = content
        elif typ == "variable" and sub == "metadata":
            try:
                extra = json.loads(content)
            except (ValueError, TypeError):
                continue
            if isinstance(extra, dict):
                if not meta["MouseID"] and extra.get("subject_id"):
                    meta["MouseID"] = str(extra["subject_id"])
                # Optional grouping fields when present in the metadata blob.
                for key in ("group", "Group"):
                    if extra.get(key):
                        meta["Group"] = str(extra[key])
                        break
                for key in ("subgroup", "Subgroup", "Run"):
                    if extra.get(key):
                        meta["Subgroup"] = str(extra[key])
                        break
    return meta


def _parse_format_c(tsv_rows):
    """Each 'Correct <arm>' print line = one C trial, each 'Incorrect <arm>'
    print line = one W trial. Session summary lines (lowercase 'correct'/
    'incorrect') are ignored."""
    cells = []
    for _t, typ, _sub, content in tsv_rows:
        if typ != "print":
            continue
        if content.startswith("Correct ") or content == "Correct":
            cells.append("C")
        elif content.startswith("Incorrect ") or content == "Incorrect":
            cells.append("W")
    return cells


# ---- Top-level dispatcher ---------------------------------------------------

def _parse_new_format(raw_lines, filename=""):
    """Parse the new TSV log (Format C). Returns (rows, meta, diag)."""
    tsv_rows = _split_tsv_rows(raw_lines)
    meta = _parse_metadata_new(tsv_rows)
    cells = _parse_format_c(tsv_rows)

    rows = []
    if cells:
        correct = sum(1 for c in cells if c == "C")
        wrong = sum(1 for c in cells if c == "W")
        scored = correct + wrong
        pct = round(100 * correct / scored, 1) if scored else 0.0
        state_label = meta["TaskName"] or "Trials"
        row = {
            "SourceFile": filename,
            "MouseID": meta["MouseID"],
            "Group": meta["Group"],
            "ExptDate": meta["ExptDate"],
            "StartTime": meta["StartTime"],
            "subgroup": meta["Subgroup"] or meta["Group"],
            "State": state_label,
            "Correct": correct,
            "Incorrect": wrong,
            "percent correct": pct,
        }
        for i in range(NUM_CHOICE_COLS):
            row[str(i + 1)] = cells[i] if i < len(cells) else ""
        rows.append(row)

    diag = {
        "format": "C (new TSV log)" if rows else "(none detected)",
        "rows_produced": len(rows),
        "fmt_c_trials": len(cells),
    }
    return rows, meta, diag


def parse_file(content, filename=""):
    """Returns (rows, meta, diag).
    Rows have base metadata + State + Correct/Incorrect/percent + 1..14."""
    raw_lines = content.splitlines()

    # New tab-separated format (Format C) is handled separately.
    if _is_new_format(raw_lines):
        return _parse_new_format(raw_lines, filename)

    lines = [ln.strip() for ln in raw_lines]
    meta = _parse_metadata(lines)

    # Try format A first; if it produces nothing, try format B.
    fmt_a = _parse_format_a(lines)
    fmt_b_cells = _parse_format_b(lines)

    rows = []
    fmt_used = None

    if fmt_a:
        fmt_used = "A (SA: by state)"
        for entry in fmt_a:
            choices = entry["choices"]
            correct = sum(1 for c in choices if c == "C")
            wrong = sum(1 for c in choices if c == "W")
            scored = correct + wrong
            pct = round(100 * correct / scored, 1) if scored else 0.0
            row = {
                "SourceFile": filename,
                "MouseID": meta["MouseID"],
                "Group": meta["Group"],
                "ExptDate": meta["ExptDate"],
                "StartTime": meta["StartTime"],
                "subgroup": meta["Subgroup"] or meta["Group"],
                "State": entry["label"],
                "Correct": correct,
                "Incorrect": wrong,
                "percent correct": pct,
            }
            for i in range(NUM_CHOICE_COLS):
                row[str(i + 1)] = choices[i] if i < len(choices) else ""
            rows.append(row)

    elif fmt_b_cells:
        fmt_used = "B (trial / Correct_choice)"
        # one row per file
        correct = sum(1 for c in fmt_b_cells if c.startswith("C"))
        wrong = sum(1 for c in fmt_b_cells if c.startswith("W"))
        scored = correct + wrong
        pct = round(100 * correct / scored, 1) if scored else 0.0
        state_label = meta["TaskName"] or "Trials"
        row = {
            "SourceFile": filename,
            "MouseID": meta["MouseID"],
            "Group": meta["Group"],
            "ExptDate": meta["ExptDate"],
            "StartTime": meta["StartTime"],
            "subgroup": meta["Subgroup"] or meta["Group"],
            "State": state_label,
            "Correct": correct,
            "Incorrect": wrong,
            "percent correct": pct,
        }
        for i in range(NUM_CHOICE_COLS):
            row[str(i + 1)] = fmt_b_cells[i] if i < len(fmt_b_cells) else ""
        rows.append(row)

    diag = {
        "format": fmt_used or "(none detected)",
        "rows_produced": len(rows),
        "fmt_a_states_with_data": len(fmt_a),
        "fmt_b_trials": len(fmt_b_cells),
    }
    return rows, meta, diag


BASE_COLS = [
    "SourceFile", "MouseID", "Group", "ExptDate", "StartTime", "subgroup",
    "State", "Correct", "Incorrect", "percent correct",
]
CHOICE_COL_NAMES = [str(i) for i in range(1, NUM_CHOICE_COLS + 1)]
ALL_COLS = BASE_COLS + CHOICE_COL_NAMES


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class DropTable(QTableWidget):
    """QTableWidget that accepts file drops and forwards them to the parent."""

    def __init__(self, on_files_dropped, parent=None):
        super().__init__(parent)
        self._on_files_dropped = on_files_dropped
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            super().dropEvent(event)
            return
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        expanded = []
        supported = {".txt", ".tsv"}
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                for ext in supported:
                    expanded.extend(str(x) for x in pp.rglob(f"*{ext}"))
            elif pp.suffix.lower() in supported:
                expanded.append(str(pp))
        if expanded:
            self._on_files_dropped(expanded)
            event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Behavior Experiment Analyzer")
        self.resize(1400, 800)

        self._rows: list[dict] = []
        self._loaded_files: list[str] = []

        # toolbar
        self.btn_add = QPushButton("Add files…")
        self.btn_add.clicked.connect(self.pick_files)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_export_tsv = QPushButton("Export TSV…")
        self.btn_export_tsv.clicked.connect(lambda: self.export(kind="tsv"))
        self.btn_export_xlsx = QPushButton("Export Excel…")
        self.btn_export_xlsx.clicked.connect(lambda: self.export(kind="xlsx"))

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_add)
        top_bar.addWidget(self.btn_clear)
        top_bar.addStretch(1)
        top_bar.addWidget(self.btn_export_tsv)
        top_bar.addWidget(self.btn_export_xlsx)

        # summary table
        self.summary_table = DropTable(self.add_files)
        self.summary_table.setColumnCount(len(ALL_COLS))
        self.summary_table.setHorizontalHeaderLabels(ALL_COLS)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        summary_label = QLabel("Per-state summary  (drop .txt or .tsv files anywhere on this table)")
        summary_label.setStyleSheet("font-weight: bold; padding: 4px;")
        summary_box = QWidget()
        sl = QVBoxLayout(summary_box)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(summary_label)
        sl.addWidget(self.summary_table)

        # per-mouse aggregation table
        self.agg_table = QTableWidget()
        self.agg_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.agg_table.setSortingEnabled(True)
        self.agg_table.setAlternatingRowColors(True)
        self.agg_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        agg_label = QLabel("Per-mouse totals")
        agg_label.setStyleSheet("font-weight: bold; padding: 4px;")
        agg_box = QWidget()
        al = QVBoxLayout(agg_box)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(agg_label)
        al.addWidget(self.agg_table)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(summary_box)
        splitter.addWidget(agg_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top_bar)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self._set_status("Drop .txt files onto the table, or click 'Add files…'.")

        open_act = QAction("Open", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.pick_files)
        self.addAction(open_act)

    # status helper
    def _set_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    # file ingest
    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select experiment files", "",
            "Experiment logs (*.txt *.tsv);;Text files (*.txt);;"
            "Tab-separated (*.tsv);;All files (*)"
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths: list[str]):
        added = 0
        skipped = []
        for p in paths:
            if p in self._loaded_files:
                continue
            try:
                content = Path(p).read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                skipped.append(f"{Path(p).name}: {exc}")
                continue
            try:
                rows, _meta, _diag = parse_file(content, Path(p).name)
            except Exception as exc:
                skipped.append(f"{Path(p).name}: parse error: {exc}")
                continue
            if not rows:
                skipped.append(f"{Path(p).name}: no trials detected")
                continue
            self._rows.extend(rows)
            self._loaded_files.append(p)
            added += len(rows)

        if added:
            self._refresh_tables()

        msg = f"{len(self._loaded_files)} file(s) loaded · {len(self._rows)} rows"
        if skipped:
            msg += f" · {len(skipped)} skipped"
            QMessageBox.warning(self, "Some files were skipped", "\n".join(skipped))
        self._set_status(msg)

    def clear_all(self):
        self._rows.clear()
        self._loaded_files.clear()
        self._refresh_tables()
        self._set_status("Cleared.")

    def _build_df(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=ALL_COLS)
        df = pd.DataFrame(self._rows)
        for col in ALL_COLS:
            if col not in df.columns:
                df[col] = "" if col not in ("Correct", "Incorrect", "percent correct") else 0
        df = df[ALL_COLS]
        for col in ("Correct", "Incorrect"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df["percent correct"] = pd.to_numeric(df["percent correct"], errors="coerce").fillna(0.0)
        return df

    # rendering
    def _refresh_tables(self):
        df = self._build_df()
        self._fill_table(self.summary_table, df, ALL_COLS)

        if df.empty:
            self._fill_table(self.agg_table, pd.DataFrame(), [])
            return

        agg = (
            df.groupby(["MouseID", "Group", "subgroup"], dropna=False)
              .agg(Correct=("Correct", "sum"),
                   Incorrect=("Incorrect", "sum"),
                   States=("State", "count"))
              .reset_index()
        )
        denom = (agg["Correct"] + agg["Incorrect"]).replace(0, pd.NA)
        agg["percent correct"] = (100 * agg["Correct"] / denom).round(1)
        agg_cols = ["MouseID", "Group", "subgroup",
                    "Correct", "Incorrect", "percent correct", "States"]
        self._fill_table(self.agg_table, agg, agg_cols)

    @staticmethod
    def _fill_table(table: QTableWidget, df: pd.DataFrame, cols: list[str]):
        table.setSortingEnabled(False)
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)

        for r, (_, row) in enumerate(df.iterrows()):
            for c, name in enumerate(cols):
                val = row.get(name, "")
                if pd.isna(val):
                    val = ""
                item = QTableWidgetItem(str(val))
                if isinstance(val, (int, float)):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(r, c, item)

        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    # export
    def export(self, kind: str):
        df = self._build_df()
        if df.empty:
            QMessageBox.information(self, "Nothing to export", "Load some files first.")
            return

        if kind == "tsv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save summary as TSV", "experiment_summary.tsv",
                "Tab-separated (*.tsv);;All files (*)"
            )
            if not path:
                return
            try:
                df.to_csv(path, sep="\t", index=False)
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))
                return
            self._set_status(f"Saved {path}")
            return

        if kind == "xlsx":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save summary as Excel", "experiment_summary.xlsx",
                "Excel workbook (*.xlsx);;All files (*)"
            )
            if not path:
                return
            try:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Summary")
                    agg = (
                        df.groupby(
                            ["MouseID", "Group", "subgroup"], dropna=False
                        ).agg(Correct=("Correct", "sum"),
                              Incorrect=("Incorrect", "sum"),
                              States=("State", "count")).reset_index()
                    )
                    denom = (agg["Correct"] + agg["Incorrect"]).replace(0, pd.NA)
                    agg["percent correct"] = (100 * agg["Correct"] / denom).round(1)
                    agg.to_excel(writer, index=False, sheet_name="PerMouse")
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))
                return
            self._set_status(f"Saved {path}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

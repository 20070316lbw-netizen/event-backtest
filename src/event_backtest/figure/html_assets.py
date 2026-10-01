"""HTML 报告的内联样式与脚本(自包含: 不引用任何 CDN / 外部文件)。

样式: CSS 变量 + 浅色默认, 跟随系统深色, 右上角可手动切换。
脚本: 纯原生 JS + SVG, 没有第三方依赖; 图表支持十字光标、tooltip、图例开关、
      区间缩放(1月/3月/6月/1年/5年/全部), 表格支持点表头排序。
数据由 figure.html 以 JSON 塞进 <script type="application/json">。
"""
from __future__ import annotations

__all__ = ["CSS", "JS"]

CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f5f6f8; --panel: #ffffff; --fg: #1b2430; --muted: #7c8698;
  --border: #e5e8ef; --grid: #eef0f4; --accent: #4f7cff; --pos: #1e9e6a; --neg: #d64545;
  --hover: #f2f5ff;
  --shadow: 0 1px 2px rgba(16, 24, 40, .04), 0 8px 24px rgba(16, 24, 40, .05);
}
html[data-theme="dark"] {
  --bg: #11151c; --panel: #171d26; --fg: #e8edf5; --muted: #8b97a8;
  --border: #232c39; --grid: #212a37; --accent: #6b93ff; --pos: #3ecf8e; --neg: #ff6b6b;
  --hover: #1d2531; --shadow: none;
}
@media (prefers-color-scheme: dark) {
  html[data-theme="auto"] {
    --bg: #11151c; --panel: #171d26; --fg: #e8edf5; --muted: #8b97a8;
    --border: #232c39; --grid: #212a37; --accent: #6b93ff; --pos: #3ecf8e; --neg: #ff6b6b;
    --hover: #1d2531; --shadow: none;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
        "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 72px; }
.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 20px; }
h1 { font-size: 22px; line-height: 1.3; margin: 0 0 6px; font-weight: 650; }
h2 { font-size: 15px; margin: 0; font-weight: 600; }
h3 { font-size: 13px; margin: 0 0 6px; font-weight: 600; color: var(--muted); }
.sub { color: var(--muted); font-size: 13px; }
button {
  font: inherit; color: var(--fg); background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 5px 10px; cursor: pointer; transition: border-color .15s, background .15s;
}
button:hover { border-color: var(--muted); }
button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }
.kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; box-shadow: var(--shadow); }
.kpi .label { color: var(--muted); font-size: 12px; }
.kpi .value { font-size: 20px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
.pos { color: var(--pos); }
.neg { color: var(--neg); }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin-bottom: 18px; box-shadow: var(--shadow); }
.card-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 10px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grid2 > .card { margin-bottom: 0; }
@media (max-width: 880px) { .grid2 { grid-template-columns: 1fr; } }
.ranges { display: flex; flex-wrap: wrap; gap: 6px; }
.chart-host { position: relative; width: 100%; }
.chart-svg svg { display: block; }
.axis text { fill: var(--muted); font-size: 11px; }
.grid-line { stroke: var(--grid); stroke-width: 1; }
.zero-line { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: .55; }
.crosshair { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: .75; }
.empty-text { fill: var(--muted); font-size: 12px; }
.tip {
  position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; box-shadow: var(--shadow); font-size: 12px;
  min-width: 132px; display: none; z-index: 5; white-space: nowrap;
}
.tip .head { color: var(--muted); margin-bottom: 4px; }
.tip .row { display: flex; justify-content: space-between; gap: 14px; }
.tip .name { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; font-size: 12px; color: var(--muted); }
.legend .item { display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.legend .dot { width: 9px; height: 9px; border-radius: 50%; }
.legend .off { opacity: .35; text-decoration: line-through; }
.tables { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 12px; position: sticky; top: 0; background: var(--panel); }
td.num, th.num { text-align: right; }
tbody tr:hover { background: var(--hover); }
.table-wrap { max-height: 440px; overflow: auto; border: 1px solid var(--border); border-radius: 10px; }
.sortable th { cursor: pointer; }
.sortable th:hover { color: var(--fg); }
.note { color: var(--muted); font-size: 12px; margin-top: 8px; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; border: 1px solid var(--border); color: var(--muted); }
"""

JS = r"""
"use strict";

const DATA = JSON.parse(document.getElementById("report-data").textContent);
const NS = "http://www.w3.org/2000/svg";
const COLORS = ["#4f7cff", "#ef8b3c", "#2fb37a", "#c05299", "#d4b13c",
                "#7a5cff", "#3aa7c4", "#e2564f", "#8a8f98", "#57a773"];

/* ---------------------------------------------------------------- 小工具 */
function el(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function svgEl(name, attrs) {
  const node = document.createElementNS(NS, name);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function isNum(value) { return typeof value === "number" && isFinite(value); }

function fmtValue(value, kind) {
  if (value === null || value === undefined) return "-";
  if (kind === "text") return String(value);
  let num = Number(value);
  if (!isFinite(num)) return "-";
  // 浮点噪声(如 3.5e-18)在百分比口径下要当成 0, 否则会出现 "-0.00%" / "5e-18%"
  if (num !== 0 && Math.abs(num) < 1e-12) num = 0;
  switch (kind) {
    case "pct2": return (num * 100).toFixed(2) + "%";
    case "pct1": return (num * 100).toFixed(1) + "%";
    case "pct0": return (num * 100).toFixed(0) + "%";
    case "int": return (Math.round(num) + 0).toLocaleString("zh-CN");
    case "money": return num.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    case "num2": return num.toLocaleString("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    case "num3": return num.toFixed(3);
    default: return String(value);
  }
}

function toneClass(value, tone) {
  if (!tone || tone === "none" || !isNum(value)) return "";
  if (tone === "pos") return "pos";
  if (tone === "neg") return "neg";
  if (tone === "auto") return value > 0 ? "pos" : (value < 0 ? "neg" : "");
  return "";
}

function niceTicks(min, max, count) {
  if (!isFinite(min) || !isFinite(max)) return [0];
  if (min === max) { min -= Math.abs(min || 1) * 0.05; max += Math.abs(max || 1) * 0.05; }
  const raw = (max - min) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  let step = norm <= 1 ? 1 : (norm <= 2 ? 2 : (norm <= 5 ? 5 : 10));
  step *= mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(v);
  return ticks.length ? ticks : [min, max];
}

function parseTimes(labels) {
  const parsed = labels.map(function (label) { return Date.parse(label); });
  for (let i = 0; i < parsed.length; i++) if (!isFinite(parsed[i])) return null;
  return parsed;
}

/* ---------------------------------------------------------------- 图表 */
function renderChart(host, spec) {
  host.classList.add("chart-host");
  const svgBox = el("div", "chart-svg");
  const tip = el("div", "tip");
  const legend = el("div", "legend");
  host.appendChild(svgBox);
  host.appendChild(tip);
  host.appendChild(legend);

  const state = { hidden: {}, from: null };
  let lastWidth = 0;

  function shown() {
    return spec.series.filter(function (s) { return !state.hidden[s.name]; });
  }

  function setRange(months) {
    const times = parseTimes(spec.labels);
    if (!times || months === null) { state.from = null; draw(); return; }
    const last = new Date(times[times.length - 1]);
    const start = new Date(last.getTime());
    start.setMonth(start.getMonth() - months);
    state.from = start.getTime();
    draw();
  }

  function drawLegend() {
    clear(legend);
    if (spec.series.length < 2) return;
    spec.series.forEach(function (s, index) {
      const item = el("div", "item" + (state.hidden[s.name] ? " off" : ""));
      const dot = el("span", "dot");
      dot.style.background = s.color || COLORS[index % COLORS.length];
      item.appendChild(dot);
      item.appendChild(el("span", null, s.name));
      item.onclick = function () {
        state.hidden[s.name] = !state.hidden[s.name];
        drawLegend();
        draw();
      };
      legend.appendChild(item);
    });
  }

  function hideTip() {
    tip.style.display = "none";
    const cross = svgBox.querySelector(".crosshair");
    if (cross) cross.remove();
    const dots = svgBox.querySelectorAll(".hover-dot");
    for (let i = 0; i < dots.length; i++) dots[i].remove();
  }

  function draw() {
    const width = Math.max(320, host.clientWidth || 680);
    lastWidth = width;
    const height = spec.height || 300;
    clear(svgBox);
    hideTip();

    const series = shown();
    const labels = spec.labels;
    const n = labels.length;

    function makeSvg() {
      const node = svgEl("svg", { width: width, height: height, viewBox: "0 0 " + width + " " + height });
      svgBox.appendChild(node);
      return node;
    }

    function note(text) {
      const margin = { top: 14, left: 64 };
      const node = makeSvg();
      const label = svgEl("text", { x: margin.left, y: margin.top + (height - 40) / 2, class: "empty-text" });
      label.textContent = text;
      node.appendChild(label);
    }

    if (n < 2 || series.length === 0) {
      note(n < 2 ? "数据不足, 画不了图" : "所有序列都被隐藏了");
      return;
    }

    const times = parseTimes(labels);
    let i0 = 0;
    let i1 = n - 1;
    if (times && state.from !== null) {
      while (i0 < i1 && times[i0] < state.from) i0++;
    }

    let min = Infinity;
    let max = -Infinity;
    series.forEach(function (s) {
      for (let i = i0; i <= i1; i++) {
        const v = s.values[i];
        if (!isNum(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    });
    if (!isFinite(min) || !isFinite(max)) {
      note("这段区间没有数据");
      return;
    }
    if (spec.zeroBase) { min = Math.min(min, 0); max = Math.max(max, 0); }
    if (spec.forceZeroMax) max = 0;
    const pad = (max - min) * 0.08 || Math.abs(max) * 0.08 || 1;
    min -= pad;
    if (!spec.forceZeroMax) max += pad;

    /* 刻度先算出来: 一是去重(0.5 / 1 都显示成 1 就只留一个), 二是按最宽的刻度文本
       定左边距 —— 净值到百万级时 64px 会把 "1,100,000.00" 截掉一半。 */
    const axisKind = spec.axisKind || spec.valueKind;
    const ticks = [];
    const seenTicks = {};
    niceTicks(min, max, 5).forEach(function (tick) {
      if (tick < min - 1e-9 || tick > max + 1e-9) return;
      // 计数轴的刻度取整(顺便把 -0 变成 0)
      const shown = axisKind === "int" ? Math.round(tick) + 0 : tick;
      const text = fmtValue(shown, axisKind);
      if (seenTicks[text]) return;
      seenTicks[text] = true;
      ticks.push({ tick: tick, text: text });
    });
    const widest = ticks.reduce(function (acc, item) { return Math.max(acc, item.text.length); }, 4);
    const margin = {
      top: 14, right: 16, bottom: 26,
      left: Math.min(132, Math.max(48, Math.round(widest * 6.9) + 14))
    };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const svg = makeSvg();

    function xOf(i) {
      if (times) {
        const span = (times[i1] - times[i0]) || 1;
        return margin.left + (times[i] - times[i0]) / span * innerW;
      }
      return margin.left + (i - i0) / Math.max(1, i1 - i0) * innerW;
    }
    function yOf(v) {
      return margin.top + (max - v) / (max - min) * innerH;
    }

    /* 网格 + y 轴 */
    const grid = svgEl("g", { class: "grid" });
    const axis = svgEl("g", { class: "axis" });
    ticks.forEach(function (item) {
      const y = yOf(item.tick);
      grid.appendChild(svgEl("line", { x1: margin.left, y1: y, x2: margin.left + innerW, y2: y, class: "grid-line" }));
      const label = svgEl("text", { x: margin.left - 8, y: y + 4, "text-anchor": "end" });
      label.textContent = item.text;
      axis.appendChild(label);
    });
    svg.appendChild(grid);
    svg.appendChild(axis);

    if (spec.type === "bar") {
      drawBars();
    } else {
      drawLines();
    }

    function drawLines() {
      if (min < 0 && max > 0) {
        const zero = yOf(0);
        svg.appendChild(svgEl("line", { x1: margin.left, y1: zero, x2: margin.left + innerW, y2: zero, class: "zero-line" }));
      }
      series.forEach(function (s) {
        const color = s.color || COLORS[spec.series.indexOf(s) % COLORS.length];
        let d = "";
        let pen = false;
        const points = [];
        for (let i = i0; i <= i1; i++) {
          const v = s.values[i];
          if (!isNum(v)) { pen = false; continue; }
          const x = xOf(i);
          const y = yOf(v);
          d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1) + " ";
          pen = true;
          points.push([i, x, y]);
        }
        if (spec.type === "area" && points.length) {
          const base = yOf(Math.min(Math.max(0, min), max));
          const area = "M" + points[0][1].toFixed(1) + " " + base.toFixed(1) + " " +
            points.map(function (p) { return "L" + p[1].toFixed(1) + " " + p[2].toFixed(1); }).join(" ") +
            " L" + points[points.length - 1][1].toFixed(1) + " " + base.toFixed(1) + " Z";
          const fill = svgEl("path", { d: area, fill: color, "fill-opacity": 0.16, stroke: "none" });
          svg.appendChild(fill);
        }
        svg.appendChild(svgEl("path", {
          d: d, fill: "none", stroke: color, "stroke-width": 1.6,
          "stroke-linejoin": "round", "stroke-linecap": "round"
        }));
      });
      drawXAxis();
      attachHover(function (index) { return xOf(index); });
    }

    function drawBars() {
      const values = series[0].values;
      const count = i1 - i0 + 1;
      const slot = innerW / Math.max(1, count);
      const barW = Math.max(1, Math.min(28, slot * 0.68));
      const zero = yOf(Math.min(Math.max(0, min), max));
      if (min < 0 && max > 0) {
        svg.appendChild(svgEl("line", { x1: margin.left, y1: zero, x2: margin.left + innerW, y2: zero, class: "zero-line" }));
      }
      for (let i = i0; i <= i1; i++) {
        const v = values[i];
        if (!isNum(v)) continue;
        const x = xOf(i) - barW / 2;
        const y = Math.min(yOf(v), zero);
        const h = Math.max(1, Math.abs(yOf(v) - zero));
        const color = spec.colorBySign ? (v >= 0 ? COLORS[2] : COLORS[7]) : COLORS[0];
        svg.appendChild(svgEl("rect", {
          x: x.toFixed(1), y: y.toFixed(1), width: barW.toFixed(1), height: h.toFixed(1),
          rx: Math.min(2, barW / 3), fill: color, "fill-opacity": 0.85
        }));
      }
      drawXAxis();
      attachHover(function (index) { return xOf(index); }, true);
    }

    function drawXAxis() {
      const ticks = 5;
      const seen = {};
      for (let k = 0; k <= ticks; k++) {
        const index = Math.round(i0 + (i1 - i0) * (k / ticks));
        const text = labels[index];
        if (seen[text]) continue;
        seen[text] = true;
        const x = xOf(index);
        // 首尾两个标签贴边会溢出, 改成向内对齐
        let anchor = "middle";
        if (x < margin.left + 28) anchor = "start";
        else if (x > width - margin.right - 28) anchor = "end";
        const label = svgEl("text", { x: x, y: margin.top + innerH + 18, "text-anchor": anchor });
        label.textContent = text;
        axis.appendChild(label);
      }
    }

    function attachHover(xAt, asBar) {
      const overlay = svgEl("rect", {
        x: margin.left, y: margin.top, width: Math.max(1, innerW), height: Math.max(1, innerH),
        fill: "transparent"
      });
      function show(px) {
        let best = i0;
        let bestDist = Infinity;
        for (let i = i0; i <= i1; i++) {
          const d = Math.abs(xAt(i) - px);
          if (d < bestDist) { bestDist = d; best = i; }
        }
        hideTip();
        const x = xAt(best);
        svg.appendChild(svgEl("line", { x1: x, y1: margin.top, x2: x, y2: margin.top + innerH, class: "crosshair" }));
        const rows = [];
        shown().forEach(function (s) {
          const v = s.values[best];
          if (!isNum(v)) return;
          const color = s.color || COLORS[spec.series.indexOf(s) % COLORS.length];
          svg.appendChild(svgEl("circle", { cx: x, cy: yOf(v), r: 3.2, fill: color, class: "hover-dot" }));
          rows.push([s.name, fmtValue(v, spec.valueKind), color]);
        });
        clear(tip);
        const head = el("div", "head", labels[best]);
        tip.appendChild(head);
        rows.forEach(function (row) {
          const line = el("div", "row");
          const name = el("span", "name", row[0]);
          name.style.color = row[2];
          line.appendChild(name);
          line.appendChild(el("span", null, row[1]));
          tip.appendChild(line);
        });
        tip.style.display = "block";
        const hostWidth = host.clientWidth || width;
        const left = Math.max(6, Math.min(hostWidth - 140, x));
        tip.style.left = left + "px";
        tip.style.top = (asBar ? 6 : 6) + "px";
      }
      overlay.addEventListener("mousemove", function (ev) {
        const box = svg.getBoundingClientRect();
        show(ev.clientX - box.left);
      });
      overlay.addEventListener("mouseleave", hideTip);
      overlay.addEventListener("touchmove", function (ev) {
        if (!ev.touches.length) return;
        const box = svg.getBoundingClientRect();
        show(ev.touches[0].clientX - box.left);
        ev.preventDefault();
      }, { passive: false });
      svg.appendChild(overlay);
    }
  }

  drawLegend();
  draw();
  if (typeof ResizeObserver !== "undefined") {
    const observer = new ResizeObserver(function () {
      const width = host.clientWidth;
      if (Math.abs(width - lastWidth) > 2) draw();
    });
    observer.observe(host);
  }
  return { setRange: setRange, draw: draw };
}

/* ---------------------------------------------------------------- 表格 */
function buildTable(spec) {
  const wrap = el("div", "table-wrap");
  const table = el("table", spec.sortable ? "sortable" : "");
  const thead = el("thead");
  const headRow = el("tr");
  spec.columns.forEach(function (column) {
    const th = el("th", column.kind === "text" ? "" : "num", column.label);
    if (spec.sortable) {
      th.onclick = function () {
        if (sortKey === column.key) { sortAsc = !sortAsc; } else { sortKey = column.key; sortAsc = false; }
        renderBody();
      };
    }
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = el("tbody");
  table.appendChild(tbody);
  wrap.appendChild(table);

  let sortKey = spec.sortKey || null;
  let sortAsc = spec.sortAsc === true;

  function renderBody() {
    clear(tbody);
    let rows = spec.rows.slice();
    if (sortKey) {
      rows.sort(function (a, b) {
        const av = a[sortKey];
        const bv = b[sortKey];
        if (isNum(av) && isNum(bv)) return sortAsc ? av - bv : bv - av;
        return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
      });
    }
    rows.forEach(function (row) {
      const tr = el("tr");
      spec.columns.forEach(function (column) {
        tr.appendChild(el("td", column.kind === "text" ? "" : "num",
          fmtValue(row[column.key], column.kind)));
      });
      tbody.appendChild(tr);
    });
  }
  renderBody();
  return wrap;
}

function pairsTable(rows) {
  const wrap = el("div", "table-wrap");
  const table = el("table");
  const tbody = el("tbody");
  rows.forEach(function (row) {
    const tr = el("tr");
    tr.appendChild(el("td", null, row[0]));
    const value = el("td", "num", row[1]);
    tr.appendChild(value);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

/* ---------------------------------------------------------------- 页面 */
function card(title, body, actions) {
  const section = el("section", "card");
  const head = el("div", "card-head");
  head.appendChild(el("h2", null, title));
  if (actions) head.appendChild(actions);
  section.appendChild(head);
  section.appendChild(body);
  return section;
}

function chartCard(title, spec, withRanges) {
  const host = el("div");
  const actions = el("div", "ranges");
  const section = card(title, host, withRanges ? actions : null);
  const chart = renderChart(host, spec);
  if (withRanges && spec.ranges) {
    const options = [{ label: "全部", months: null }];
    spec.ranges.forEach(function (months) {
      const names = { 1: "1月", 3: "3月", 6: "6月", 12: "1年", 36: "3年", 60: "5年", 120: "10年" };
      options.push({ label: names[months] || (months + "月"), months: months });
    });
    options.forEach(function (option, index) {
      const button = el("button", index === 0 ? "active" : "", option.label);
      button.onclick = function () {
        Array.prototype.forEach.call(actions.children, function (child) { child.className = ""; });
        button.className = "active";
        chart.setRange(option.months);
      };
      actions.appendChild(button);
    });
  }
  return section;
}

function kpiGrid(kpis) {
  const grid = el("div", "kpis");
  kpis.forEach(function (kpi) {
    const box = el("div", "kpi");
    box.appendChild(el("div", "label", kpi.label));
    const value = el("div", "value " + toneClass(kpi.value, kpi.tone), fmtValue(kpi.value, kpi.kind));
    box.appendChild(value);
    grid.appendChild(box);
  });
  return grid;
}

function sectionsCard(title, sections) {
  const grid = el("div", "tables");
  sections.forEach(function (section) {
    const box = el("div");
    box.appendChild(el("h3", null, section.title));
    const wrap = el("div", "table-wrap");
    const table = el("table");
    const tbody = el("tbody");
    section.rows.forEach(function (row) {
      const tr = el("tr");
      tr.appendChild(el("td", null, row[0]));
      tr.appendChild(el("td", "num", row[1]));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    box.appendChild(wrap);
    grid.appendChild(box);
  });
  return card(title || "指标", grid);
}

function toggleTheme() {
  const root = document.documentElement;
  const current = root.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  try { localStorage.setItem("eb-theme", next); } catch (err) { /* file:// 下可能不可用 */ }
}

function build() {
  const wrap = el("div", "wrap");
  const head = el("header", "page-head");
  const left = el("div");
  left.appendChild(el("h1", null, DATA.title));
  left.appendChild(el("div", "sub", DATA.subtitle + " · 生成于 " + DATA.generated));
  head.appendChild(left);
  const right = el("div");
  const themeButton = el("button", null, "深色 / 浅色");
  themeButton.onclick = toggleTheme;
  right.appendChild(themeButton);
  head.appendChild(right);
  wrap.appendChild(head);

  try {
    const saved = localStorage.getItem("eb-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch (err) { /* 忽略 */ }

  if (DATA.kpis && DATA.kpis.length) wrap.appendChild(kpiGrid(DATA.kpis));
  if (DATA.nav) wrap.appendChild(chartCard(DATA.nav.title || "净值", DATA.nav, true));
  if (DATA.drawdown) wrap.appendChild(chartCard(DATA.drawdown.title || "回撤", DATA.drawdown, true));
  if (DATA.monthly) wrap.appendChild(chartCard(DATA.monthly.title || "月度收益", DATA.monthly, false));
  if (DATA.trade_hist) wrap.appendChild(chartCard(DATA.trade_hist.title || "回合收益分布", DATA.trade_hist, false));
  if (DATA.fold_bars) wrap.appendChild(chartCard(DATA.fold_bars.title || "各折测试段收益", DATA.fold_bars, false));
  if (DATA.sections && DATA.sections.length) wrap.appendChild(sectionsCard("指标", DATA.sections));
  if (DATA.table) {
    const body = el("div");
    body.appendChild(buildTable(DATA.table));
    if (DATA.table.note) body.appendChild(el("div", "note", DATA.table.note));
    wrap.appendChild(card(DATA.table.title || "对比", body));
  }
  if (DATA.folds) {
    const body = el("div");
    body.appendChild(buildTable(DATA.folds));
    wrap.appendChild(card(DATA.folds.title || "各折明细", body));
  }
  if (DATA.trades) {
    const body = el("div");
    body.appendChild(buildTable(DATA.trades));
    if (DATA.trades.note) body.appendChild(el("div", "note", DATA.trades.note));
    wrap.appendChild(card(DATA.trades.title || "回合交易", body));
  }
  if (DATA.orders) {
    const body = el("div");
    body.appendChild(buildTable(DATA.orders));
    if (DATA.orders.note) body.appendChild(el("div", "note", DATA.orders.note));
    wrap.appendChild(card(DATA.orders.title || "委托明细", body));
  }
  if (DATA.reject_reasons && DATA.reject_reasons.length) {
    wrap.appendChild(card("拒单原因", pairsTable(DATA.reject_reasons)));
  }
  if (DATA.footer) wrap.appendChild(el("div", "note", DATA.footer));

  document.getElementById("app").appendChild(wrap);
}

build();
"""

/* Interactive figures for the exercise pages, drawn with ECharts (loaded site-wide).
 *
 * Markdown declares a figure with:
 *     <div class="chart" data-chart="fig1"></div>
 * and each figure's own file registers a builder:
 *     ANNCharts.register("fig1", function (data, helpers, el) { return option; });
 *
 * The numbers come from charts.json, written by the exercise's own scripts, so a
 * figure can never drift from the code that produced it.
 */
(function () {
  "use strict";

  var builders = {};
  var fetched = {};
  var live = [];

  // ------------------------------------------------------------- helpers
  // Passed to every builder so all figures share one visual language.

  function dark() {
    return document.body.dataset.mdColorScheme === "slate";
  }

  var H = {
    dark: dark,

    /** Primary text colour for the current palette. */
    ink: function () {
      return dark() ? "#d2d5da" : "#41464c";
    },

    /** Grid and axis lines: present, never loud. */
    faint: function () {
      return dark() ? "rgba(255,255,255,0.10)" : "rgba(0,0,0,0.09)";
    },

    /** Page background, for symbol borders that need to punch out of the plot. */
    paper: function () {
      return dark() ? "#1b1d1f" : "#ffffff";
    },

    /** Axis defaults: every axis ends up named, gridded and legible. */
    axis: function (name, extra) {
      return Object.assign(
        {
          name: name,
          nameLocation: "middle",
          nameGap: 30,
          nameTextStyle: { color: H.ink(), fontSize: 12 },
          axisLabel: { color: H.ink(), fontSize: 11, hideOverlap: true },
          axisLine: { lineStyle: { color: H.faint() } },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: H.faint() } },
        },
        extra || {}
      );
    },

    /** The figure's own title block. */
    heading: function (text, sub) {
      return {
        text: text,
        subtext: sub || "",
        left: "center",
        top: 2,
        textStyle: { color: H.ink(), fontSize: 13, fontWeight: 600 },
        subtextStyle: { color: H.ink(), fontSize: 11, opacity: 0.72 },
      };
    },

    /** A title over one panel of a multi-panel figure. */
    panelTitle: function (text, left, top) {
      return {
        text: text,
        left: left,
        top: top,
        textAlign: "center",
        textStyle: { color: H.ink(), fontSize: 12, fontWeight: 600 },
      };
    },

    legend: function (extra) {
      return Object.assign(
        { textStyle: { color: H.ink(), fontSize: 11 }, itemHeight: 9, itemWidth: 14, icon: "circle" },
        extra || {}
      );
    },

    /** A scatter series with the house point styling. */
    dot: function (name, color, data, extra) {
      return Object.assign(
        {
          name: name,
          type: "scatter",
          data: data,
          symbolSize: 7,
          itemStyle: { color: color, opacity: 0.72, borderColor: H.paper(), borderWidth: 0.4 },
          emphasis: { focus: "series", itemStyle: { opacity: 1 } },
          blendMode: "source-over",
        },
        extra || {}
      );
    },

    /** Shared bin edges, so two overlaid histograms stay comparable. */
    bins: function (values, n) {
      var lo = Math.min.apply(null, values);
      var hi = Math.max.apply(null, values);
      return { lo: lo, hi: hi, w: (hi - lo) / n, n: n };
    },

    /** Bin `values` onto `b`, returning [centre, count] pairs. */
    counts: function (values, b) {
      var out = new Array(b.n).fill(0);
      values.forEach(function (v) {
        out[Math.max(0, Math.min(b.n - 1, Math.floor((v - b.lo) / b.w)))] += 1;
      });
      return out.map(function (c, i) {
        return [b.lo + b.w * (i + 0.5), c];
      });
    },

    /** An overlaid histogram bar series. */
    bar: function (name, color, data, extra) {
      return Object.assign(
        {
          name: name,
          type: "bar",
          data: data,
          barWidth: "96%",
          barGap: "-100%",
          itemStyle: { color: color, opacity: 0.62 },
          emphasis: { focus: "series", itemStyle: { opacity: 0.95 } },
        },
        extra || {}
      );
    },

    pct: function (v, nd) {
      return (v * 100).toFixed(nd === undefined ? 1 : nd) + "%";
    },

    /** Attach a control strip under a figure. Returns the element; the builder
     *  must return it as `controls` so it is removed on re-render. */
    controls: function (el, html) {
      var ui = document.createElement("div");
      ui.className = "chart-controls";
      ui.innerHTML = html;
      el.parentNode.insertBefore(ui, el.nextSibling);
      return ui;
    },
  };

  // ------------------------------------------------------------- plumbing

  function teardown() {
    live.forEach(function (c) {
      if (c.controls && c.controls.parentNode) c.controls.parentNode.removeChild(c.controls);
      window.removeEventListener("resize", c.resize);
      c.chart.dispose();
    });
    live = [];
  }

  function draw(el, data) {
    var build = builders[el.dataset.chart];
    if (!build) return;
    el.innerHTML = ""; // drop the static PNG fallback; the canvas replaces it
    var chart = echarts.init(el, null, { renderer: "canvas" });
    var built = build(data, H, el) || {};
    var option = built.option || built;
    chart.setOption(
      Object.assign({ animationDuration: 420, textStyle: { fontFamily: "inherit" } }, option)
    );
    if (built.ready) built.ready(chart);
    // Builders lay out in pixels of the container; the CSS breakpoint changes
    // that height, so a resize that crosses it needs a rebuild, not a resize.
    var h0 = el.offsetHeight;
    var resize = function () {
      if (el.offsetHeight !== h0) scheduleRender();
      else chart.resize();
    };
    window.addEventListener("resize", resize);
    live.push({ chart: chart, resize: resize, controls: built.controls });
  }

  var renderTimer = null;
  function scheduleRender() {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(render, 150);
  }

  function render() {
    teardown();
    var els = [].slice.call(document.querySelectorAll("[data-chart]"));
    if (!els.length || typeof echarts === "undefined") return;
    var src = els[0].dataset.src || "charts.json";
    if (!fetched[src]) {
      fetched[src] = fetch(src).then(function (r) {
        if (!r.ok) throw new Error(r.status + " " + r.statusText);
        return r.json();
      });
    }
    fetched[src]
      .then(function (data) {
        els.forEach(function (el) {
          draw(el, data);
        });
      })
      .catch(function (err) {
        // Leave the static PNG fallback in place if it is still there.
        els.forEach(function (el) {
          if (!el.querySelector("img")) {
            el.textContent = "Could not load chart data (" + err.message + ").";
          }
        });
      });
  }

  window.ANNCharts = {
    register: function (name, fn) {
      builders[name] = fn;
    },
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(render); // Material's instant navigation swaps the DOM
  } else {
    document.addEventListener("DOMContentLoaded", render);
  }

  // The palette toggle changes colours that are baked into each option.
  new MutationObserver(function (muts) {
    if (
      muts.some(function (m) {
        return m.attributeName === "data-md-color-scheme";
      })
    ) {
      render();
    }
  }).observe(document.body, { attributes: true });
})();

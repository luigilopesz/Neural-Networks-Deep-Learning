/* Figure 5 — per-class distribution of the radius ‖x‖, both 5D datasets (Exercise 2, C.3).
 *
 * Left  (Dataset I, shifted Gaussians): A and B differ in radius only as a side effect
 *       of the A→B mean shift — visibly separated, real overlap.
 * Right (Dataset II, concentric shells): radius IS the class signal. C (core, ρ~N(2,0.4))
 *       and D (shell, ρ~N(5,0.4)) sit on opposite sides of the same panel that showed
 *       Figure 4's two circular, overlapping PCA clouds — same centre (centroidDist.II =
 *       0.2662), but the magnitude histograms barely touch.
 *
 * The slider is the point of the figure: dragging the Dataset II threshold across the
 * empty gap between the shells keeps the "core if ‖x‖ < T" rule at 100% for a whole band
 * of T, not just at the one value quoted in the report — that band IS what "trivially
 * separable in magnitude" means.
 */
ANNCharts.register("fig5", function (data, H, el) {
  "use strict";

  var ex = data.ex2;
  var C = ex.colors;
  var R = ex.radius;
  var T0 = ex.threshold;
  var SLIDER_MIN = 0.6;
  var SLIDER_MAX = 7.0;
  var N_BINS = 24;

  function mean(arr) {
    return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length;
  }

  var binsI = H.bins(R.A.concat(R.B), N_BINS);
  var binsII = H.bins(R.C.concat(R.D), N_BINS);
  var padI = binsI.w * 0.5;
  var xMinI = 0;
  var xMaxI = +(binsI.hi + padI).toFixed(3);

  var gap = mean(R.D) - mean(R.C);

  // ------------------------------------------------------------- layout
  // CSS fixes the container height (460px / 380px narrow); everything else is
  // budgeted in px, top to bottom: heading · panel titles · legends · grids.
  function geom(w, h) {
    var narrow = w < 760;
    var m = narrow ? 50 : 56; // left gutter: y labels + rotated y-axis name
    var gp = narrow ? 58 : 72; // between panels: holds panel II's y axis
    var pad = narrow ? 18 : 24; // right margin, for the last x label
    var top = 100;
    var bot = 46; // x labels + x-axis name
    var pw = Math.max(80, (w - m - gp - pad) / 2);
    var ph = Math.max(80, h - top - bot);
    return { compact: w < 560, top: top, pw: pw, ph: ph, l1: m, l2: m + pw + gp };
  }

  function panel(text, sub, cx, top) {
    return Object.assign(H.panelTitle(text, cx, top), {
      subtext: sub,
      subtextStyle: { color: H.ink(), fontSize: 10, opacity: 0.68, align: "center" },
      itemGap: 2, // keep the title+subtext block short enough to clear the legend below it
    });
  }

  function frame(g) {
    return {
      grid: [
        { left: g.l1, top: g.top, width: g.pw, height: g.ph },
        { left: g.l2, top: g.top, width: g.pw, height: g.ph },
      ],
      title: [
        H.heading(
          "Figure 5 · Distribution of ‖x‖ per class, both 5D datasets",
          g.compact
            ? "Dataset II: centroids 0.2662 apart, radii gap " + gap.toFixed(2)
            : "Dataset II's class centroids nearly coincide (‖μ_C − μ_D‖ = " +
                ex.centroidDist.II +
                ") yet the radii separate by " +
                gap.toFixed(2) +
                " — same center, different spread"
        ),
        panel(
          "Dataset I · A vs B",
          "means " + mean(R.A).toFixed(2) + " / " + mean(R.B).toFixed(2) + " — separated, real overlap",
          g.l1 + g.pw / 2,
          g.top - 54
        ),
        panel(
          "Dataset II · C (core) vs D (shell)",
          "means " + mean(R.C).toFixed(2) + " / " + mean(R.D).toFixed(2) + " — essentially non-overlapping",
          g.l2 + g.pw / 2,
          g.top - 54
        ),
      ],
      legend: [
        H.legend({ data: ["Class A", "Class B"], left: g.l1, top: g.top - 20, padding: [2, 6] }),
        H.legend({ data: ["Class C (core)", "Class D (shell)"], left: g.l2, top: g.top - 20, padding: [2, 6] }),
      ],
    };
  }

  function markLineFor(T) {
    return {
      silent: true,
      symbol: "none",
      lineStyle: { color: H.ink(), width: 1.6, type: [6, 4] },
      label: {
        formatter: "T = " + T.toFixed(4),
        color: H.ink(),
        fontSize: 11,
        position: "insideEndTop",
      },
      data: [{ xAxis: T }],
    };
  }

  var g0 = geom(el.clientWidth || 900, el.clientHeight || 460);

  var option = Object.assign(
    {
      tooltip: {
        trigger: "item",
        confine: true,
        backgroundColor: H.paper(),
        borderColor: H.faint(),
        textStyle: { color: H.ink(), fontSize: 11 },
        formatter: function (p) {
          var w = p.seriesIndex < 2 ? binsI.w : binsII.w;
          var c = p.value[0];
          return (
            p.marker +
            "<b>" +
            p.seriesName +
            "</b><br/>‖x‖ " +
            (c - w / 2).toFixed(2) +
            " – " +
            (c + w / 2).toFixed(2) +
            "<br/>count <b>" +
            p.value[1] +
            "</b>"
          );
        },
      },
      xAxis: [
        H.axis("‖x‖ (Euclidean norm)", { type: "value", gridIndex: 0, min: xMinI, max: xMaxI }),
        H.axis("‖x‖ (Euclidean norm)", { type: "value", gridIndex: 1, min: 0, max: SLIDER_MAX }),
      ],
      yAxis: [
        H.axis("Count", { type: "value", gridIndex: 0, min: 0, nameGap: 40 }),
        H.axis("Count", { type: "value", gridIndex: 1, min: 0, nameGap: 40 }),
      ],
      series: [
        H.bar("Class A", C.A, H.counts(R.A, binsI), { xAxisIndex: 0, yAxisIndex: 0 }),
        H.bar("Class B", C.B, H.counts(R.B, binsI), { xAxisIndex: 0, yAxisIndex: 0 }),
        H.bar("Class C (core)", C.C, H.counts(R.C, binsII), { xAxisIndex: 1, yAxisIndex: 1 }),
        H.bar("Class D (shell)", C.D, H.counts(R.D, binsII), {
          xAxisIndex: 1,
          yAxisIndex: 1,
          markLine: markLineFor(T0),
        }),
      ],
    },
    frame(g0)
  );

  var ui = H.controls(
    el,
    '<label for="fig5-t">Dataset II threshold T</label>' +
      '<input id="fig5-t" type="range" min="' +
      SLIDER_MIN +
      '" max="' +
      SLIDER_MAX +
      '" step="0.01" value="' +
      T0 +
      '">' +
      '<output for="fig5-t">' +
      T0.toFixed(4) +
      "</output>" +
      '<span class="chart-readout">rule: core (C) if ‖x‖ &lt; T, else shell (D) &nbsp;→&nbsp; accuracy ' +
      '<strong id="fig5-acc">—</strong></span>'
  );

  return {
    option: option,
    controls: ui,
    ready: function (chart) {
      var input = ui.querySelector("input");
      var out = ui.querySelector("output");
      var acc = ui.querySelector("#fig5-acc");

      function accuracyAt(T) {
        var ok = 0,
          i;
        for (i = 0; i < R.C.length; i++) if (R.C[i] < T) ok++;
        for (i = 0; i < R.D.length; i++) if (R.D[i] >= T) ok++;
        return ok / (R.C.length + R.D.length);
      }

      function apply(T) {
        out.textContent = T.toFixed(4);
        acc.textContent = H.pct(accuracyAt(T), 2);
        // Only series index 3 (Class D) carries the markLine; the empty objects
        // leave the other three series (and their bar data) untouched.
        chart.setOption({ series: [{}, {}, {}, { markLine: markLineFor(T) }] });
      }
      input.addEventListener("input", function () {
        apply(parseFloat(input.value));
      });
      apply(T0);

      // The pixel geometry above has to be recomputed on resize; the shared
      // plumbing only calls chart.resize(). Self-disconnects once disposed
      // (dark-mode toggle and instant navigation both dispose and rebuild).
      var ro = null;
      function fit() {
        if (chart.isDisposed()) {
          if (ro) ro.disconnect();
          return;
        }
        chart.resize(); // container-only resizes don't fire the window listener
        chart.setOption(frame(geom(chart.getWidth(), chart.getHeight())));
      }
      fit();
      if (typeof ResizeObserver === "function") {
        ro = new ResizeObserver(fit);
        ro.observe(el);
      }
    },
  };
});

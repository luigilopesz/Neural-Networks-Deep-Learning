/* Figure 4 — PCA(2) projections of the two 5D datasets of Exercise 2, side by side.
 *
 * Left  (Dataset I, shifted correlated Gaussians): the class signal is a mean shift,
 *       which is linear, so PCA sees it — A and B split along PC1.
 * Right (Dataset II, concentric spherical shells): the class signal is a radius, which
 *       no linear projection can expose — C and D collapse into two overlapping
 *       roughly circular clouds.
 *
 * The right panel is drawn on a SQUARE grid with equal x/y spans, i.e. equal aspect
 * (matplotlib's ax.set_aspect("equal")). Radially symmetric data on a stretched panel
 * would render circular shells as ellipses, inventing a preferred direction that is
 * not in the data. The left panel is free to fill its slot: its structure is directional.
 */
ANNCharts.register("fig4", function (data, H, el) {
  var ex = data.ex2;
  var C = ex.colors;

  var totI = ex.evrI[0] + ex.evrI[1];
  var totII = ex.evrII[0] + ex.evrII[1];

  // --- panel II gets equal aspect: a square grid plus equal data spans on both axes.
  var span = function (rows, k) {
    var lo = Infinity,
      hi = -Infinity;
    rows.forEach(function (p) {
      if (p[k] < lo) lo = p[k];
      if (p[k] > hi) hi = p[k];
    });
    return { c: (lo + hi) / 2, w: hi - lo };
  };
  var sx = span(ex.pcaII, 0);
  var sy = span(ex.pcaII, 1);
  var half = Math.ceil((Math.max(sx.w, sy.w) / 2) * 1.06); // whole units, so the ticks stay clean
  var sq = function (s) {
    return { min: Math.round(s.c) - half, max: Math.round(s.c) + half };
  };

  // --- geometry, in px: the vertical budget is fixed by CSS (460px, 380px narrow),
  // so titles/legend/axis names get explicit room and the grids take what is left.
  // Stacked from the top: heading 2..38, panel titles 44..73, legend 80..95, grid 100.
  function geom(w, h) {
    var narrow = w < 760;
    var m = narrow ? 92 : 104; // left gutter: y tick labels + rotated y-axis name (nameGap 28 + text)
    var gap = narrow ? 64 : 76; // between panels: holds panel II's y axis
    var pad = narrow ? 18 : 24; // right margin, for the last x label
    var top = 100;
    var bot = 64; // x tick labels + x-axis name (nameGap 14 + text)
    var pw = Math.max(80, (w - m - gap - pad) / 2);
    var ph = Math.max(80, h - top - bot);
    var side = Math.min(pw, ph); // panel II: square
    return {
      compact: w < 560,
      top: top,
      pw: pw,
      ph: ph,
      side: side,
      l1: m,
      l2: m + pw + gap + (pw - side) / 2,
      t2: top + (ph - side) / 2,
    };
  }

  // Components anchored to the grids; recomputed whenever the container resizes.
  function frame(g) {
    return {
      grid: [
        { left: g.l1, top: g.top, width: g.pw, height: g.ph },
        { left: g.l2, top: g.t2, width: g.side, height: g.side },
      ],
      title: [
        H.heading(
          "Figure 4 · PCA(2) projections of the two 5D datasets",
          g.compact
            ? "PCA(2) keeps " + H.pct(totI, 2) + " vs " + H.pct(totII, 2) + " of the variance"
            : "PCA(2) keeps " +
                H.pct(totI, 2) +
                " of Dataset I's variance, but only " +
                H.pct(totII, 2) +
                " of Dataset II's"
        ),
        panel("Dataset I · shifted Gaussians", "PC1 separates A from B", g.l1 + g.pw / 2, g.top - 56),
        panel("Dataset II · concentric shells", "equal aspect · C and D overlap", g.l2 + g.side / 2, g.top - 56),
        {
          text: "shift + wheel to zoom · drag to pan",
          left: 2,
          bottom: 1,
          textStyle: { color: H.ink(), fontSize: 10, opacity: 0.55, fontWeight: "normal" },
        },
      ],
      legend: [
        H.legend({ data: ["Class A", "Class B"], left: g.l1, top: g.top - 20, padding: [2, 6] }),
        H.legend({ data: ["Class C (core)", "Class D (shell)"], left: g.l2, top: g.top - 20, padding: [2, 6] }),
      ],
    };
  }

  function panel(text, sub, cx, top) {
    return Object.assign(H.panelTitle(text, cx, top), {
      subtext: sub,
      subtextStyle: { color: H.ink(), fontSize: 10, opacity: 0.68, align: "center" },
      itemGap: 2, // keep the title+subtext block short enough to clear the legend below it
    });
  }

  // 500 points per class overplot heavily. Small semi-transparent symbols with no
  // border read as density; `large: true` is not used — at 500 points per series it
  // buys nothing and costs the per-point hover and the focus-on-hover fade.
  function cloud(name, color, rows, ax) {
    return H.dot(name, color, rows, {
      symbolSize: 5,
      xAxisIndex: ax,
      yAxisIndex: ax,
      itemStyle: { color: color, opacity: 0.5 },
      emphasis: { focus: "series", scale: 1.7, itemStyle: { opacity: 1 } },
      blur: { itemStyle: { opacity: 0.12 } },
    });
  }

  function ax(pc, evr, i, extra) {
    return H.axis("PC" + pc + " (" + H.pct(evr) + " of variance)", Object.assign({ gridIndex: i, scale: true }, extra));
  }

  var g0 = geom(el.clientWidth || 900, el.clientHeight || 460);
  var option = Object.assign(
    {
      tooltip: {
        trigger: "item",
        confine: true,
        formatter: function (p) {
          return (
            '<b style="color:' +
            p.color +
            '">' +
            p.seriesName +
            "</b><br/>PC1 " +
            p.value[0].toFixed(3) +
            "<br/>PC2 " +
            p.value[1].toFixed(3)
          );
        },
      },
      xAxis: [ax(1, ex.evrI[0], 0, { nameGap: 14 }), Object.assign(ax(1, ex.evrII[0], 1, { nameGap: 14 }), sq(sx))],
      yAxis: [
        ax(2, ex.evrI[1], 0, { nameRotate: 90, nameGap: 28 }),
        Object.assign(ax(2, ex.evrII[1], 1, { nameRotate: 90, nameGap: 28 }), sq(sy)),
      ],
      // Plain wheel must keep scrolling the page — this figure sits mid-article.
      dataZoom: [0, 1].map(function (i) {
        return {
          type: "inside",
          xAxisIndex: i,
          yAxisIndex: i,
          filterMode: "none",
          zoomOnMouseWheel: "shift",
          moveOnMouseMove: true,
          moveOnMouseWheel: false,
        };
      }),
      series: [
        cloud("Class A", C.A, ex.pcaI.slice(0, 500), 0),
        cloud("Class B", C.B, ex.pcaI.slice(500), 0),
        cloud("Class C (core)", C.C, ex.pcaII.slice(0, 500), 1),
        cloud("Class D (shell)", C.D, ex.pcaII.slice(500), 1),
      ],
    },
    frame(g0)
  );

  return {
    option: option,
    ready: function (chart) {
      var apply = function () {
        chart.setOption(frame(geom(chart.getWidth(), chart.getHeight())));
      };
      apply();
      // The shared plumbing only calls chart.resize(); the px geometry above has to
      // be recomputed too. Self-disconnects once the chart is disposed (dark-mode
      // toggle and instant navigation both dispose and rebuild).
      if (typeof ResizeObserver === "function") {
        var ro = new ResizeObserver(function () {
          if (chart.isDisposed()) return ro.disconnect();
          chart.resize();
          apply();
        });
        ro.observe(el);
      }
    },
  };
});

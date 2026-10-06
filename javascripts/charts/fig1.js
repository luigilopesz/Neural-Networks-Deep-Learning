/* Figure 1 — the four Gaussian classes at s = 1.0 (Exercise 1, parts A and C.2).
 *
 * Two layout decisions, both deliberate:
 *
 * 1. Equal scale on both axes. The data is wide (x spans ~18 units) and short
 *    (y spans ~14), and Class 2 is the only isotropic cloud (std [0.9, 0.9]).
 *    If ECharts were left to stretch the grid to the container, that round
 *    cloud would render as an ellipse and the figure would contradict the text.
 *    So the grid box is sized to give both axes the same pixels-per-unit and is
 *    centred in whatever is left; the spare width becomes margin, not distortion.
 *    Recomputed on resize, since the plumbing only calls chart.resize().
 * 2. The x-range is NOT broken to close the gap in front of Class 3. That gap
 *    is the point of Class 3 — zoom/pan is what makes the crowded left readable.
 *
 * The boundary polylines are Voronoi *cell outlines*, so every interior edge
 * appears twice (once per adjacent cell). Drawn as-is, each dashed line is
 * overprinted at a different dash phase and reads as a solid smear, so shared
 * segments are dropped and the survivors stitched back into long chains.
 */
ANNCharts.register("fig1", function (data, H, el) {
  var ex = data.ex1;
  var colors = ex.colors;

  // A filled plus, turned 45° — the black-edged 'X' of the static original.
  var CROSS =
    "path://M37,0 L63,0 L63,37 L100,37 L100,63 L63,63 L63,100 L37,100 L37,63 L0,63 L0,37 L37,37 Z";

  function className(i) {
    return "Class " + i;
  }

  // Print exactly what charts.json holds — no re-rounding of a stored number.
  function num(v) {
    return String(v);
  }

  // ---------------------------------------------------------------- extent
  // Whole numbers so the ticks stay clean; the boundary clips to the grid.
  var xs = ex.partA.map(function (p) { return p[0]; });
  var ys = ex.partA.map(function (p) { return p[1]; });
  var x0 = Math.floor(Math.min.apply(null, xs) - 0.6);
  var x1 = Math.ceil(Math.max.apply(null, xs) + 0.6);
  var y0 = Math.floor(Math.min.apply(null, ys) - 0.6);
  var y1 = Math.ceil(Math.max.apply(null, ys) + 0.6);
  var xSpan = x1 - x0;
  var ySpan = y1 - y0;

  // Space reserved outside the plot box: axis name + labels, title, legend.
  var PAD = { l: 56, r: 22, t: 80, b: 58 };

  function gridFor(w, h) {
    var aw = Math.max(60, (w || 900) - PAD.l - PAD.r);
    var ah = Math.max(60, (h || 620) - PAD.t - PAD.b);
    var gw = Math.min(aw, (ah * xSpan) / ySpan); // equal pixels per unit
    var gh = (gw * ySpan) / xSpan;
    return {
      left: Math.round(PAD.l + (aw - gw) / 2),
      top: Math.round(PAD.t + (ah - gh) / 2),
      width: Math.round(gw),
      height: Math.round(gh),
    };
  }

  // ------------------------------------------------------------- boundary
  // Keep each edge once, in the longest runs the input order allows.
  function chains(polys) {
    var seen = {};
    var out = [];
    polys.forEach(function (poly) {
      var run = [];
      for (var i = 1; i < poly.length; i++) {
        var a = poly[i - 1];
        var b = poly[i];
        var key =
          a[0] < b[0] || (a[0] === b[0] && a[1] <= b[1])
            ? a[0] + "," + a[1] + "|" + b[0] + "," + b[1]
            : b[0] + "," + b[1] + "|" + a[0] + "," + a[1];
        if (seen[key]) {
          if (run.length > 1) out.push(run);
          run = [];
          continue;
        }
        seen[key] = 1;
        if (!run.length) run.push(a);
        run.push(b);
      }
      if (run.length > 1) out.push(run);
    });
    return out;
  }

  // --------------------------------------------------------------- series
  var series = [
    {
      name: "Nearest-center boundary",
      type: "lines",
      coordinateSystem: "cartesian2d",
      polyline: true,
      silent: true, // geometry, not data: never steals a tooltip
      z: 1,
      data: chains(ex.boundary).map(function (c) {
        return { coords: c };
      }),
      lineStyle: {
        color: H.ink(),
        width: 1.1,
        opacity: H.dark() ? 0.55 : 0.45,
        type: [5, 4],
      },
    },
  ];

  colors.forEach(function (color, i) {
    series.push(
      H.dot(
        className(i),
        color,
        ex.partA
          .filter(function (p) { return p[2] === i; })
          .map(function (p) { return [p[0], p[1]]; }),
        { z: 3 }
      )
    );
  });

  // The true means. Same series name as the cloud, so one legend entry hides
  // a class and its mean together.
  colors.forEach(function (color, i) {
    series.push({
      name: className(i),
      type: "scatter",
      z: 5,
      symbol: CROSS,
      symbolSize: 13,
      symbolRotate: 45,
      symbolKeepAspect: true,
      data: [{ value: ex.means[i], name: "mean" }],
      itemStyle: { color: color, borderColor: H.ink(), borderWidth: 1.4 },
      emphasis: { scale: 1.2 },
      blur: { itemStyle: { opacity: 1 } }, // stays put while a cloud is focused
    });
  });

  var option = {
    title: H.heading(
      "Figure 1 · Four Gaussian classes at s = 1.0",
      "100 points per class · ✕ = true mean · dashed = nearest-center partition"
    ),
    legend: H.legend({
      top: 50,
      left: "center",
      width: "92%",
      itemGap: 18,
      data: colors.map(function (_, i) { return className(i); }),
    }),
    grid: gridFor(el.clientWidth, el.clientHeight),
    tooltip: {
      trigger: "item",
      confine: true,
      backgroundColor: H.paper(),
      borderColor: H.faint(),
      textStyle: { color: H.ink(), fontSize: 11 },
      formatter: function (p) {
        var v = p.value;
        return (
          p.marker +
          "<b>" +
          p.seriesName +
          (p.name === "mean" ? " · true mean" : "") +
          "</b><br/>Feature 1 &nbsp;<b>" +
          num(v[0]) +
          "</b><br/>Feature 2 &nbsp;<b>" +
          num(v[1]) +
          "</b>"
        );
      },
    },
    xAxis: H.axis("Feature 1", { type: "value", min: x0, max: x1 }),
    yAxis: H.axis("Feature 2", { type: "value", min: y0, max: y1, nameGap: 40 }),
    // Both axes zoom by the same wheel step, so the equal scale survives.
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: "shift",
        moveOnMouseWheel: false,
        moveOnMouseMove: true,
      },
      {
        type: "inside",
        yAxisIndex: 0,
        filterMode: "none",
        zoomOnMouseWheel: "shift",
        moveOnMouseWheel: false,
        moveOnMouseMove: true,
      },
    ],
    series: series,
  };

  var ui = H.controls(
    el,
    '<span class="chart-readout">Closest pair: <strong>Class 0 / Class 1</strong>' +
      ', separation ratio r<sub>01</sub> = <strong>' + num(ex.r_ij["01"]) + "</strong></span>" +
      "<label>Shift + scroll to zoom · drag to pan</label>" +
      '<button type="button" style="margin-left:auto">Reset view</button>'
  );

  return {
    option: option,
    controls: ui,
    ready: function (chart) {
      var ro = null;
      var fit = function () {
        if (chart.isDisposed()) {
          if (ro) ro.disconnect();
          return;
        }
        chart.setOption({ grid: gridFor(el.clientWidth, el.clientHeight) });
      };
      if (typeof ResizeObserver !== "undefined") {
        ro = new ResizeObserver(fit);
        ro.observe(el);
      }
      ui.querySelector("button").addEventListener("click", function () {
        chart.setOption({ dataZoom: [{ start: 0, end: 100 }, { start: 0, end: 100 }] });
      });
    },
  };
});

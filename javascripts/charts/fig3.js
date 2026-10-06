/* Figure 3 — mixing rate against the spread scale s (Exercise 1, part B.4).
 *
 * Four data points, so the figure has to argue rather than merely describe.
 * The means are fixed and only the spread scales, so the smallest pairwise
 * separation ratio r01 falls exactly as 1.3258 / s. Below r01 = 1 the two
 * closest class means sit nearer to each other than their combined average
 * spread; from the first sampled scale where that holds, s = 2, the plane is
 * shaded and annotated in place — no second series or axis needed to carry
 * that one fact.
 */
ANNCharts.register("fig3", function (data, H) {
  var ex = data.ex1;
  var N = ex.partA.length; // 400 points per draw
  var ink = H.ink();
  // Class 1's hue: the mixing is driven almost entirely by the 0/1 border,
  // the pair whose separation ratio r01 is the smallest of the six.
  var accent = ex.colors[1];

  function rgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }
  function l2(s) {
    return Math.log(s) / Math.LN2;
  }

  // The scales are geometric, so the x axis is log2: ticks land on 0.5/1/2/4.
  var XMIN = -1.5;
  var XMAX = 2.5;
  var LOST = l2(2); // first sampled scale with r01 < 1
  var rLost = ex.r01 / 2;

  var mixed = ex.scales.map(function (s) {
    var rate = ex.mixing[s.toFixed(1)];
    return { value: [l2(s), rate], s: s, n: Math.round(rate * N) };
  });

  var x = H.axis("spread scale  s   (log₂ axis)", {
    type: "value",
    min: XMIN,
    max: XMAX,
    interval: 0.5, // half-steps keep the 0.5/1/2/4 ticks on the interval grid
    nameGap: 32,
    splitLine: { show: false },
  });
  x.axisLabel = Object.assign({}, x.axisLabel, {
    fontSize: 12,
    formatter: function (v) {
      if (v !== Math.round(v)) return ""; // label only 0.5, 1, 2, 4
      var s = Math.pow(2, v);
      return s < 1 ? s.toFixed(1) : String(s);
    },
  });

  var yLeft = H.axis("mixing rate", {
    min: 0,
    max: 0.55,
    interval: 0.1,
    nameGap: 42,
  });
  yLeft.axisLabel = Object.assign({}, yLeft.axisLabel, {
    formatter: function (v) {
      return H.pct(v, 0);
    },
  });

  return {
    title: H.heading(
      "Figure 3 — Mixing rate against the spread scale  s",
      "share of " + N + " points closer to another class's center"
    ),
    legend: H.legend({ top: 46, left: "center" }),
    grid: { left: 30, right: 24, top: 86, bottom: 44, containLabel: true },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", snap: true, lineStyle: { color: H.faint(), width: 1 } },
      backgroundColor: H.paper(),
      borderColor: H.faint(),
      borderWidth: 1,
      textStyle: { color: ink, fontSize: 12 },
      extraCssText: "box-shadow:0 2px 12px rgba(0,0,0,0.16)",
      formatter: function (ps) {
        if (!ps.length) return "";
        var p = ps[0];
        var dim = rgba(ink, 0.66);
        return (
          '<div style="font-weight:700;margin-bottom:2px">s = ' + p.data.s.toFixed(1) + "</div>" +
          "<div>" + p.marker + "mixing rate &nbsp;<b>" + H.pct(p.data.value[1], 2) + "</b>" +
          '<span style="color:' + dim + '"> &nbsp;' + p.data.n + " / " + N + " points</span></div>"
        );
      },
    },
    xAxis: x,
    yAxis: yLeft,
    series: [
      {
        name: "mixing rate",
        type: "line",
        data: mixed,
        z: 5,
        symbol: "circle",
        symbolSize: 11,
        lineStyle: { color: accent, width: 2.6 },
        itemStyle: { color: accent, borderColor: H.paper(), borderWidth: 2 },
        areaStyle: { color: rgba(accent, H.dark() ? 0.16 : 0.1) },
        emphasis: { scale: 1.35 },
        label: {
          show: true,
          position: "top",
          distance: 9,
          color: ink,
          fontSize: 12,
          fontWeight: 700,
          backgroundColor: H.paper(),
          padding: [2, 3],
          formatter: function (p) {
            return H.pct(p.data.value[1], 2);
          },
        },
        // The threshold the whole argument turns on, drawn twice: a crisp line
        // at s = 2 and the shaded half-plane beyond it, labelled with the
        // exact r01 value — the one number that decides the punchline.
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          lineStyle: { color: rgba(accent, 0.7), type: [6, 5], width: 1.6 },
          data: [{ xAxis: LOST }],
        },
        markArea: {
          silent: true,
          itemStyle: { color: rgba(accent, H.dark() ? 0.12 : 0.08) },
          label: {
            show: true,
            position: "insideTopLeft",
            distance: 10,
            lineHeight: 15,
            align: "left",
            color: rgba(ink, 0.92),
            fontSize: 11,
            formatter:
              "linear separability lost\nr₀₁ = " + rLost.toFixed(4) + " < 1 from s = 2",
          },
          data: [[{ xAxis: LOST }, { xAxis: XMAX }]],
        },
      },
    ],
  };
});

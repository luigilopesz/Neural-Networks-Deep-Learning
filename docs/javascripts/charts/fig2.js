/* Figure 2 — the same four classes regenerated at four spread scales.
 *
 * Exercise 1, part B.1: four independent 400-point draws, s ∈ {0.5, 1, 2, 4}.
 * The point of the figure is the comparison, so all four panels are pinned to
 * one shared pair of axis limits (the union of the four draws) and one shared
 * zoom window — nothing here is allowed to autoscale per panel.
 */
ANNCharts.register("fig2", function (data, H, el) {
  var ex = data.ex1;
  var scales = ex.scales;
  var key = function (s) {
    return s.toFixed(1); // charts.json keys the draws "0.5" … "4.0"
  };
  var names = ex.colors.map(function (_, c) {
    return "Class " + c;
  });

  // ---- shared limits: the union of all four draws, padded, rounded outward.
  var lo = [Infinity, Infinity];
  var hi = [-Infinity, -Infinity];
  scales.forEach(function (s) {
    ex.points[key(s)].forEach(function (p) {
      for (var k = 0; k < 2; k++) {
        if (p[k] < lo[k]) lo[k] = p[k];
        if (p[k] > hi[k]) hi[k] = p[k];
      }
    });
  });
  var lim = [0, 1].map(function (k) {
    var pad = (hi[k] - lo[k]) * 0.04;
    return [Math.floor((lo[k] - pad) / 5) * 5, Math.ceil((hi[k] + pad) / 5) * 5]; // multiples of 5: clean ticks
  });

  // ---- layout, in pixels of the actual container. Everything above and below
  // the panels is fixed-size text, so the panels get whatever is left:
  //   heading 0..44 | legend 50 | row-0 title 78 | row-0 grid 98..98+G
  //   | x labels + name 42 | gap 12 | row-1 title | 20 | row-1 grid | 42 | 4
  // => 2G = H - 218. The plumbing re-renders when the container height changes.
  var H_PX = el.offsetHeight || 620;
  var G = Math.max(120, (H_PX - 218) / 2);
  var COL = [10, 58];
  var W = 36;
  var LEGEND_TOP = 50;
  var ROW = [98, 98 + G + 42 + 12 + 20 + 20];
  var TITLE_GAP = 20;

  var grids = [];
  var xAxes = [];
  var yAxes = [];
  var series = [];
  var sOf = []; // scale of each series, for the tooltip
  var titles = [
    H.heading(
      "Figure 2 — The same four classes at four spread scales",
      "Identical axis limits everywhere — shift-scroll to zoom, drag to pan"
    ),
  ];

  scales.forEach(function (s, i) {
    var left = COL[i % 2];
    var top = ROW[Math.floor(i / 2)];

    grids.push({ left: left + "%", width: W + "%", top: top, height: G });
    xAxes.push(
      H.axis("Feature 1", { type: "value", gridIndex: i, min: lim[0][0], max: lim[0][1] })
    );
    yAxes.push(
      H.axis("Feature 2", {
        type: "value",
        gridIndex: i,
        min: lim[1][0],
        max: lim[1][1],
        nameRotate: 90,
        nameGap: 34,
      })
    );
    titles.push(
      H.panelTitle(
        "s = " + s.toFixed(1) + "  ·  mixing " + H.pct(ex.mixing[key(s)], 2),
        left + W / 2 + "%",
        top - TITLE_GAP
      )
    );

    var byClass = [[], [], [], []];
    ex.points[key(s)].forEach(function (p) {
      byClass[p[2]].push([p[0], p[1]]);
    });
    byClass.forEach(function (pts, c) {
      // Same series name in all four panels: four legend entries, and toggling
      // a class hides it everywhere at once.
      var dot = H.dot(names[c], ex.colors[c], pts, {
        xAxisIndex: i,
        yAxisIndex: i,
        symbolSize: 5,
        emphasis: { focus: "none", scale: 2.2, itemStyle: { opacity: 1 } },
      });
      dot.itemStyle.opacity = 0.62; // 400 points in a small panel: let them stack
      dot.itemStyle.borderWidth = 0;
      series.push(dot);
      sOf.push(s);
    });
  });

  return {
    title: titles,
    legend: H.legend({ data: names, top: LEGEND_TOP, left: "center", itemGap: 18 }),
    toolbox: {
      right: 6,
      top: 2,
      itemSize: 13,
      iconStyle: { borderColor: H.ink(), opacity: 0.7 },
      emphasis: { iconStyle: { borderColor: H.ink(), opacity: 1 } },
      feature: { restore: { title: "Reset zoom" } },
    },
    tooltip: {
      trigger: "item",
      backgroundColor: H.paper(),
      borderColor: H.faint(),
      borderWidth: 1,
      padding: [7, 10],
      textStyle: { color: H.ink(), fontSize: 11 },
      formatter: function (p) {
        return (
          p.marker +
          "<b>" +
          p.seriesName +
          "</b>  ·  s = " +
          sOf[p.seriesIndex].toFixed(1) +
          "<br/>Feature 1  " +
          p.value[0] +
          "<br/>Feature 2  " +
          p.value[1]
        );
      },
    },
    // One window per direction (a single dataZoom cannot hold two), each bound to
    // all four panels, so zoom and pan stay linked across the grid.
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: [0, 1, 2, 3],
        filterMode: "none",
        startValue: lim[0][0],
        endValue: lim[0][1],
        zoomOnMouseWheel: "shift", // plain wheel keeps scrolling the page
        moveOnMouseWheel: false,
        moveOnMouseMove: true,
      },
      {
        type: "inside",
        yAxisIndex: [0, 1, 2, 3],
        filterMode: "none",
        startValue: lim[1][0],
        endValue: lim[1][1],
        zoomOnMouseWheel: "shift",
        moveOnMouseWheel: false,
        moveOnMouseMove: true,
      },
    ],
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series: series,
  };
});

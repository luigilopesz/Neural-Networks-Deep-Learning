/* Figure 6 — what log(1 + x) does to a heavy-tailed spending column.
 *
 * FoodCourt on the training split (6,954 passengers), pre-binned in charts.json,
 * raw on the left and after log(1 + x) on the right. The two x axes carry
 * different units and scales, so each panel keeps its own.
 *
 * The count axis is logarithmic and shared by both panels. On a linear count
 * axis the raw panel is one 6,081-passenger bar plus 39 that round to nothing —
 * it reads as a broken render rather than as the finding. A log count axis keeps
 * the spike dominant while making every bin of the tail out to 29,813 credits
 * visible, which is the whole point of the figure.
 */
ANNCharts.register("fig6", function (data, H) {
  "use strict";

  var ex3 = data.ex3;
  var raw = ex3.foodcourtBefore;
  var logged = ex3.foodcourtAfter;
  var stats = ex3.foodcourtStats;

  var total = raw.counts.reduce(function (a, b) {
    return a + b;
  }, 0); // 6,954 passengers

  // 0.1 so the count-1 bins of the far tail still draw a bar instead of sitting
  // flat on the baseline; 10,000 leaves headroom above the 6,081 spike. Both are
  // exact powers of ten — ECharts only snaps a log axis's "nice" ticks to clean
  // 1/10/100/... labels when the extent itself sits on the same grid, otherwise
  // the interior ticks land on ugly values like 5.0118723362727.
  var Y_MIN = 0.1; // a power of ten, so the log ticks land on clean decades
  var Y_MAX = 10000;

  var rawMax = raw.centers[raw.centers.length - 1] + raw.width / 2; // 29,813
  var logMax = logged.centers[logged.centers.length - 1] + logged.width / 2; // 10.303

  // The x-axis "interval" below only produces clean ticks (and avoids ECharts
  // forcing one extra, unrounded tick at the exact data max) when max itself is
  // a multiple of interval, so round each panel's max up to the next one.
  var X_MAX_RAW = Math.ceil(rawMax / 5000) * 5000; // 30,000
  var X_MAX_LOG = Math.ceil(logMax / 2) * 2; // 12

  function whole(v) {
    return (Math.round(v) + 0).toLocaleString("en-US"); // +0 kills -0 from bin edges like -0.0005
  }

  /** Compact axis labels: 1, 10, 100, 1k, 10k / 0, 5k, 10k. Rounded defensively —
   *  floating-point bin math must never leak a long decimal into a tick label. */
  function short(v) {
    v = Math.round(v * 100) / 100;
    return v >= 1000 ? Math.round(v / 100) / 10 + "k" : String(v);
  }

  /** Count-axis label: blank out the Y_MIN=0.1 floor tick — it exists so the
   *  count-1 tail bins draw a bar, not because 0.1 of a passenger means anything. */
  function countLabel(v) {
    return v < 1 ? "" : short(v);
  }

  /** Credits, at a precision that stays honest near zero. */
  function credits(v) {
    return v < 10 ? v.toFixed(1) : whole(v);
  }

  function bars(name, color, h, i) {
    return H.bar(
      name,
      color,
      h.counts.map(function (c, j) {
        return [h.centers[j], c || null]; // a log axis cannot plot a zero count
      }),
      {
        xAxisIndex: i,
        yAxisIndex: i,
        barWidth: "99%",
        itemStyle: { color: color, opacity: 0.88 },
        emphasis: { focus: "series", itemStyle: { opacity: 1 } },
      }
    );
  }

  /** The small line of numbers under a panel title. */
  function note(text, left, top) {
    return {
      text: text,
      left: left,
      top: top,
      textAlign: "center",
      textStyle: { color: H.ink(), fontSize: 10, fontWeight: "normal", opacity: 0.7 },
    };
  }

  var countAxis = function (i) {
    return H.axis("Passengers (log scale)", {
      type: "log",
      logBase: 10,
      min: Y_MIN,
      max: Y_MAX,
      interval: 1, // one decade per tick: 1, 10, 100, 1k, 10k
      gridIndex: i,
      position: i === 0 ? "left" : "right", // mirrored, so the panels share the gutter
      nameRotate: 90, // matches fig2/fig4's convention for a name this long beside a grid
      nameGap: 34,
      axisLabel: { color: H.ink(), fontSize: 11, formatter: countLabel },
    });
  };

  return {
    title: [
      H.heading(
        "Figure 6 · FoodCourt spending before and after log(1 + x)",
        "n = " + whole(total) + " · mean " + stats.mean.toFixed(2) + " vs median " + stats.median.toFixed(2) + " — that gap is the skew"
      ),
      H.panelTitle("Before — raw credits", "26%", 72),
      note(H.pct(raw.counts[0] / total, 1) + " in bin 1 · tail to " + short(rawMax), "26%", 90),
      H.panelTitle("After — log(1 + x)", "74%", 72),
      note(H.pct(logged.counts[0] / total, 1) + " at 0 · rest to " + logMax.toFixed(1), "74%", 90),
    ],

    legend: H.legend({
      top: 54,
      left: "center",
      icon: "roundRect",
      itemGap: 22,
      selectedMode: false, // both panels must always stay visible — no toggle-to-hide
    }),

    grid: [
      { left: 56, right: "54%", top: 110, bottom: 56 },
      { left: "54%", right: 56, top: 110, bottom: 56 },
    ],

    xAxis: [
      H.axis("FoodCourt spend (credits)", {
        type: "value",
        min: 0,
        max: X_MAX_RAW,
        interval: 5000,
        gridIndex: 0,
        axisLabel: { color: H.ink(), fontSize: 11, hideOverlap: true, formatter: short },
      }),
      H.axis("log(1 + FoodCourt spend)", {
        type: "value",
        min: 0,
        max: X_MAX_LOG,
        interval: 2,
        gridIndex: 1,
        axisLabel: { color: H.ink(), fontSize: 11, hideOverlap: true, formatter: short },
      }),
    ],

    yAxis: [countAxis(0), countAxis(1)],

    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: H.ink(), opacity: 0.35 } },
      backgroundColor: H.paper(),
      borderColor: H.faint(),
      textStyle: { color: H.ink(), fontSize: 11 },
      formatter: function (ps) {
        if (!ps.length) return "";
        var first = ps[0];
        var before = first.seriesIndex === 0;
        var h = before ? raw : logged;
        var i = first.dataIndex;
        var lo = h.centers[i] - h.width / 2;
        var hi = lo + h.width;
        var n = h.counts[i];

        var head = before
          ? whole(lo) + " – " + whole(hi) + " credits"
          : "log(1 + x) " + lo.toFixed(2) + " – " + hi.toFixed(2);
        var body = before
          ? ""
          : "<div style='opacity:.7'>≈ " +
            credits(Math.expm1(lo)) +
            " – " +
            credits(Math.expm1(hi)) +
            " credits</div>";

        return (
          "<b>" +
          head +
          "</b>" +
          body +
          "<div>" +
          (n ? whole(n) + " passengers · " + H.pct(n / total, 1) : "no passengers") +
          "</div>"
        );
      },
    },

    series: [
      bars("Raw FoodCourt spend", ex3.colors.before, raw, 0),
      bars("After log(1 + x)", ex3.colors.after, logged, 1),
    ],
  };
});

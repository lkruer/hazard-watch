/* Hazard Watch — location report page.
   Reads /f/<id>.json + /history/<id>.jsonl. Every hazard card leads with a
   plain-word verdict (the audience first, the number second), the freshness
   line is computed from the data rather than asserted, and the trend chart
   is drawn at the container's real pixel size so its text is always legible. */

(function () {
  "use strict";

  var H = window.HW;
  var DATA_ROOT = "../serve/out_live";
  var id = new URLSearchParams(location.search).get("id");
  var sheet = document.getElementById("sheet");

  if (!id) {
    sheet.innerHTML = '<div class="skeleton">This page needs a location ' +
      'to show. <a href="index.html">Pick one on the map →</a></div>';
    return;
  }

  function fetchJSON(u) {
    return fetch(u, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(u + " → " + r.status);
      return r.json();
    });
  }
  function fetchLines(u) {
    return fetch(u, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) return [];
      return r.text().then(function (t) {
        return t.split("\n").filter(Boolean).map(function (l) {
          try { return JSON.parse(l); } catch (e) { return null; }
        }).filter(Boolean);
      });
    }).catch(function () { return []; });
  }

  var esc = H.esc, pct = H.pct, fmtDate = H.fmtDate;

  function maxDefined(vals) {
    var ok = vals.filter(function (v) { return v != null && isFinite(v); });
    return ok.length ? Math.max.apply(null, ok) : null;
  }

  // ---- hazard cards ---------------------------------------------------

  function card(title, verdict, vclass, big, biglabel, sub, frac) {
    var bar = "";
    if (frac != null && isFinite(frac)) {
      bar = '<div class="bar"><i style="--w:' +
        Math.round(Math.max(0, Math.min(1, frac)) * 100) + '%"></i></div>';
    }
    return '<div class="card"><h3>' + title + "</h3>" +
      '<div class="verdict verdict--' + vclass + '">' + verdict + "</div>" +
      '<div class="big">' + big +
      (biglabel ? ' <span class="biglabel">' + biglabel + "</span>" : "") +
      '</div><div class="sub">' + sub + "</div>" + bar + "</div>";
  }

  function hazardCards(d) {
    var w = d.weather || {};
    var st = d.static || {};
    var untrusted = w.data_quality === "disagree";
    var warn = untrusted
      ? ' · <span class="warn">⚠ sources disagree — treat as unknown</span>' : "";
    var cards = [];

    // 1 — landslide: the two-layer model (slopes = static, rain = trigger)
    var rain = maxDefined([w.rain3d_pctl, w.rain30d_pctl]);
    var prone = (st.susceptibility_nearby_max || 0) >= 0.30;
    var lv, lc;
    if (rain == null) { lv = "No fresh rain data"; lc = "unknown"; }
    else if (untrusted) { lv = "Data uncertain"; lc = "unknown"; }
    else if (prone && rain >= 0.98) { lv = "Danger signs on slopes"; lc = "red"; }
    else if (prone && rain >= 0.95) { lv = "Slopes worth watching"; lc = "yellow"; }
    else if (rain >= 0.95) { lv = "Rain unusually heavy"; lc = "yellow"; }
    else { lv = "Nothing unusual"; lc = "green"; }
    cards.push(card("Landslide", lv, lc,
      pct(rain), "rain vs normal",
      "slopes nearby: " + (prone ? "the kind that can fail"
                                 : "not especially failure-prone") +
      " · 3-day rain " + pct(w.rain3d_pctl) +
      " · 30-day " + pct(w.rain30d_pctl) + warn,
      untrusted ? null : rain));

    // 2 — fire weather (trigger model + KBDI fuel dryness)
    var fv, fc;
    if (w.fire_danger == null) { fv = "No data"; fc = "unknown"; }
    else if (w.fire_alert) { fv = "Dangerous fire weather"; fc = "red"; }
    else if (w.fire_watch) { fv = "Elevated"; fc = "yellow"; }
    else { fv = "Normal"; fc = "green"; }
    cards.push(card("Fire weather", fv, fc,
      pct(w.fire_danger), "danger level",
      (w.kbdi != null ? "fuel dryness " + w.kbdi + " / 800" : "no data") +
      (w.fire_date && w.fire_date !== d.as_of
        ? " · fire reading from " + fmtDate(w.fire_date) : ""),
      w.fire_danger));

    // 3 — drought (headline points the SAME direction as the title)
    var dry = (w.spi90 == null || !isFinite(w.spi90)) ? null : 1 - w.spi90;
    var dv, dc;
    if (dry == null) { dv = "No data"; dc = "unknown"; }
    else if (untrusted) { dv = "Data uncertain"; dc = "unknown"; }
    else if (w.spi90 <= 0.02) { dv = "Extreme drought"; dc = "red"; }
    else if (w.spi90 <= 0.05) { dv = "Severe drought"; dc = "red"; }
    else if (w.spi90 <= 0.15) { dv = "Much drier than usual"; dc = "yellow"; }
    else if (w.spi90 <= 0.35) { dv = "Somewhat dry"; dc = "yellow"; }
    else { dv = "No drought"; dc = "green"; }
    cards.push(card("Drought", dv, dc,
      dry == null ? "—" : pct(dry), "dryness",
      "3-month rain vs normal " + pct(w.spi90) +
      " · 12-month " + pct(w.spi365) +
      (w.p_severe_drought != null
        ? " · severe-drought odds " + pct(w.p_severe_drought) : "") + warn,
      untrusted ? null : dry));

    // 4 — river level (GloFAS history + operational forecast)
    var rv, rc;
    if (w.flood_pctl == null) { rv = "No river tracked here"; rc = "unknown"; }
    else if (w.flood_pctl >= 0.98) { rv = "Exceptionally high"; rc = "red"; }
    else if (w.flood_pctl >= 0.95) { rv = "Running high"; rc = "yellow"; }
    else { rv = "Normal range"; rc = "green"; }
    var riverSub = w.flood_pctl == null
      ? (w.flood_note ? esc(w.flood_note) : "no major river at this point")
      : "vs this river’s own record for the season";
    if (w.flood_week_max_pctl != null) {
      riverSub += " · week ahead peaks near " + pct(w.flood_week_max_pctl) +
        " around " + fmtDate(w.flood_week_max_date);
    }
    cards.push(card("River level", rv, rc,
      pct(w.flood_pctl), "vs seasonal record", riverSub, w.flood_pctl));

    return '<div class="grid2">' + cards.join("") + "</div>";
  }

  // ---- trend chart: drawn at real pixel size, redrawn on resize -------

  var LINES = [
    { key: "rain30d_pctl", color: "var(--accent)", w: 2, label: "30-day rain" },
    { key: "fire_danger", color: "var(--red)", w: 1.8, label: "fire danger" },
    { key: "flood_pctl", color: "#8ab4f8", w: 1.8, label: "river" },
    { key: "spi90", color: "var(--yellow)", w: 1.6, dash: "4 3",
      label: "3-month rain (drought when low)" }
  ];

  function trendAria(hist) {
    var n = hist.length;
    var counts = { red: 0, yellow: 0, unknown: 0 };
    hist.forEach(function (h) {
      if (counts[h.status] != null) counts[h.status] += 1;
    });
    var r0 = hist[0].rain30d_pctl, r1 = hist[n - 1].rain30d_pctl;
    var s = "Chart of the last " + n + " days. ";
    if (r0 != null && r1 != null) {
      s += "30-day rain went from " + pct(r0) + " to " + pct(r1) +
        " of normal. ";
    }
    s += "Status was Alert on " + counts.red + " days, Watch on " +
      counts.yellow + ", uncertain on " + counts.unknown +
      ", otherwise normal.";
    return s;
  }

  function drawTrend(hist, box) {
    var W = Math.max(300, box.clientWidth - 4);
    var HT = 190, PL = 44, PR = 10, PT = 12, PB = 40;
    var n = hist.length;
    var xs = function (i) { return PL + (W - PL - PR) * (i / Math.max(1, n - 1)); };
    var ys = function (v) { return PT + (HT - PT - PB) * (1 - v); };

    var grid = [0.25, 0.5, 0.75, 1].map(function (g) {
      return '<line x1="' + PL + '" y1="' + ys(g) + '" x2="' + (W - PR) +
        '" y2="' + ys(g) + '" stroke="rgba(255,255,255,.07)"/>' +
        '<text x="' + (PL - 7) + '" y="' + (ys(g) + 3.5) +
        '" text-anchor="end" font-size="10.5" fill="var(--dim)">' +
        Math.round(g * 100) + "%</text>";
    }).join("");

    var lines = LINES.map(function (L_) {
      var pts = [];
      hist.forEach(function (h, i) {
        var v = h[L_.key];
        if (v == null || !isFinite(v)) return;
        pts.push((pts.length ? "L" : "M") + xs(i).toFixed(1) + " " +
                 ys(v).toFixed(1));
      });
      return '<path d="' + pts.join(" ") + '" fill="none" stroke="' + L_.color +
        '" stroke-width="' + L_.w + '"' +
        (L_.dash ? ' stroke-dasharray="' + L_.dash + '"' : "") + "/>";
    }).join("");

    // contiguous status strip; red days are also TALLER (shape, not just color)
    var step = (W - PL - PR) / Math.max(1, n - 1);
    var stripY = HT - PB + 12;
    var strip = hist.map(function (h, i) {
      var c = { red: "var(--red)", yellow: "var(--yellow)",
                green: "var(--green)", unknown: "var(--unknown)" }[h.status] ||
              "var(--unknown)";
      var tall = h.status === "red";
      return '<rect x="' + (xs(i) - step / 2).toFixed(1) +
        '" y="' + (tall ? stripY - 5 : stripY) +
        '" width="' + (step + 0.6).toFixed(1) +
        '" height="' + (tall ? 13 : 8) + '" fill="' + c + '"/>';
    }).join("");

    var dates =
      '<text x="' + PL + '" y="' + (HT - 4) +
      '" font-size="10.5" fill="var(--dim)">' +
      esc(fmtDate(hist[0].as_of)) + "</text>" +
      '<text x="' + (W - PR) + '" y="' + (HT - 4) +
      '" text-anchor="end" font-size="10.5" fill="var(--dim)">' +
      esc(fmtDate(hist[n - 1].as_of)) + "</text>";

    box.innerHTML =
      '<svg viewBox="0 0 ' + W + " " + HT + '" width="' + W + '" height="' +
      HT + '" role="img" aria-label="' + esc(trendAria(hist)) + '">' +
      grid + lines + strip + dates + "</svg>";
  }

  function trendCard(hist) {
    if (hist.length < 2) return "";
    var legend = LINES.map(function (L_) {
      return '<span><i style="background:' + L_.color + '"></i>' +
        L_.label + "</span>";
    }).join("") + "<span>· colored band = daily status (tall red = alert)</span>";
    return '<div class="card trend"><h3>Recent history</h3>' +
      '<div id="trendbox"></div><div class="legend">' + legend + "</div></div>";
  }

  // ---- page -----------------------------------------------------------

  var TIER_WORD = { A: "high-quality regional model",
                    B: "global model, moderate confidence",
                    C: "limited data here" };

  function freshness(d) {
    var w = d.weather || {};
    if (w.staleness_days === 0) {
      return "weather current through <b>today</b>";
    }
    if (w.staleness_days != null) {
      return "weather is <b>" + w.staleness_days + " day" +
        (w.staleness_days === 1 ? "" : "s") + " behind</b>";
    }
    return d.as_of ? "as of <b>" + esc(fmtDate(d.as_of)) + "</b>" : "";
  }

  function render(d, hist) {
    document.title = d.name + " — Hazard Watch";
    var s = d.status;
    var w = d.weather || {};
    var st = d.static || {};
    var word = H.STATUS_LABEL[s] || s;

    var quality = w.data_quality === "disagree"
      ? '<div class="qnote"><b>Why “data uncertain”:</b> two independent ' +
        "rainfall sources tell different stories at this cell right now (" +
        esc((w.data_quality_reasons || []).join("; ")) +
        "), so rain-based scores are withheld rather than served wrong.</div>"
      : "";

    // provenance + caveats live together in Notes
    var notes = (d.caveats || []).map(function (c) {
      return "<li>" + esc(c) + "</li>";
    });
    if (w.rain_tail_source) {
      notes.push("<li>rain series: " + esc(w.rain_tail_source) + "</li>");
    }
    if (w.fire_tail_source) {
      notes.push("<li>fire series: " + esc(w.fire_tail_source) + "</li>");
    }
    if (w.flood_source) {
      notes.push("<li>river: " + esc(w.flood_source) + "</li>");
    }

    var tier = st.tier_susceptibility
      ? "<span>confidence: <b>" +
        esc(TIER_WORD[st.tier_susceptibility] || st.tier_susceptibility) +
        '</b> · <a href="about.html">how this works</a></span>'
      : "";

    sheet.innerHTML =
      '<section class="hero"><div class="hero__band hero__band--' + s + '"></div>' +
      '<div class="hero__body">' +
      '<div class="hero__top"><div><h1 class="hero__name">' + esc(d.name) + "</h1>" +
      '<div class="hero__coords">' + d.lat.toFixed(3) + ", " + d.lon.toFixed(3) +
      " · id " + esc(d.location_id) + "</div></div>" +
      '<span class="statuspill statuspill--' + s + '"><span class="dot dot--' + s +
      '"></span>' + word + "</span></div>" +
      '<p class="hero__msg" id="speakable">' + esc(d.message) + "</p>" +
      '<div class="hero__meta">' +
      "<span>" + freshness(d) + "</span>" +
      (st.people_10km ? "<span><b>" + st.people_10km.toLocaleString() +
        "</b> people within 10 km</span>" : "") +
      tier + "</div>" +
      '<button class="listen" id="listen" aria-pressed="false">' +
      '<span aria-hidden="true">🔊</span> Listen to this report</button>' +
      "</div></section>" +
      quality +
      hazardCards(d) +
      trendCard(hist) +
      '<div class="card"><h3>Notes</h3><ul class="caveats">' +
      notes.join("") + "</ul></div>" +
      '<footer class="sitefoot">Free &amp; open · built on public data · ' +
      '<a href="about.html">sources &amp; methodology</a></footer>';

    var box = document.getElementById("trendbox");
    if (box) {
      drawTrend(hist, box);
      var t;
      addEventListener("resize", function () {
        clearTimeout(t);
        t = setTimeout(function () { drawTrend(hist, box); }, 150);
      });
    }

    // Web Speech: free, on-device, no server
    var btn = document.getElementById("listen");
    var IDLE = '<span aria-hidden="true">🔊</span> Listen to this report';
    var BUSY = '<span aria-hidden="true">⏹</span> Stop reading';
    function reset() { btn.innerHTML = IDLE; btn.removeAttribute("data-speaking");
                       btn.setAttribute("aria-pressed", "false"); }
    if (!("speechSynthesis" in window)) { btn.style.display = "none"; }
    btn.addEventListener("click", function () {
      if (speechSynthesis.speaking) { speechSynthesis.cancel(); reset(); return; }
      var u = new SpeechSynthesisUtterance(
        d.name + ". Status: " + word + ". " + d.message);
      u.rate = 0.95;
      u.lang = document.documentElement.lang || "en-US";
      u.onend = reset;
      u.onerror = reset;
      btn.innerHTML = BUSY;
      btn.dataset.speaking = "1";
      btn.setAttribute("aria-pressed", "true");
      speechSynthesis.speak(u);
    });
  }

  Promise.all([
    fetchJSON(DATA_ROOT + "/f/" + id + ".json"),
    fetchLines(DATA_ROOT + "/history/" + id + ".jsonl")
  ]).then(function (res) { render(res[0], res[1]); })
    .catch(function (e) {
      var msg = navigator.onLine === false
        ? "You’re offline and this report isn’t saved on your device yet."
        : "Couldn’t load this location. Please try again in a moment.";
      if (window.console && console.error) console.error(e);
      sheet.innerHTML = '<div class="skeleton">' + msg +
        '<br><a href="index.html">← Back to the map</a> · ' +
        '<button class="btn retry" onclick="location.reload()">Try again' +
        "</button></div>";
    });
})();

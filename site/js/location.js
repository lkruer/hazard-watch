/* Hazard Watch — location report page.
   Reads /f/<id>.json + /history/<id>.jsonl, renders the one-color status,
   plain-language sentence, hazard cards, a 180-day trend chart (hand-rolled
   SVG), and a Web Speech "listen" button for low-literacy access. */

(function () {
  "use strict";

  var DATA_ROOT = "../serve/out_live";
  var id = new URLSearchParams(location.search).get("id");
  var sheet = document.getElementById("sheet");
  var STATUS_WORD = { red: "Alert", yellow: "Watch", green: "All normal",
                      unknown: "Data uncertain" };

  if (!id) { sheet.innerHTML = '<div class="skeleton">No location id.</div>'; return; }

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

  function pct(v) { return v == null ? "—" : Math.round(v * 100) + "%"; }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function hazardCards(d) {
    var w = d.weather || {};
    var untrusted = w.data_quality === "disagree";
    var warn = untrusted ? " · ⚠ source disagreement — treat as unknown" : "";
    var cards = [];

    cards.push(card("Rain — landslide trigger",
      pct(Math.max(w.rain3d_pctl || 0, w.rain30d_pctl || 0) || null),
      "3-day " + pct(w.rain3d_pctl) + " · 30-day " + pct(w.rain30d_pctl) +
      " of normal-for-season" + warn,
      Math.max(w.rain3d_pctl || 0, w.rain30d_pctl || 0), false));

    cards.push(card("Fire weather",
      w.fire_danger == null ? "—" : pct(w.fire_danger),
      w.kbdi != null ? "fuel-dryness index " + w.kbdi + " / 800" : "no data",
      w.fire_danger, false));

    cards.push(card("Drought",
      pct(w.spi90),
      "3-month rain vs history · 12-month " + pct(w.spi365) +
      (w.p_severe_drought != null ? " · severe-drought odds " + pct(w.p_severe_drought) : "") + warn,
      w.spi90, true));

    var riverSub = w.flood_source ? esc(w.flood_source)
      : (w.flood_note ? esc(w.flood_note) : "no river at this point");
    if (w.flood_week_max_pctl != null) {
      riverSub = "week ahead peaks at " + pct(w.flood_week_max_pctl) +
        " around " + esc(w.flood_week_max_date) + " · " + riverSub;
    }
    cards.push(card("River level",
      w.flood_pctl == null ? "—" : pct(w.flood_pctl),
      riverSub, w.flood_pctl, false));

    return '<div class="grid2">' + cards.join("") + "</div>";
  }

  function card(title, big, sub, frac, invert) {
    var bar = "";
    if (frac != null && isFinite(frac)) {
      bar = '<div class="bar' + (invert ? " bar--drought" : "") +
        '"><i style="width:' + Math.round(frac * 100) + '%"></i></div>';
    }
    return '<div class="card"><h3>' + title + '</h3><div class="big">' + big +
      '</div><div class="sub">' + sub + "</div>" + bar + "</div>";
  }

  function trendSVG(hist) {
    if (!hist.length) return "";
    var W = 700, H = 180, P = 28;
    var xs = function (i) { return P + (W - 2 * P) * (i / Math.max(1, hist.length - 1)); };
    var ys = function (v) { return H - P - (H - 2 * P) * v; };
    function path(key) {
      var pts = [];
      hist.forEach(function (h, i) {
        var v = h[key];
        if (v == null || !isFinite(v)) return;
        pts.push((pts.length ? "L" : "M") + xs(i).toFixed(1) + " " + ys(v).toFixed(1));
      });
      return pts.join(" ");
    }
    var grid = [0.25, 0.5, 0.75, 1].map(function (g) {
      return '<line x1="' + P + '" y1="' + ys(g) + '" x2="' + (W - P) +
        '" y2="' + ys(g) + '" stroke="rgba(255,255,255,.07)"/>' +
        '<text x="' + (P - 6) + '" y="' + (ys(g) + 3) +
        '" text-anchor="end" font-size="9" fill="#63707f">' + g + "</text>";
    }).join("");
    var lines =
      '<path d="' + path("rain30d_pctl") + '" fill="none" stroke="#3ec2d4" stroke-width="2"/>' +
      '<path d="' + path("fire_danger") + '" fill="none" stroke="#ef5f4f" stroke-width="1.8"/>' +
      '<path d="' + path("flood_pctl") + '" fill="none" stroke="#8ab4f8" stroke-width="1.8"/>' +
      '<path d="' + path("spi90") + '" fill="none" stroke="#ecc23f" stroke-width="1.6" stroke-dasharray="4 3"/>';
    var dots = hist.map(function (h, i) {
      var c = { red: "#ef5f4f", yellow: "#ecc23f", green: "#37c47e",
                unknown: "#8d9aab" }[h.status] || "#8d9aab";
      return '<circle cx="' + xs(i) + '" cy="' + (H - 10) + '" r="3" fill="' + c + '"/>';
    }).join("");
    return '<div class="card trend"><h3>Recent history</h3>' +
      '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="trend chart">' +
      grid + lines + dots + "</svg>" +
      '<div class="legend">' +
      '<span><i style="background:#3ec2d4"></i>30-day rain</span>' +
      '<span><i style="background:#ef5f4f"></i>fire danger</span>' +
      '<span><i style="background:#8ab4f8"></i>river</span>' +
      '<span><i style="background:#ecc23f"></i>3-month rain (drought when low)</span>' +
      "<span>· dots = daily status</span></div></div>";
  }

  function render(d, hist) {
    document.title = d.name + " — Hazard Watch";
    var s = d.status;
    var w = d.weather || {};
    var st = d.static || {};
    var quality = w.data_quality === "disagree"
      ? '<div class="qnote"><b>Why “data uncertain”:</b> two independent ' +
        "rainfall sources tell different stories at this cell right now (" +
        esc((w.data_quality_reasons || []).join("; ")) +
        "), so rain-based scores are withheld rather than served wrong.</div>"
      : "";

    sheet.innerHTML =
      '<section class="hero"><div class="hero__band hero__band--' + s + '"></div>' +
      '<div class="hero__body">' +
      '<div class="hero__top"><div><h1 class="hero__name">' + esc(d.name) + "</h1>" +
      '<div class="hero__coords">' + d.lat.toFixed(3) + ", " + d.lon.toFixed(3) +
      " · id " + esc(d.location_id) + "</div></div>" +
      '<span class="statuspill statuspill--' + s + '"><span class="dot dot--' + s +
      '"></span>' + STATUS_WORD[s] + "</span></div>" +
      '<p class="hero__msg" id="speakable">' + esc(d.message) + "</p>" +
      '<div class="hero__meta">' +
      "<span>as of <b>" + esc(d.as_of || "—") + "</b> (weather data lags ~3 days)</span>" +
      (st.people_10km ? "<span><b>" + st.people_10km.toLocaleString() +
        "</b> people within 10 km</span>" : "") +
      (st.tier_susceptibility ? "<span>confidence tier <b>" +
        esc(st.tier_susceptibility) + (st.region_model ? " · " +
        esc(st.region_model) + " model" : "") + "</b></span>" : "") +
      "</div>" +
      '<button class="listen" id="listen">🔊 Listen to this report</button>' +
      "</div></section>" +
      quality +
      hazardCards(d) +
      trendSVG(hist) +
      '<div class="card"><h3>Notes</h3><ul class="caveats">' +
      (d.caveats || []).map(function (c) { return "<li>" + esc(c) + "</li>"; }).join("") +
      "</ul></div>" +
      '<footer class="sitefoot">Free &amp; open · built on public data · ' +
      '<a href="about.html">sources &amp; methodology</a></footer>';

    // Web Speech: free, on-device, no server
    var btn = document.getElementById("listen");
    if (!("speechSynthesis" in window)) { btn.style.display = "none"; }
    btn.addEventListener("click", function () {
      if (speechSynthesis.speaking) { speechSynthesis.cancel(); btn.dataset.speaking = ""; return; }
      var u = new SpeechSynthesisUtterance(
        d.name + ". Status: " + STATUS_WORD[s] + ". " + d.message);
      u.rate = 0.95;
      u.onend = function () { btn.dataset.speaking = ""; };
      btn.dataset.speaking = "1";
      speechSynthesis.speak(u);
    });
  }

  Promise.all([
    fetchJSON(DATA_ROOT + "/f/" + id + ".json"),
    fetchLines(DATA_ROOT + "/history/" + id + ".jsonl")
  ]).then(function (res) { render(res[0], res[1]); })
    .catch(function (e) {
      sheet.innerHTML = '<div class="skeleton">Couldn’t load this location (' +
        esc(e.message) + ").</div>";
    });
})();

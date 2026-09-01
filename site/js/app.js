/* Hazard Watch — map page.
   Satellite Earth (Esri World Imagery, keyless with attribution) + live
   status markers from the nightly-updated JSON in this repository. */

(function () {
  "use strict";

  var DATA_ROOT = "../serve/out_live";
  var STATUS_ORDER = { red: 0, yellow: 1, unknown: 2, green: 3 };
  var STATUS_LABEL = { red: "Alert", yellow: "Watch", green: "Normal", unknown: "No data" };

  // ---- map ------------------------------------------------------------
  var map = L.map("map", {
    zoomControl: true,
    worldCopyJump: true,
    minZoom: 2,
    attributionControl: true
  }).setView([16, 12], 2.4);

  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 17,
      attribution: "Imagery © Esri — Source: Esri, Maxar, Earthstar Geographics"
    }
  ).addTo(map);

  // subtle place labels over the imagery
  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 17, opacity: 0.9, attribution: "Labels © Esri" }
  ).addTo(map);

  map.zoomControl.setPosition("bottomright");

  // clicking empty map: honest "not tracked yet" note
  map.on("click", function (e) {
    var la = e.latlng.lat.toFixed(3);
    var lo = e.latlng.lng.toFixed(3);
    L.popup({ closeButton: true })
      .setLatLng(e.latlng)
      .setContent(
        '<div class="pop"><div class="pop__name">Not tracked yet</div>' +
        '<div class="pop__msg">This spot (' + la + ", " + lo + ") isn’t on the " +
        "watch list. Every monitored point is registered once, then updated " +
        "daily. Ask for it via GitHub, or run<br><code>register.py add " +
        la + " " + lo + "</code></div></div>"
      )
      .openOn(map);
  });

  // ---- data -----------------------------------------------------------
  function fetchJSON(url) {
    return fetch(url, { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(url + " → " + r.status);
      return r.json();
    });
  }

  function markerIcon(status) {
    return L.divIcon({
      className: "",
      iconSize: [18, 18],
      iconAnchor: [9, 9],
      html:
        '<div class="marker marker--' + status + '">' +
        '<div class="marker__pulse"></div><div class="marker__core"></div></div>'
    });
  }

  function popupHTML(d) {
    var w = d.weather || {};
    return (
      '<div class="pop">' +
      '<div class="pop__status"><span class="dot dot--' + d.status + '"></span>' +
      '<span class="pop__name">' + d.name + "</span></div>" +
      '<div class="pop__msg">' + d.message + "</div>" +
      '<a class="btn" href="location.html?id=' + d.location_id + '">Full report →</a>' +
      "</div>"
    );
  }

  function peopleFmt(n) {
    if (n == null) return "";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M people";
    if (n >= 1e3) return Math.round(n / 1e3) + "k people";
    return n + " people";
  }

  var detailCache = {};

  fetchJSON(DATA_ROOT + "/index.json")
    .then(function (index) {
      var asof = document.getElementById("asof");
      asof.textContent = "data through " + (index.locations[0] && index.locations[0].as_of || "—");

      // fetch every location detail (small files, parallel)
      return Promise.all(
        index.locations.map(function (row) {
          return fetchJSON(DATA_ROOT + "/f/" + row.location_id + ".json")
            .then(function (d) { detailCache[row.location_id] = d; return d; })
            .catch(function () { return null; });
        })
      );
    })
    .then(function (details) {
      var ds = details.filter(Boolean);
      ds.sort(function (a, b) {
        var s = STATUS_ORDER[a.status] - STATUS_ORDER[b.status];
        return s !== 0 ? s : (b.static.people_10km || 0) - (a.static.people_10km || 0);
      });

      // summary chips
      var counts = { red: 0, yellow: 0, green: 0, unknown: 0 };
      ds.forEach(function (d) { counts[d.status] = (counts[d.status] || 0) + 1; });
      document.getElementById("strip").innerHTML = ["red", "yellow", "green", "unknown"]
        .filter(function (k) { return counts[k]; })
        .map(function (k) {
          return '<span class="chip"><span class="dot dot--' + k + '"></span><b>' +
            counts[k] + "</b>&nbsp;" + STATUS_LABEL[k] + "</span>";
        }).join("");

      // rail list + markers
      var list = document.getElementById("list");
      list.innerHTML = "";
      ds.forEach(function (d) {
        var m = L.marker([d.lat, d.lon], {
          icon: markerIcon(d.status),
          title: d.name + " — " + STATUS_LABEL[d.status],
          alt: d.name
        }).addTo(map).bindPopup(popupHTML(d), { maxWidth: 280 });

        var row = document.createElement("button");
        row.className = "locrow";
        row.innerHTML =
          '<span class="dot dot--' + d.status + '"></span>' +
          '<span><span class="locrow__name">' + d.name + "</span><br>" +
          '<span class="locrow__sub">' + STATUS_LABEL[d.status] +
          (d.as_of ? " · as of " + d.as_of : "") + "</span></span>" +
          '<span class="locrow__people">' + peopleFmt(d.static.people_10km) + "</span>";
        row.addEventListener("click", function () {
          map.flyTo([d.lat, d.lon], Math.max(map.getZoom(), 7), { duration: 0.9 });
          m.openPopup();
        });
        list.appendChild(row);
      });
    })
    .catch(function (err) {
      document.getElementById("list").innerHTML =
        '<div class="skeleton">Couldn’t load live data (' + err.message +
        "). If you’re running locally, serve the repository root, e.g. " +
        "<code>python -m http.server</code>, and open /site/.</div>";
    });
})();

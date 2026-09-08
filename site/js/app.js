/* Hazard Watch — map page.
   Satellite Earth (Esri World Imagery, keyless with attribution) + live
   status markers. Renders from ONE fetch: index.json carries everything the
   first paint needs (per-location details load only on the report page). */

(function () {
  "use strict";

  var H = window.HW;
  var DATA_ROOT = "../serve/out_live";

  // ---- map (optional: the rail still works if Leaflet failed to load) ---
  var hasMap = typeof L !== "undefined";
  var map = null;
  var rows = [];                     // loaded locations, for nearest-place

  if (hasMap) {
    map = L.map("map", {
      zoomControl: true,
      minZoom: 1,
      zoomSnap: 0.25,
      maxBounds: [[-85, -180], [85, 180]],
      maxBoundsViscosity: 0.7,
      attributionControl: true
    }).setView([16, 12], 2);

    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        maxZoom: 17, noWrap: true, bounds: [[-85, -180], [85, 180]],
        attribution: "Imagery © Esri — Source: Esri, Maxar, Earthstar Geographics"
      }
    ).addTo(map);

    // subtle place labels over the imagery
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 17, noWrap: true, bounds: [[-85, -180], [85, 180]],
        opacity: 0.9, attribution: "Labels © Esri" }
    ).addTo(map);

    map.zoomControl.setPosition("bottomright");
    window.__hwMap = map;            // console access for field debugging

    // clicking empty map: honest "not tracked yet", with a real way forward
    map.on("click", function (e) {
      var la = e.latlng.lat.toFixed(3);
      var lo = e.latlng.lng.toFixed(3);
      var near = "";
      if (rows.length) {
        var best = null, bd = Infinity;
        rows.forEach(function (r) {
          var dy = r.lat - e.latlng.lat;
          var dx = (r.lon - e.latlng.lng) *
                   Math.cos(e.latlng.lat * Math.PI / 180);
          var d2 = dy * dy + dx * dx;
          if (d2 < bd) { bd = d2; best = r; }
        });
        if (best) {
          near = '<div class="pop__near">Nearest watched place: ' +
            '<a href="location.html?id=' + H.esc(best.location_id) + '">' +
            H.esc(best.name) + "</a> (~" +
            Math.round(Math.sqrt(bd) * 111) + " km)</div>";
        }
      }
      L.popup({ closeButton: true })
        .setLatLng(e.latlng)
        .setContent(
          '<div class="pop"><div class="pop__name">We don’t watch this spot yet</div>' +
          '<div class="pop__msg">Every place on the list is checked daily ' +
          "for landslide, fire, drought and river conditions. Want this one " +
          "added?</div>" + near +
          '<a class="btn" target="_blank" rel="noopener" ' +
          'href="https://github.com/lkruer/hazard-watch/issues/new?title=' +
          encodeURIComponent("Please add " + la + ", " + lo) +
          '">Ask us to add it</a></div>'
        )
        .openOn(map);
    });
  } else {
    document.getElementById("map").innerHTML =
      '<div class="skeleton">The map couldn’t load — the list on this ' +
      "page still works.</div>";
  }

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
      iconSize: [44, 44],           // invisible 44px hit area (tap target)
      iconAnchor: [22, 22],
      html:
        '<div class="marker marker--' + status + '">' +
        '<div class="marker__pulse"></div><div class="marker__core"></div></div>'
    });
  }

  function popupHTML(d) {
    return (
      '<div class="pop">' +
      '<div class="pop__status"><span class="dot dot--' + d.status + '"></span>' +
      '<span class="pop__name">' + H.esc(d.name) + "</span></div>" +
      '<div class="pop__word pop__word--' + d.status + '">' +
      H.STATUS_LABEL[d.status] + "</div>" +
      '<div class="pop__msg">' + H.esc(d.message || "") + "</div>" +
      '<div class="pop__meta">' +
      (d.as_of ? "updated " + H.fmtDate(d.as_of) : "") +
      (d.people_10km ? " · " + H.peopleFmt(d.people_10km) + " nearby" : "") +
      "</div>" +
      '<a class="btn" href="location.html?id=' + H.esc(d.location_id) +
      '">Full report →</a></div>'
    );
  }

  function render(ds) {
    rows = ds;
    ds.sort(function (a, b) {
      var s = H.STATUS_ORDER[a.status] - H.STATUS_ORDER[b.status];
      return s !== 0 ? s : (b.people_10km || 0) - (a.people_10km || 0);
    });

    // freshest date across the board
    var asof = document.getElementById("asof");
    var newest = ds.reduce(function (m, d) {
      return d.as_of > m ? d.as_of : m;
    }, "");
    asof.textContent = newest ? "updated " + H.fmtDate(newest) : "";

    // summary chips — all four statuses, always (zero counts included)
    var counts = { red: 0, yellow: 0, green: 0, unknown: 0 };
    ds.forEach(function (d) { counts[d.status] = (counts[d.status] || 0) + 1; });
    document.getElementById("strip").innerHTML =
      ["red", "yellow", "green", "unknown"].map(function (k) {
        return '<span class="chip"><span class="dot dot--' + k + '"></span><b>' +
          counts[k] + "</b>&nbsp;" + H.STATUS_LABEL[k] + "</span>";
      }).join("");

    // rail list + markers
    var list = document.getElementById("list");
    list.innerHTML = "";
    var bounds = [];
    ds.forEach(function (d) {
      var m = null;
      if (hasMap && d.lat != null) {
        var so = H.STATUS_ORDER[d.status];
        m = L.marker([d.lat, d.lon], {
          icon: markerIcon(d.status),
          title: d.name + " — " + H.STATUS_LABEL[d.status],
          // severity controls stacking: red above yellow above unknown/green
          zIndexOffset: (3 - (so == null ? 3 : so)) * 1000
        }).addTo(map).bindPopup(popupHTML(d), { maxWidth: 300 });
        var el = m.getElement();
        if (el) {
          el.setAttribute("role", "button");
          el.setAttribute("aria-label",
            d.name + " — " + H.STATUS_LABEL[d.status]);
        }
        bounds.push([d.lat, d.lon]);
      }

      var row = document.createElement("button");
      row.className = "locrow";
      row.innerHTML =
        '<span class="dot dot--' + d.status + '"></span>' +
        '<span><span class="locrow__name">' + H.esc(d.name) + "</span><br>" +
        '<span class="locrow__sub">' + H.STATUS_LABEL[d.status] +
        (d.as_of ? " · " + H.fmtDate(d.as_of) : "") + "</span></span>" +
        '<span class="locrow__people">' + H.peopleFmt(d.people_10km) + "</span>";
      row.addEventListener("click", function () {
        if (m) {
          map.flyTo([d.lat, d.lon], Math.max(map.getZoom(), 7), { duration: 0.9 });
          m.openPopup();
        } else {
          location.href = "location.html?id=" + d.location_id;
        }
      });
      list.appendChild(row);
    });

    // frame the world the markers actually live in
    if (hasMap && bounds.length) {
      var fit = function () {
        var mobile = matchMedia("(max-width: 720px)").matches;
        map.fitBounds(L.latLngBounds(bounds), mobile
          ? { paddingTopLeft: [16, 60],
              paddingBottomRight: [16, Math.round(innerHeight * 0.42) + 16],
              maxZoom: 5 }
          : { paddingTopLeft: [380, 70], paddingBottomRight: [16, 16],
              maxZoom: 5 });
      };
      // a map launched in a hidden tab / background PWA measures 0×0 and
      // fitBounds degenerates — wait for the container to actually have size
      var framed = false;
      var tryFit = function () {
        if (framed) return;
        map.invalidateSize();
        if (map.getSize().x > 0) { framed = true; fit(); }
      };
      tryFit();
      if (!framed) {
        if (typeof ResizeObserver !== "undefined") {
          var ro = new ResizeObserver(function () {
            tryFit();
            if (framed) ro.disconnect();
          });
          ro.observe(document.getElementById("map"));
        }
        document.addEventListener("visibilitychange", tryFit);
        var iv = setInterval(function () {
          tryFit();
          if (framed) clearInterval(iv);
        }, 500);
        setTimeout(function () { clearInterval(iv); }, 120000);
      }
    }
  }

  function failMessage(err) {
    var msg = navigator.onLine === false
      ? "You’re offline. We couldn’t load the latest status — " +
        "reconnect and try again."
      : "Couldn’t load the live data. Please try again in a moment.";
    if (location.protocol === "file:" || location.hostname === "localhost" ||
        location.hostname === "127.0.0.1" || location.hostname === "") {
      msg += " (Running locally? Serve the repository root: " +
        "<code>python -m http.server</code>, then open /site/.)";
    }
    if (window.console && console.error) console.error(err);
    document.getElementById("list").innerHTML =
      '<div class="skeleton">' + msg +
      '<br><button class="btn retry" id="retry">Try again</button></div>';
    var b = document.getElementById("retry");
    if (b) b.addEventListener("click", function () { location.reload(); });
  }

  fetchJSON(DATA_ROOT + "/index.json")
    .then(function (index) {
      var list = index.locations || [];
      if (list.length && list[0].lat != null) return list;  // enriched index
      // older cached index shape: fall back to fetching details
      return Promise.all(list.map(function (row) {
        return fetchJSON(DATA_ROOT + "/f/" + row.location_id + ".json")
          .then(function (d) {
            return { location_id: d.location_id, name: d.name,
                     status: d.status, as_of: d.as_of, lat: d.lat, lon: d.lon,
                     message: d.message,
                     people_10km: (d.static || {}).people_10km };
          })
          .catch(function () { return null; });
      })).then(function (ds) { return ds.filter(Boolean); });
    })
    .then(render)
    .catch(failMessage);

  // mobile bottom-sheet collapse
  var handle = document.getElementById("railtoggle");
  if (handle) {
    handle.addEventListener("click", function () {
      var collapsed = document.body.classList.toggle("rail-collapsed");
      handle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    });
  }
})();

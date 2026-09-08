/* Hazard Watch service worker — offline last-known-status (PWA).
   - data JSON: network-first, cached fallback (offline shows last known)
   - app shell (incl. vendored Leaflet): cache-first + background refresh
   - fonts/tiles: runtime cache-first with a small LRU cap
   Bump VERSION whenever site files change: the byte change triggers a SW
   update, install re-caches the shell, activate deletes the old caches. */

var VERSION = "v2";
var SHELL = "hw-shell-" + VERSION;
var DATA = "hw-data-v1";           // stable: never wipe last-known status
var RUNTIME = "hw-runtime-v1";
var RUNTIME_MAX = 60;

var SHELL_FILES = [
  "index.html", "location.html", "about.html",
  "css/styles.css", "js/labels.js", "js/app.js", "js/location.js",
  "vendor/leaflet.css", "vendor/leaflet.js",
  "icons/icon-192.png", "icons/icon-512.png", "icons/icon-maskable-512.png",
  "manifest.webmanifest"
];

var RUNTIME_HOSTS = [
  "fonts.googleapis.com", "fonts.gstatic.com", "server.arcgisonline.com"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(SHELL).then(function (c) { return c.addAll(SHELL_FILES); })
      .then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) {
        return k !== SHELL && k !== DATA && k !== RUNTIME;
      }).map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

function trimRuntime() {
  caches.open(RUNTIME).then(function (c) {
    c.keys().then(function (keys) {
      if (keys.length > RUNTIME_MAX) {
        c.delete(keys[0]).then(trimRuntime);
      }
    });
  });
}

self.addEventListener("fetch", function (e) {
  if (e.request.method !== "GET") return;
  var url = new URL(e.request.url);

  // live data: network first, cache fallback (offline shows last known)
  if (url.pathname.indexOf("/serve/out_live/") !== -1) {
    e.respondWith(
      fetch(e.request).then(function (res) {
        var copy = res.clone();
        caches.open(DATA).then(function (c) { c.put(e.request, copy); });
        return res;
      }).catch(function () { return caches.match(e.request); })
    );
    return;
  }

  // fonts + map tiles: cache-first with LRU cap. Opaque responses (no-cors
  // tiles, the fonts stylesheet) are cacheable but have ok === false.
  if (RUNTIME_HOSTS.indexOf(url.hostname) !== -1) {
    e.respondWith(
      caches.match(e.request).then(function (hit) {
        return hit || fetch(e.request).then(function (res) {
          if (res.ok || res.type === "opaque") {
            var copy = res.clone();
            caches.open(RUNTIME).then(function (c) {
              c.put(e.request, copy);
              trimRuntime();
            });
          }
          return res;
        });
      })
    );
    return;
  }

  // shell: cache-first (ignore ?id= etc. on navigations so a cached
  // location.html serves every report offline) + background refresh
  var nav = e.request.mode === "navigate" &&
            url.origin === self.location.origin;
  e.respondWith(
    caches.match(e.request, nav ? { ignoreSearch: true } : undefined)
      .then(function (hit) {
        var net = fetch(e.request).then(function (res) {
          if (res.ok && url.origin === self.location.origin) {
            var copy = res.clone();
            caches.open(SHELL).then(function (c) { c.put(e.request, copy); });
          }
          return res;
        });
        if (hit) { net.catch(function () {}); return hit; }
        return net;
      })
  );
});

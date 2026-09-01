/* Hazard Watch service worker — offline-last-known-status (PWA-lite).
   Strategy: network-first for data JSON (freshness matters), falling back to
   the last cached copy when offline; cache-first for the app shell. */

var SHELL = "hw-shell-v1";
var DATA = "hw-data-v1";
var SHELL_FILES = [
  "index.html", "location.html", "about.html",
  "css/styles.css", "js/app.js", "js/location.js", "manifest.webmanifest"
];

self.addEventListener("install", function (e) {
  e.waitUntil(
    caches.open(SHELL).then(function (c) { return c.addAll(SHELL_FILES); })
      .then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (e) {
  e.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", function (e) {
  var url = new URL(e.request.url);
  if (e.request.method !== "GET") return;

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

  // shell + fonts + tiles: cache first, then network
  e.respondWith(
    caches.match(e.request).then(function (hit) {
      return hit || fetch(e.request);
    })
  );
});

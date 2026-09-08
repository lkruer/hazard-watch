/* Hazard Watch — shared vocabulary + tiny helpers.
   One place for status words so the map and the report can never drift. */

window.HW = (function () {
  "use strict";

  var STATUS_LABEL = {
    red: "Alert",
    yellow: "Watch",
    green: "All normal",
    unknown: "Data uncertain"
  };
  var STATUS_ORDER = { red: 0, yellow: 1, unknown: 2, green: 3 };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;",
               '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  /* "today" / "yesterday" / "5 Sep" — ISO dates are for machines */
  function fmtDate(iso) {
    if (!iso) return "";
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
    if (!m) return iso;
    var then = new Date(+m[1], +m[2] - 1, +m[3]);
    var now = new Date();
    var today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    var days = Math.round((today - then) / 864e5);
    if (days === 0) return "today";
    if (days === 1) return "yesterday";
    var s = then.getDate() + " " + MONTHS[then.getMonth()];
    if (then.getFullYear() !== now.getFullYear()) s += " " + then.getFullYear();
    return s;
  }

  function peopleFmt(n) {
    if (n == null) return "";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M people";
    if (n >= 1e3) return Math.round(n / 1e3) + "k people";
    return n + " people";
  }

  function pct(v) {
    return (v == null || !isFinite(v)) ? "—" : Math.round(v * 100) + "%";
  }

  return { STATUS_LABEL: STATUS_LABEL, STATUS_ORDER: STATUS_ORDER,
           esc: esc, fmtDate: fmtDate, peopleFmt: peopleFmt, pct: pct };
})();

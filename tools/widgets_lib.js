/* Shared helpers for explainer widgets. Load BEFORE the topic's widgets.js
   (build_spec.py passes [widgets_lib.js, widgets.js] as the spec's scripts).
   Exposes window.WLib = { $, cssVar, mulberry32, gauss, onTheme }. */
window.WLib = (function () {
  'use strict';
  function $(id) { return document.getElementById(id); }
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function mulberry32(a) {           // seeded PRNG for repeatable scrubbing
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {              // Box-Muller
    var u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  var redraws = [];
  function onTheme(fn) { redraws.push(fn); }   // canvases re-render on theme flip
  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  if (mq.addEventListener) mq.addEventListener('change', function () {
    redraws.forEach(function (f) { f(); });
  });
  return { $: $, cssVar: cssVar, mulberry32: mulberry32, gauss: gauss, onTheme: onTheme };
})();

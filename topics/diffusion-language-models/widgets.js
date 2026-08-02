(function () {
  'use strict';
  var $ = WLib.$, cssVar = WLib.cssVar, mulberry32 = WLib.mulberry32,
      gauss = WLib.gauss, onTheme = WLib.onTheme;

  /* ---- W1: image forward-noising scrubber ---------------------------- */
  if ($('w-noise')) {
    var nCv = $('w-noise-cv'), nCtx = nCv.getContext('2d');
    var NG = 26, cell = 10;
    nCv.width = NG * cell; nCv.height = NG * cell;
    var base = new Float32Array(NG * NG);
    (function () {
      function ear(x, y, ex) { var dy = y - 5; return dy >= 0 && dy <= 5 && Math.abs(x - ex) <= dy * 0.8; }
      function eye(x, y, ex) { return (x - ex) * (x - ex) + (y - 12) * (y - 12) <= 1.4; }
      for (var y = 0; y < NG; y++) for (var x = 0; x < NG; x++) {
        var cx = (x - 12.5) / 9, cy = (y - 14) / 8.5, v = 0.92;
        if (cx * cx + cy * cy < 1) v = 0.35;
        if (ear(x, y, 6) || ear(x, y, 19)) v = 0.35;
        if (eye(x, y, 9) || eye(x, y, 16)) v = 0.05;
        if (Math.abs(x - 12.5) <= 1 && y >= 15 && y <= 16) v = 0.08;
        if (y === 18 && x >= 10 && x <= 15) v = 0.12;
        base[y * NG + x] = v;
      }
    })();
    var nRng = mulberry32(7);
    var nEps = new Float32Array(NG * NG);
    for (var ni = 0; ni < nEps.length; ni++) nEps[ni] = gauss(nRng);
    var nSlider = $('w-noise-t'), nStat = $('w-noise-stat');
    var drawNoise = function () {
      var t = nSlider.value / 1000;
      var ab = Math.pow(Math.cos(Math.PI * t / 2), 2);
      var sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);
      for (var i = 0; i < base.length; i++) {
        var v = sa * (base[i] * 2 - 1) + sn * nEps[i] * 0.85;
        var g = Math.max(0, Math.min(255, Math.round((v + 1) * 127.5)));
        nCtx.fillStyle = 'rgb(' + g + ',' + g + ',' + g + ')';
        nCtx.fillRect((i % NG) * cell, Math.floor(i / NG) * cell, cell, cell);
      }
      nStat.textContent = 't = ' + t.toFixed(2) + '   ᾱ = ' + ab.toFixed(3);
    };
    var nTimer = null;
    var nAnim = function (dir) {
      clearInterval(nTimer);
      nTimer = setInterval(function () {
        var v = +nSlider.value + dir * 12;
        if (v >= 1000) { v = 1000; clearInterval(nTimer); }
        if (v <= 0) { v = 0; clearInterval(nTimer); }
        nSlider.value = v; drawNoise();
      }, 30);
    };
    $('w-noise-play').addEventListener('click', function () { nAnim(1); });
    $('w-noise-rev').addEventListener('click', function () { nAnim(-1); });
    nSlider.addEventListener('input', function () { clearInterval(nTimer); drawNoise(); });
    drawNoise();
  }

  /* ---- W3: text masking scrubber ------------------------------------- */
  if ($('w-mask')) {
    var mWords = ['the', 'little', 'robot', 'painted', 'a', 'bright', 'mural', 'on', 'the', 'station', 'wall', 'today'];
    var mRng = mulberry32(11);
    var mTh = mWords.map(function () { return mRng(); });
    var mRow = $('w-mask-row'), mSlider = $('w-mask-t'), mStat = $('w-mask-stat');
    var mBoxes = mWords.map(function (w) {
      var d = document.createElement('div');
      d.className = 'box ok'; d.textContent = w; mRow.appendChild(d); return d;
    });
    var drawMask = function () {
      var t = mSlider.value / 1000, masked = 0;
      mBoxes.forEach(function (b, i) {
        if (mTh[i] < t) { b.className = 'box dim'; b.textContent = '▢'; masked++; }
        else { b.className = 'box ok'; b.textContent = mWords[i]; }
      });
      mStat.textContent = 'α = ' + (1 - t).toFixed(2) +
        '   expected masked = ' + (t * mWords.length).toFixed(1) + '   actual = ' + masked;
    };
    mSlider.addEventListener('input', drawMask);
    drawMask();
  }

  /* ---- W2: typewriter vs editor race --------------------------------- */
  if ($('w-race')) {
    var rWords = 'the quick brown fox jumps over the lazy dog while the cat naps on the mat'.split(' ');
    var rOrder = [[3, 8, 12, 15], [1, 2, 4, 11], [0, 5, 6, 13], [7, 9, 10, 14]];
    var rArRow = $('w-race-ar'), rDlRow = $('w-race-dl');
    var rArB, rDlB, rTick, rTimer = null;
    var rBuild = function (row) {
      row.innerHTML = '';
      return rWords.map(function () {
        var d = document.createElement('div');
        d.className = 'box dim'; d.textContent = '▢'; row.appendChild(d); return d;
      });
    };
    var rReset = function () {
      clearInterval(rTimer); rTimer = null;
      rArB = rBuild(rArRow); rDlB = rBuild(rDlRow); rTick = 0;
      $('w-race-ar-stat').textContent = 'passes: 0';
      $('w-race-dl-stat').textContent = 'passes: 0';
    };
    var rStep = function () {
      rTick++;
      if (rTick <= rWords.length) {
        rArB.forEach(function (b) { if (b.className === 'box accent') b.className = 'box ok'; });
        rArB[rTick - 1].className = 'box accent';
        rArB[rTick - 1].textContent = rWords[rTick - 1];
        $('w-race-ar-stat').textContent = 'passes: ' + rTick + (rTick === rWords.length ? ' — done' : '');
      }
      if (rTick <= rOrder.length) {
        rDlB.forEach(function (b) { if (b.className === 'box accent') b.className = 'box ok'; });
        rOrder[rTick - 1].forEach(function (i) {
          rDlB[i].className = 'box accent'; rDlB[i].textContent = rWords[i];
        });
        $('w-race-dl-stat').textContent = 'passes: ' + rTick + (rTick === rOrder.length ? ' — done ✓ (waiting for the typewriter…)' : '');
      } else {
        rDlB.forEach(function (b) { if (b.className === 'box accent') b.className = 'box ok'; });
      }
      if (rTick >= rWords.length) {
        rArB.forEach(function (b) { if (b.className === 'box accent') b.className = 'box ok'; });
        clearInterval(rTimer); rTimer = null;
      }
    };
    $('w-race-play').addEventListener('click', function () {
      if (rTimer) return;
      if (rTick >= rWords.length) rReset();
      rTimer = setInterval(rStep, 420);
    });
    $('w-race-reset').addEventListener('click', rReset);
    rReset();
  }

  /* ---- W7: independence-error Monte Carlo ----------------------------- */
  if ($('w-indep')) {
    var iMode = 'par', iN = 0, iGood = 0;
    var iChips = $('w-indep-chips'), iStat = $('w-indep-stat');
    var iParB = $('w-indep-par'), iSeqB = $('w-indep-seq');
    var iSetMode = function (m) {
      iMode = m;
      iParB.className = 'wbtn' + (m === 'par' ? ' active' : '');
      iSeqB.className = 'wbtn' + (m === 'seq' ? ' active' : '');
    };
    iParB.addEventListener('click', function () { iSetMode('par'); });
    iSeqB.addEventListener('click', function () { iSetMode('seq'); });
    var iDraw = function () {
      var first, second;
      if (iMode === 'par') {
        first = Math.random() < 0.5 ? 'hot' : 'ice';
        second = Math.random() < 0.5 ? 'dog' : 'cream';
      } else {
        first = Math.random() < 0.5 ? 'hot' : 'ice';
        second = first === 'hot' ? 'dog' : 'cream';
      }
      var ok = (first === 'hot' && second === 'dog') || (first === 'ice' && second === 'cream');
      iN++; if (ok) iGood++;
      var c = document.createElement('span');
      c.className = 'wchip ' + (ok ? 'good' : 'bad');
      c.textContent = first + ' ' + second;
      iChips.prepend(c);
      while (iChips.children.length > 120) iChips.lastChild.remove();
      iStat.textContent = 'coherent: ' + iGood + '/' + iN + ' (' + (100 * iGood / iN).toFixed(0) + '%)' +
        (iMode === 'par' ? ' — 1 pass per sample' : ' — 2 passes per sample');
      if (window.__indepPredCheck) window.__indepPredCheck(iMode, iN, iGood);
    };
    $('w-indep-1').addEventListener('click', iDraw);
    $('w-indep-50').addEventListener('click', function () { for (var k = 0; k < 50; k++) iDraw(); });
    $('w-indep-reset').addEventListener('click', function () {
      iN = 0; iGood = 0; iChips.innerHTML = ''; iStat.textContent = 'no samples yet';
    });
  }

  /* ---- W4: DDIM sampling lab (exact score on a 2-D mixture) ----------- */
  if ($('w-ddim')) {
    var dCv = $('w-ddim-cv'), dCtx = dCv.getContext('2d');
    var comps = [
      { x: -0.42, y: 0.38, s: 0.07, w: 0.14 },
      { x: 0.42, y: 0.38, s: 0.07, w: 0.14 }
    ];
    for (var ci = 0; ci < 6; ci++) {
      var an = (205 + ci * 26) * Math.PI / 180;
      comps.push({ x: 0.62 * Math.cos(an), y: 0.12 + 0.62 * Math.sin(an), s: 0.06, w: 0.72 / 6 });
    }
    // exact eps-hat: q_t is a Gaussian mixture with means sqrt(ab)*mu, var ab*s^2+1-ab
    var scoreEps = function (px, py, ab) {
      var sa = Math.sqrt(ab), nb = 1 - ab;
      var Z = 0, gx = 0, gy = 0;
      for (var k = 0; k < comps.length; k++) {
        var c = comps[k], v = ab * c.s * c.s + nb;
        var dx = px - sa * c.x, dy = py - sa * c.y;
        var g = c.w / v * Math.exp(-(dx * dx + dy * dy) / (2 * v));
        Z += g; gx -= g * dx / v; gy -= g * dy / v;
      }
      var sn = Math.sqrt(nb);
      return [-sn * gx / (Z + 1e-12), -sn * gy / (Z + 1e-12)];
    };
    var NP = 420, dStart = [], dPts = [];
    var dSampleNoise = function (seed) {
      var rng = mulberry32(seed);
      dStart = [];
      for (var i = 0; i < NP; i++) dStart.push([gauss(rng), gauss(rng)]);
      dPts = dStart.map(function (p) { return p.slice(); });
    };
    var tRng = mulberry32(4), dTarget = [];
    for (var ti = 0; ti < NP; ti++) {
      var r = tRng(), cc = comps[comps.length - 1];
      for (var k2 = 0; k2 < comps.length; k2++) {
        if (r < comps[k2].w) { cc = comps[k2]; break; }
        r -= comps[k2].w;
      }
      dTarget.push([cc.x + gauss(tRng) * cc.s, cc.y + gauss(tRng) * cc.s]);
    }
    var alphaBar = function (t) { return Math.pow(Math.cos(Math.PI * t / 2), 2); };
    var dS = 4, dRunning = false;
    var ddimStep = function (p, ab, abN) {
      var e = scoreEps(p[0], p[1], ab);
      var sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);
      var x0x = (p[0] - sn * e[0]) / sa, x0y = (p[1] - sn * e[1]) / sa;
      var saN = Math.sqrt(abN), snN = Math.sqrt(1 - abN);
      return [saN * x0x + snN * e[0], saN * x0y + snN * e[1]];
    };
    var toPx = function (p) {
      return [dCv.width / 2 + p[0] * dCv.width / 2.6, dCv.height / 2 - p[1] * dCv.height / 2.6];
    };
    var dDraw = function (note) {
      var acc = cssVar('--accent') || '#2458c5';
      var mut = cssVar('--line') || '#ccc';
      var bgc = cssVar('--bg') || '#fff';
      dCtx.fillStyle = bgc; dCtx.fillRect(0, 0, dCv.width, dCv.height);
      dCtx.fillStyle = mut;
      dTarget.forEach(function (p) { var q = toPx(p); dCtx.fillRect(q[0] - 1.2, q[1] - 1.2, 2.4, 2.4); });
      dCtx.fillStyle = acc;
      dPts.forEach(function (p) {
        var q = toPx(p);
        dCtx.beginPath(); dCtx.arc(q[0], q[1], 2.1, 0, 7); dCtx.fill();
      });
      $('w-ddim-stat').textContent = note;
    };
    var dRun = function () {
      if (dRunning) return;
      dRunning = true;
      dPts = dStart.map(function (p) { return p.slice(); });
      var ts = [];
      for (var i = 0; i <= dS; i++) ts.push(0.985 * (1 - i / dS) + 0.005);
      dDraw('S = ' + dS + ' — starting from pure noise');
      var i2 = 0;
      var tickMs = Math.max(90, 700 / dS);
      var stepFn = function () {
        if (i2 >= dS) {
          dDraw('S = ' + dS + ' — done (deterministic: rerunning gives the same picture)');
          dRunning = false;
          if (window.__ddimPredResolve) window.__ddimPredResolve(dS);
          return;
        }
        var ab = alphaBar(ts[i2]), abN = alphaBar(ts[i2 + 1]);
        dPts = dPts.map(function (p) { return ddimStep(p, ab, abN); });
        i2++;
        dDraw('S = ' + dS + ' — step ' + i2 + '/' + dS);
        setTimeout(stepFn, tickMs);
      };
      setTimeout(stepFn, 350);
    };
    var sBtns = document.querySelectorAll('#w-ddim [data-s]');
    sBtns.forEach(function (b) {
      b.addEventListener('click', function () {
        sBtns.forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        dS = +b.getAttribute('data-s');
      });
    });
    $('w-ddim-run').addEventListener('click', dRun);
    $('w-ddim-reset').addEventListener('click', function () {
      dSampleNoise((Math.random() * 1e9) | 0);
      dDraw('fresh noise sampled — press Run');
    });
    dSampleNoise(21);
    dDraw('press Run');
    onTheme(function () { dDraw($('w-ddim-stat').textContent); });
  }

  /* ---- W6: reveal-probability calculator ------------------------------ */
  if ($('w-reveal')) {
    var vAs = $('w-rev-as'), vAt = $('w-rev-at'), vCv = $('w-rev-cv'), vCtx = vCv.getContext('2d');
    var vDraw = function () {
      var as = +vAs.value / 100, at = +vAt.value / 100;
      if (at >= as) { at = Math.max(0, as - 0.01); vAt.value = Math.round(at * 100); }
      var reveal = (as - at) / ((1 - at) || 1e-9);
      var okC = cssVar('--ok') || 'green', dimC = cssVar('--line') || '#ccc';
      var bgc = cssVar('--box-bg') || '#eee', fgc = cssVar('--fg') || '#000';
      vCtx.fillStyle = bgc; vCtx.fillRect(0, 0, vCv.width, vCv.height);
      var bw = vCv.width - 20;
      vCtx.fillStyle = okC; vCtx.fillRect(10, 12, bw * reveal, 28);
      vCtx.fillStyle = dimC; vCtx.fillRect(10 + bw * reveal, 12, bw * (1 - reveal), 28);
      vCtx.fillStyle = fgc; vCtx.font = '13px ui-monospace, Menlo, monospace'; vCtx.textAlign = 'center';
      vCtx.fillText('reveal ' + (100 * reveal).toFixed(1) + '%  ·  stay masked ' + (100 * (1 - reveal)).toFixed(1) + '%', vCv.width / 2, 58);
      $('w-rev-stat').textContent = 'αs = ' + as.toFixed(2) + '   αt = ' + at.toFixed(2) +
        '   P(reveal) = (' + as.toFixed(2) + ' − ' + at.toFixed(2) + ') / (1 − ' + at.toFixed(2) + ') = ' + reveal.toFixed(3);
    };
    vAs.addEventListener('input', vDraw);
    vAt.addEventListener('input', vDraw);
    vDraw();
    onTheme(vDraw);
  }

  /* ---- W5: schedule & NELBO-weight explorer --------------------------- */
  if ($('w-sched')) {
    var sCv = $('w-sched-cv'), sCtx = sCv.getContext('2d');
    var scheds = {
      linear: { a: function (t) { return 1 - t; }, da: function () { return -1; } },
      cosine: { a: function (t) { return Math.pow(Math.cos(Math.PI * t / 2), 2); }, da: function (t) { return -Math.PI / 2 * Math.sin(Math.PI * t); } },
      quad: { a: function (t) { return 1 - t * t; }, da: function (t) { return -2 * t; } }
    };
    var sCur = 'linear', WMAX = 6;
    var sDraw = function () {
      var line = cssVar('--line') || '#ccc', mut = cssVar('--muted') || '#777';
      var acc = cssVar('--accent') || '#2458c5', ok = cssVar('--ok') || 'green';
      var bgc = cssVar('--box-bg') || '#fff';
      var W = sCv.width, H = sCv.height, L = 40, B = H - 26, T = 12, R = W - 12;
      sCtx.fillStyle = bgc; sCtx.fillRect(0, 0, W, H);
      sCtx.strokeStyle = line; sCtx.lineWidth = 1;
      sCtx.beginPath(); sCtx.moveTo(L, T); sCtx.lineTo(L, B); sCtx.lineTo(R, B); sCtx.stroke();
      sCtx.fillStyle = mut; sCtx.font = '12px ui-monospace, Menlo, monospace'; sCtx.textAlign = 'center';
      sCtx.fillText('t = 0', L + 16, B + 17); sCtx.fillText('t = 1', R - 16, B + 17);
      sCtx.textAlign = 'right';
      sCtx.fillText('1', L - 5, T + 9); sCtx.fillText('0', L - 5, B + 3);
      var s = scheds[sCur];
      var px = function (t) { return L + (R - L) * t; };
      sCtx.strokeStyle = acc; sCtx.lineWidth = 2.2; sCtx.beginPath();
      for (var i = 0; i <= 200; i++) {
        var t = i / 200, y = B - (B - T) * s.a(t);
        if (i) sCtx.lineTo(px(t), y); else sCtx.moveTo(px(t), y);
      }
      sCtx.stroke();
      sCtx.strokeStyle = ok; sCtx.beginPath();
      var started = false;
      for (var j = 1; j <= 200; j++) {
        var t2 = j / 200;
        var w = Math.min(WMAX, -s.da(t2) / Math.max(1e-6, 1 - s.a(t2)));
        var y2 = B - (B - T) * (w / WMAX);
        if (started) sCtx.lineTo(px(t2), y2); else sCtx.moveTo(px(t2), y2);
        started = true;
      }
      sCtx.stroke();
    };
    document.querySelectorAll('#w-sched [data-sched]').forEach(function (b) {
      b.addEventListener('click', function () {
        document.querySelectorAll('#w-sched [data-sched]').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        sCur = b.getAttribute('data-sched');
        sDraw();
      });
    });
    sDraw();
    onTheme(sDraw);
  }

  /* ---- predict-before-run: DDIM lab ----------------------------------- */
  if ($('w-ddim-predict')) {
    var pPicked = null, pResolved = false;
    var pFb = $('w-ddim-predfb');
    var pBtns = document.querySelectorAll('#w-ddim-predict [data-pred]');
    pBtns.forEach(function (b) {
      b.addEventListener('click', function () {
        if (pPicked) return;
        pPicked = b.getAttribute('data-pred');
        pBtns.forEach(function (x) { x.disabled = true; });
        b.classList.add('active');
        pFb.style.display = 'block';
        pFb.textContent = 'Prediction locked in. Now set S = 1 and press Run to find out.';
      });
    });
    window.__ddimPredResolve = function (S) {
      if (pResolved || !pPicked || S !== 1) return;
      pResolved = true;
      pFb.style.display = 'block';
      pFb.innerHTML = (pPicked === 'blob' ? '<strong>Your prediction was right.</strong> ' :
        '<strong>Not what you predicted.</strong> ') +
        'At S = 1 each particle jumps straight to a posterior average of the modes — a mush ' +
        'between them, not a face and not uniform noise. One step means one chance to commit, ' +
        'so the model hedges. More steps let it commit progressively.';
    };
  }

  /* ---- predict-before-run: independence demo -------------------------- */
  if ($('w-indep-predict')) {
    var qPicked = null, qResolved = false;
    var qFb = $('w-indep-predfb');
    var qBtns = document.querySelectorAll('#w-indep-predict [data-pred]');
    qBtns.forEach(function (b) {
      b.addEventListener('click', function () {
        if (qPicked) return;
        qPicked = b.getAttribute('data-pred');
        qBtns.forEach(function (x) { x.disabled = true; });
        b.classList.add('active');
        qFb.style.display = 'block';
        qFb.textContent = 'Prediction locked in. Draw at least 20 samples in one-pass mode to find out.';
      });
    });
    window.__indepPredCheck = function (mode, n, good) {
      if (qResolved || !qPicked || mode !== 'par' || n < 20) return;
      qResolved = true;
      qFb.style.display = 'block';
      qFb.innerHTML = (qPicked === '50' ? '<strong>Your prediction was right:</strong> ' :
        '<strong>Not what you predicted:</strong> ') +
        'the tally is at ' + (100 * good / n).toFixed(0) + '% and converges to 50% — two ' +
        'independent fair coins agree half the time (&frac14; “hot cream” + &frac14; “ice dog” wrong).';
    };
  }

  /* ---- concept map ----------------------------------------------------- */
  if ($('w-map')) {
    var NODES = [
      { id: 'ar', x: 95, y: 120, label: 'typewriter (AR)', sec: '#background', desc: 'One token per forward pass, left to right, write-once. Fully general via the chain rule — its limits are computational, not representational.' },
      { id: 'mem', x: 95, y: 240, label: 'bandwidth-bound', sec: '#background', desc: 'AR decode streams all weights + KV cache per token; the GPU is starved for data, not FLOPs. This is why N sequential passes hurt.' },
      { id: 'chain', x: 95, y: 360, label: 'chain rule', sec: '#background', desc: 'P(x) = ∏ P(x_t | x_<t). Any joint distribution factorizes this way; AR pays for it with sequential decoding.' },
      { id: 'img', x: 300, y: 80, label: 'image diffusion', sec: '#background', desc: 'Corrupt with Gaussian noise, learn to reverse. Coarse-to-fine, all pixels in parallel, steps tunable.' },
      { id: 'fwd', x: 300, y: 190, label: 'forward q(x_t|x_0)', sec: '#math-continuous', desc: '𝒩(√ᾱ·x₀, (1−ᾱ)·I): one-jump corruption to any noise level — the closed form that makes training cheap.' },
      { id: 'elbo', x: 300, y: 300, label: 'ELBO → L_simple', sec: '#math-continuous', desc: 'Per-step Gaussian KLs collapse to “guess the noise”: ‖ε − ε_θ(x_t, t)‖². One regression trains the whole hierarchy.' },
      { id: 'score', x: 300, y: 410, label: 'score ∇log q', sec: '#math-continuous', desc: 's_θ = −ε_θ/√(1−ᾱ): denoising is gradient ascent on log-density. The bridge to the SDE view.' },
      { id: 'ddim', x: 300, y: 520, label: 'DDIM', sec: '#math-continuous', desc: 'Non-Markovian family sharing the marginals; σ=0 is deterministic and skips steps. Same network, far fewer passes.' },
      { id: 'mask', x: 530, y: 90, label: 'masking = noise', sec: '#intuition', desc: 'For discrete tokens the corruption is hiding: Cat(α_t·x + (1−α_t)·m). Absorbing mask state.' },
      { id: 'rev', x: 530, y: 200, label: 'reveal posterior', sec: '#math-discrete', desc: 'Masked token revealed with prob (α_s−α_t)/(1−α_t); visible tokens carry over with prob 1 — the frozen-token theorem.' },
      { id: 'mdlm', x: 530, y: 310, label: 'MDLM objective', sec: '#math-discrete', desc: 'NELBO = ∫ w(t)·(BERT loss) dt with w = −α′/(1−α); schedule-invariant. Fill-in-the-blank is secretly an ELBO.' },
      { id: 'mf', x: 530, y: 420, label: 'mean-field error', sec: '#math-discrete', desc: 'Parallel commits sample independent marginals (“hot cream”). Fewer steps → more simultaneous commits → more error.' },
      { id: 'dial', x: 420, y: 530, label: 'step dial S', sec: '#intuition', desc: 'The shared currency of both worlds: steps buy quality (continuous: discretization; discrete: coordination).' },
      { id: 'block', x: 760, y: 110, label: 'Block Diffusion', sec: '#math-sota', desc: 'AR across blocks, diffusion within: variable length + KV caching restored.' },
      { id: 'remdm', x: 760, y: 210, label: 'ReMDM', sec: '#math-sota', desc: 'Marginal-preserving remasking sampler: committed tokens can be revised. The editor, no retraining.' },
      { id: 'guide', x: 760, y: 310, label: 'guidance', sec: '#math-sota', desc: 'γ-sharpened conditional/unconditional ratio per position: steer sentiment, style, molecules at sampling time.' },
      { id: 'd1', x: 760, y: 410, label: 'd1 reasoning', sec: '#math-sota', desc: 'SFT + diffu-GRPO (one-step masked log-prob estimates): the reasoning playbook running on diffusion.' },
      { id: 'mercury', x: 760, y: 520, label: 'Mercury', sec: '#papers', desc: 'Inception\\u2019s commercial dLLM family: all of the above stacked, claiming ~5\\u00d7 AR speed.' }
    ];
    var EDGES = [
      ['ar', 'mem'], ['ar', 'chain'], ['ar', 'img'],
      ['img', 'fwd'], ['fwd', 'elbo'], ['elbo', 'score'], ['score', 'ddim'],
      ['ddim', 'dial'], ['img', 'mask'], ['mask', 'rev'], ['rev', 'mdlm'],
      ['mdlm', 'mf'], ['mf', 'dial'],
      ['mdlm', 'block'], ['rev', 'remdm'], ['mdlm', 'guide'], ['mdlm', 'd1'],
      ['block', 'mercury'], ['remdm', 'mercury'], ['guide', 'mercury'], ['d1', 'mercury']
    ];
    var HUBS = { mdlm: 1, dial: 1, mercury: 1 };
    var svg = $('w-map-svg'), info = $('w-map-info');
    var NS = 'http://www.w3.org/2000/svg';
    var byId = {};
    NODES.forEach(function (n) { byId[n.id] = n; });
    var edgeEls = [], nodeEls = {};
    EDGES.forEach(function (e) {
      var a = byId[e[0]], b = byId[e[1]];
      var ln = document.createElementNS(NS, 'line');
      ln.setAttribute('x1', a.x); ln.setAttribute('y1', a.y);
      ln.setAttribute('x2', b.x); ln.setAttribute('y2', b.y);
      ln.setAttribute('class', 'cmap-edge');
      ln.__ends = [e[0], e[1]];
      svg.appendChild(ln); edgeEls.push(ln);
    });
    NODES.forEach(function (n) {
      var g = document.createElementNS(NS, 'g');
      g.setAttribute('class', 'cmap-node' + (HUBS[n.id] ? ' hub' : ''));
      var wpx = n.label.length * 7.2 + 22;
      var rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', n.x - wpx / 2); rect.setAttribute('y', n.y - 15);
      rect.setAttribute('width', wpx); rect.setAttribute('height', 30);
      rect.setAttribute('rx', 8);
      var tx = document.createElementNS(NS, 'text');
      tx.setAttribute('x', n.x); tx.setAttribute('y', n.y + 4.5);
      tx.setAttribute('text-anchor', 'middle');
      tx.textContent = n.label;
      g.appendChild(rect); g.appendChild(tx);
      g.addEventListener('mouseenter', function () { highlight(n.id); });
      g.addEventListener('mouseleave', clearHl);
      g.addEventListener('click', function () {
        var target = (window.__PAGES__ && window.__PAGES__[n.sec.replace('#', '')]) || n.sec;
        info.innerHTML = '<strong>' + n.label + '.</strong> ' + n.desc +
          ' <a href="' + target + '">jump to section \\u2192</a>';
      });
      svg.appendChild(g); nodeEls[n.id] = g;
    });
    function highlight(id) {
      var near = {}; near[id] = 1;
      edgeEls.forEach(function (ln) {
        var hit = ln.__ends.indexOf(id) >= 0;
        ln.setAttribute('class', 'cmap-edge' + (hit ? ' hl' : ' dim'));
        if (hit) { near[ln.__ends[0]] = 1; near[ln.__ends[1]] = 1; }
      });
      NODES.forEach(function (n) {
        var cls = 'cmap-node' + (HUBS[n.id] ? ' hub' : '');
        if (n.id === id) cls += ' hl';
        else if (!near[n.id]) cls += ' dim';
        nodeEls[n.id].setAttribute('class', cls);
      });
    }
    function clearHl() {
      edgeEls.forEach(function (ln) { ln.setAttribute('class', 'cmap-edge'); });
      NODES.forEach(function (n) {
        nodeEls[n.id].setAttribute('class', 'cmap-node' + (HUBS[n.id] ? ' hub' : ''));
      });
    }
  }
})();

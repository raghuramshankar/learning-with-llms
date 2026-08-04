/* Interactive widgets for the language-modelling explainer.
   All simulations are the real algorithm, computed live in the browser.
   Each widget guards on its container id so this file is safe to load anywhere. */
(function () {
  'use strict';
  var $ = WLib.$, cssVar = WLib.cssVar, mulberry32 = WLib.mulberry32,
      gauss = WLib.gauss, onTheme = WLib.onTheme;

  function col(name, fb) { var c = cssVar(name); return c && c.trim() ? c.trim() : fb; }
  function softmax(z) {
    var m = Math.max.apply(null, z), s = 0, o = z.map(function (v) {
      var e = Math.exp(v - m); s += e; return e;
    });
    return o.map(function (e) { return e / s; });
  }
  function entropy(p) {
    var h = 0;
    p.forEach(function (v) { if (v > 1e-12) h -= v * Math.log(v); });
    return h;
  }

  /* =====================================================================
     1. BPE: train a real byte-pair encoder live and watch tokens grow.
     ===================================================================== */
  (function bpeWidget() {
    var host = document.getElementById('w-bpe');
    if (!host) return;
    var TEXT = "the deeper the network the lower the loss; the lower the loss " +
               "the better the model, and the model that models the model wins";
    var bytes = [];
    for (var i = 0; i < TEXT.length; i++) bytes.push(TEXT.charCodeAt(i));

    // Precompute the full merge list once; the slider just replays a prefix.
    function trainAll(maxMerges) {
      var ids = bytes.slice(), merges = [], vocab = {};
      for (var m = 0; m < maxMerges; m++) {
        var counts = new Map(), best = null, bestN = 1;
        for (var j = 0; j < ids.length - 1; j++) {
          var k = ids[j] + ',' + ids[j + 1];
          var c = (counts.get(k) || 0) + 1; counts.set(k, c);
          if (c > bestN) { bestN = c; best = k; }
        }
        if (!best) break;
        var parts = best.split(',').map(Number), nid = 256 + m;
        merges.push({ pair: parts, id: nid, count: bestN });
        vocab[nid] = (vocab[parts[0]] || String.fromCharCode(parts[0])) +
                     (vocab[parts[1]] || String.fromCharCode(parts[1]));
        var out = [], p = 0;
        while (p < ids.length) {
          if (p < ids.length - 1 && ids[p] === parts[0] && ids[p + 1] === parts[1]) {
            out.push(nid); p += 2;
          } else { out.push(ids[p]); p++; }
        }
        ids = out;
      }
      return { merges: merges, vocab: vocab };
    }
    var trained = trainAll(80);
    // BPE halts once no adjacent pair repeats, so a short corpus yields fewer
    // merges than asked for. Clamp the control to what actually exists rather
    // than indexing past the end.
    var MAXM = trained.merges.length;

    function encode(n) {
      var ids = bytes.slice();
      n = Math.min(n, MAXM);
      for (var m = 0; m < n; m++) {
        var mg = trained.merges[m], out = [], p = 0;
        while (p < ids.length) {
          if (p < ids.length - 1 && ids[p] === mg.pair[0] && ids[p + 1] === mg.pair[1]) {
            out.push(mg.id); p += 2;
          } else { out.push(ids[p]); p++; }
        }
        ids = out;
      }
      return ids;
    }
    function label(id) {
      return id < 256 ? String.fromCharCode(id) : trained.vocab[id];
    }

    var slider = $('w-bpe-n'), chips = $('w-bpe-chips'), stat = $('w-bpe-stat');
    slider.max = String(MAXM);
    function draw() {
      var n = Math.min(+slider.value, MAXM), ids = encode(n);
      chips.innerHTML = '';
      ids.forEach(function (id) {
        var s = document.createElement('span');
        s.className = 'wchip';
        s.textContent = label(id).replace(/ /g, '·');
        if (id >= 256) s.style.borderColor = col('--accent', '#5b8def');
        if (id >= 256) s.style.color = col('--accent', '#5b8def');
        chips.appendChild(s);
      });
      var bpt = (bytes.length / ids.length).toFixed(3);
      var last = n > 0 ? trained.merges[n - 1] : null;
      stat.innerHTML = '<strong>' + ids.length + '</strong> tokens for ' + bytes.length +
        ' bytes &mdash; <strong>' + bpt + '</strong> bytes/token, vocabulary ' + (256 + n) +
        (last ? '. Last merge: <code>' + label(last.id).replace(/ /g, '·') +
                '</code> (' + last.count + ' occurrences)' : '') +
        (n >= MAXM ? ' &mdash; <em>exhausted: no adjacent pair repeats any more, ' +
                     'so this corpus cannot fill a larger vocabulary.</em>' : '');
    }
    slider.addEventListener('input', draw);
    draw();
  })();

  /* =====================================================================
     2. Attention: why the 1/sqrt(d_k) is not cosmetic.
        Predict-before-run, then a live sweep over d_k.
     ===================================================================== */
  (function attnWidget() {
    var host = document.getElementById('w-attn');
    if (!host) return;
    var cv = $('w-attn-cv'), stat = $('w-attn-stat'), slider = $('w-attn-d');
    var scaled = true, predicted = null;

    $('w-attn-scale').addEventListener('click', function () {
      scaled = !scaled;
      this.textContent = scaled ? 'scaling: 1/√dₖ (on)' : 'scaling: none (off)';
      draw();
    });
    var predWrap = $('w-attn-predict');
    if (predWrap) {
      predWrap.querySelectorAll('[data-pred]').forEach(function (b) {
        b.addEventListener('click', function () {
          if (predicted) return;
          predicted = b.getAttribute('data-pred');
          predWrap.querySelectorAll('[data-pred]').forEach(function (x) {
            x.disabled = true;
          });
          b.style.borderColor = col('--accent', '#5b8def');
          resolvePrediction();
        });
      });
    }
    function resolvePrediction() {
      var fb = $('w-attn-predfb');
      if (!fb || !predicted) return;
      var right = predicted === 'sharper';
      fb.style.display = '';
      fb.innerHTML = (right ? '✓ ' : '✗ ') +
        'Turn the scaling <em>off</em> and drag dₖ up: the attention collapses onto a ' +
        'single key. Dot products of two random dₖ-dimensional unit-variance vectors have ' +
        'variance dₖ, so the logits spread as √dₖ and softmax saturates. Dividing ' +
        'by √dₖ holds the logit variance at 1 no matter how wide the head is' +
        (right ? '.' : ' — so attention gets <em>sharper</em>, not flatter.');
    }

    function draw() {
      var d = +slider.value, n = 12, rnd = mulberry32(7);
      var Q = [], K = [];
      for (var i = 0; i < n; i++) {
        var q = [], k = [];
        for (var j = 0; j < d; j++) { q.push(gauss(rnd)); k.push(gauss(rnd)); }
        Q.push(q); K.push(k);
      }
      var rows = [], maxw = 0, hsum = 0;
      for (i = 0; i < n; i++) {
        var s = [];
        for (var m = 0; m < n; m++) {
          var dot = 0;
          for (j = 0; j < d; j++) dot += Q[i][j] * K[m][j];
          s.push(m > i ? -Infinity : (scaled ? dot / Math.sqrt(d) : dot));
        }
        var p = softmax(s);
        rows.push(p);
        maxw = Math.max(maxw, Math.max.apply(null, p));
        hsum += entropy(p) / Math.log(i + 1 || 1) || 0;
      }
      // draw heatmap. Fall back to a sane width when the element has not been
      // laid out yet (hidden tab, print, zero-size viewport) — otherwise the
      // canvas silently paints nothing.
      var dpr = window.devicePixelRatio || 1, H = 190;
      var W = cv.clientWidth || (cv.parentElement && cv.parentElement.clientWidth) || 640;
      cv.width = W * dpr; cv.height = H * dpr;
      var g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      var cell = Math.min(H / n, W / n), x0 = (W - cell * n) / 2;
      var acc = col('--accent', '#5b8def');
      for (i = 0; i < n; i++) for (m = 0; m <= i; m++) {
        var a = rows[i][m];
        g.fillStyle = acc.replace(/^#/, '') === acc ? acc : acc;
        g.globalAlpha = Math.pow(a, 0.55);
        g.fillRect(x0 + m * cell, i * cell, cell - 1, cell - 1);
      }
      g.globalAlpha = 1;
      var lastRow = rows[n - 1];
      stat.innerHTML = 'dₖ = <strong>' + d + '</strong>, scaling ' +
        (scaled ? '<strong>on</strong>' : '<strong>off</strong>') +
        ' &mdash; peak attention weight in the last row: <strong>' +
        Math.max.apply(null, lastRow).toFixed(3) + '</strong>, entropy <strong>' +
        entropy(lastRow).toFixed(3) + '</strong> nats (uniform over ' + n + ' would be ' +
        Math.log(n).toFixed(3) + ')';
    }
    slider.addEventListener('input', draw);
    onTheme(draw);
    window.addEventListener('resize', draw);
    draw();
  })();

  /* =====================================================================
     3. Sampling: temperature / top-k / top-p on a fixed distribution,
        with a Monte-Carlo tally that converges to the theoretical mass.
     ===================================================================== */
  (function sampleWidget() {
    var host = document.getElementById('w-sample');
    if (!host) return;
    var BASE = [3.2, 2.6, 2.4, 1.1, 0.9, 0.4, 0.1, -0.3, -0.8, -1.5];
    var WORDS = ['model', 'network', 'system', 'method', 'token',
                 'banana', 'pencil', 'cloud', 'tuesday', 'quark'];
    var tally = new Array(BASE.length).fill(0), total = 0;
    var rnd = mulberry32(11);
    var tSl = $('w-sample-t'), kSl = $('w-sample-k'), pSl = $('w-sample-p');
    var bars = $('w-sample-bars'), stat = $('w-sample-stat');

    function dist() {
      var t = +tSl.value / 100, k = +kSl.value, p = +pSl.value / 100;
      var z = BASE.map(function (v) { return t > 0 ? v / t : v * 1e6; });
      if (k < BASE.length) {
        var sorted = z.slice().sort(function (a, b) { return b - a; });
        var cut = sorted[k - 1];
        z = z.map(function (v) { return v < cut ? -Infinity : v; });
      }
      var q = softmax(z);
      if (p < 1) {
        var idx = q.map(function (v, i) { return [v, i]; })
                   .sort(function (a, b) { return b[0] - a[0]; });
        var c = 0, keep = new Set();
        for (var i = 0; i < idx.length; i++) {
          keep.add(idx[i][1]); c += idx[i][0];
          if (c >= p) break;                       // include the crosser
        }
        var s = 0;
        q = q.map(function (v, i) { var kept = keep.has(i) ? v : 0; s += kept; return kept; });
        q = q.map(function (v) { return v / s; });
      }
      return q;
    }
    function drawBars() {
      var q = dist();
      bars.innerHTML = '';
      q.forEach(function (v, i) {
        var row = document.createElement('div');
        row.className = 'bar-row';
        var emp = total ? tally[i] / total : 0;
        row.innerHTML = "<span class='bar-lab'>" + WORDS[i] + "</span>" +
          "<span class='bar-track'><span class='bar-fill' style='width:" +
          (v * 100).toFixed(1) + "%'></span>" +
          "<span class='bar-emp' style='width:" + (emp * 100).toFixed(1) + "%'></span></span>" +
          "<span class='bar-val'>" + (v * 100).toFixed(1) + "%</span>";
        bars.appendChild(row);
      });
      stat.innerHTML = total
        ? '<strong>' + total + '</strong> draws &mdash; the thin bar is the empirical ' +
          'frequency, converging to the theoretical mass above it.'
        : 'Solid bars: the distribution you are sampling from. Draw to overlay the empirical counts.';
    }
    function draw(n) {
      var q = dist();
      for (var i = 0; i < n; i++) {
        var u = rnd(), c = 0;
        for (var j = 0; j < q.length; j++) { c += q[j]; if (u <= c) { tally[j]++; break; } }
        total++;
      }
      drawBars();
    }
    [tSl, kSl, pSl].forEach(function (s) {
      s.addEventListener('input', function () {
        tally = new Array(BASE.length).fill(0); total = 0;
        $('w-sample-cfg').textContent =
          'temperature ' + (+tSl.value / 100).toFixed(2) +
          ' · top-k ' + kSl.value +
          ' · top-p ' + (+pSl.value / 100).toFixed(2);
        drawBars();
      });
    });
    $('w-sample-1').addEventListener('click', function () { draw(1); });
    $('w-sample-200').addEventListener('click', function () { draw(200); });
    $('w-sample-reset').addEventListener('click', function () {
      tally = new Array(BASE.length).fill(0); total = 0; drawBars();
    });
    $('w-sample-cfg').textContent = 'temperature 1.00 · top-k 10 · top-p 1.00';
    drawBars();
  })();

  /* =====================================================================
     4. Compute budget: Chinchilla under both the published and the
        replicated fits, plus the 6ND accounting.
     ===================================================================== */
  (function budgetWidget() {
    var host = document.getElementById('w-budget');
    if (!host) return;
    var FITS = {
      published: { E: 1.69, A: 406.4, B: 410.7, a: 0.34, b: 0.28,
                   name: 'Hoffmann Approach 3 (as published)' },
      replicated: { E: 1.82, A: 482.0, B: 2085.0, a: 0.35, b: 0.37,
                    name: 'Besiroglu et al. re-estimate' }
    };
    function optimal(C, f) {
      var bestN = 0, bestL = Infinity, bestD = 0;
      for (var e = 7; e <= 13; e += 0.002) {
        var N = Math.pow(10, e), D = C / (6 * N);
        if (D < 1e6) continue;
        var L = f.E + f.A / Math.pow(N, f.a) + f.B / Math.pow(D, f.b);
        if (L < bestL) { bestL = L; bestN = N; bestD = D; }
      }
      return { N: bestN, D: bestD, L: bestL };
    }
    function human(x) {
      if (x >= 1e12) return (x / 1e12).toFixed(2) + 'T';
      if (x >= 1e9) return (x / 1e9).toFixed(2) + 'B';
      if (x >= 1e6) return (x / 1e6).toFixed(2) + 'M';
      return x.toExponential(2);
    }
    var sl = $('w-budget-c'), out = $('w-budget-out');
    function draw() {
      var C = Math.pow(10, +sl.value / 10);
      var rows = Object.keys(FITS).map(function (k) {
        var r = optimal(C, FITS[k]);
        return "<tr><td>" + FITS[k].name + "</td><td><strong>" + human(r.N) +
               "</strong></td><td><strong>" + human(r.D) + "</strong></td><td>" +
               (r.D / r.N).toFixed(1) + "</td><td>" + r.L.toFixed(3) + "</td></tr>";
      }).join('');
      out.innerHTML =
        "<div class='wstat'>Budget C = <strong>" + C.toExponential(1) +
        "</strong> FLOPs &nbsp;(&asymp; " + human(C / 6 / 1e9 * 1e9 / 1e9 * 1e9) +
        " parameter-tokens, since C &asymp; 6ND)</div>" +
        "<table><thead><tr><th>fit</th><th>optimal N</th><th>optimal D</th>" +
        "<th>D/N</th><th>loss</th></tr></thead><tbody>" + rows + "</tbody></table>";
    }
    sl.addEventListener('input', draw);
    draw();
  })();

  /* =====================================================================
     5. KV cache calculator: the memory wall of long-context decoding.
     ===================================================================== */
  (function kvWidget() {
    var host = document.getElementById('w-kv');
    if (!host) return;
    var ctx = $('w-kv-ctx'), batch = $('w-kv-b'), mode = $('w-kv-mode');
    var out = $('w-kv-out');
    var CFG = { layers: 80, heads: 64, head_dim: 128, bytes: 2 };
    function perToken(m) {
      if (m === 'mha') return 2 * CFG.layers * CFG.heads * CFG.head_dim * CFG.bytes;
      if (m === 'gqa') return 2 * CFG.layers * 8 * CFG.head_dim * CFG.bytes;
      return CFG.layers * 512 * CFG.bytes;                     // MLA latent
    }
    function draw() {
      var L = Math.pow(10, +ctx.value / 10), B = +batch.value;
      var names = { mha: 'MHA (64 kv heads)', gqa: 'GQA (8 groups)', mla: 'MLA (latent 512)' };
      var rows = ['mha', 'gqa', 'mla'].map(function (m) {
        var gb = perToken(m) * L * B / 1e9;
        var flag = gb > 80 ? " style='color:var(--fail,#d9705f)'" : '';
        return "<tr" + flag + "><td>" + names[m] + "</td><td>" +
               (perToken(m) / 1024).toFixed(1) + " KB</td><td><strong>" +
               gb.toFixed(2) + " GB</strong></td><td>" +
               (gb / 80).toFixed(2) + "&times;</td></tr>";
      }).join('');
      out.innerHTML =
        "<div class='wstat'>context <strong>" + Math.round(L).toLocaleString() +
        "</strong> tokens &times; batch <strong>" + B + "</strong> " +
        "&mdash; 70B-class model, bf16</div>" +
        "<table><thead><tr><th>attention variant</th><th>per token</th>" +
        "<th>KV cache</th><th>of one 80&nbsp;GB GPU</th></tr></thead><tbody>" +
        rows + "</tbody></table>";
    }
    [ctx, batch].forEach(function (s) { s.addEventListener('input', draw); });
    draw();
  })();

  /* =====================================================================
     6. Concept map.
     ===================================================================== */
  (function conceptMap() {
    var host = document.getElementById('w-cmap');
    if (!host) return;
    var N = [
      { id: 'bytes', x: 60, y: 40, w: 108, t: 'bytes / text', s: 'background' },
      { id: 'bpe', x: 210, y: 40, w: 104, t: 'BPE merges', s: 'math-token', hub: false },
      { id: 'vocab', x: 356, y: 40, w: 128, t: 'vocab / context', s: 'math-token' },
      { id: 'ce', x: 530, y: 40, w: 150, t: 'cross entropy → perplexity', s: 'math-token', hub: true },
      { id: 'embed', x: 60, y: 130, w: 108, t: 'embeddings', s: 'math-arch' },
      { id: 'attn', x: 210, y: 130, w: 104, t: 'attention', s: 'math-arch', hub: true },
      { id: 'rope', x: 356, y: 130, w: 128, t: 'RoPE / position', s: 'math-arch' },
      { id: 'ffn', x: 530, y: 130, w: 150, t: 'SwiGLU FFN + RMSNorm', s: 'math-arch' },
      { id: 'moe', x: 60, y: 220, w: 108, t: 'MoE routing', s: 'math-sota' },
      { id: 'flops', x: 210, y: 220, w: 104, t: 'C ≈ 6ND', s: 'math-train', hub: true },
      { id: 'adamw', x: 356, y: 220, w: 128, t: 'AdamW + schedule', s: 'math-train' },
      { id: 'scaling', x: 530, y: 220, w: 150, t: 'Chinchilla allocation', s: 'math-train', hub: true },
      { id: 'flash', x: 60, y: 310, w: 108, t: 'FlashAttention', s: 'math-train' },
      { id: 'parallel', x: 210, y: 310, w: 104, t: 'ZeRO / TP / PP', s: 'math-train' },
      { id: 'kv', x: 356, y: 310, w: 128, t: 'KV cache / GQA', s: 'math-sota' },
      { id: 'align', x: 530, y: 310, w: 150, t: 'SFT → RLHF / DPO / GRPO', s: 'math-sota', hub: true }
    ];
    var E = [
      ['bytes', 'bpe'], ['bpe', 'vocab'], ['vocab', 'ce'], ['vocab', 'embed'],
      ['embed', 'attn'], ['attn', 'rope'], ['attn', 'ffn'], ['ffn', 'moe'],
      ['attn', 'flops'], ['ffn', 'flops'], ['flops', 'scaling'], ['ce', 'scaling'],
      ['ce', 'adamw'], ['adamw', 'scaling'], ['attn', 'flash'], ['flops', 'parallel'],
      ['attn', 'kv'], ['kv', 'align'], ['ce', 'align'], ['moe', 'flops'],
      ['rope', 'kv'], ['scaling', 'align']
    ];
    var RECAP = {
      bytes: 'Text is bytes. Everything downstream is a choice about how to group them.',
      bpe: 'Merge the most frequent adjacent pair, repeat. Compression buys shorter sequences.',
      vocab: 'Bigger vocabulary means shorter sequences but a higher entropy ceiling per token.',
      ce: 'The training objective. Perplexity is its exponential: effective number of choices.',
      embed: 'Token ids become vectors; the unembedding ties back to the vocabulary.',
      attn: 'softmax(QKᵀ/√dₖ)V. The 1/√dₖ keeps the softmax out of saturation.',
      rope: 'Rotate query and key by position; the dot product then sees only relative distance.',
      ffn: 'Where most parameters live. RMSNorm drops the mean subtraction and the bias.',
      moe: 'Route each token to a few experts: more parameters, near-constant FLOPs per token.',
      flops: 'Forward+backward costs about 6 FLOPs per parameter per token. The whole budget.',
      adamw: 'Scale-free first step, decoupled weight decay, warmup then decay.',
      scaling: 'Given C, split it between N and D. The famous ratio is about 20 tokens/parameter.',
      flash: 'Tile attention in SRAM so the N² matrix is never written to HBM.',
      parallel: 'Shard optimizer state, layers, or the batch. Each choice trades memory for bandwidth.',
      kv: 'Decoding is memory-bandwidth bound; the KV cache is what you are moving.',
      align: 'Pretraining predicts text. Alignment makes it answer, via preferences or verifiers.'
    };
    var svg = host.querySelector('svg');
    var byId = {}; N.forEach(function (n) { byId[n.id] = n; });
    function cx(n) { return n.x + n.w / 2; } function cy(n) { return n.y + 17; }

    var frag = '';
    E.forEach(function (e, i) {
      var a = byId[e[0]], b = byId[e[1]];
      frag += "<path class='cmap-edge' data-a='" + e[0] + "' data-b='" + e[1] +
              "' d='M" + cx(a) + ' ' + cy(a) + ' L' + cx(b) + ' ' + cy(b) + "'/>";
    });
    N.forEach(function (n) {
      frag += "<g class='cmap-node" + (n.hub ? ' hub' : '') + "' data-id='" + n.id + "'>" +
              "<rect x='" + n.x + "' y='" + n.y + "' width='" + n.w + "' height='34' rx='7'/>" +
              "<text x='" + cx(n) + "' y='" + (n.y + 22) + "' text-anchor='middle'>" +
              n.t + "</text></g>";
    });
    svg.innerHTML = frag;

    var info = $('w-cmap-info');
    function highlight(id) {
      svg.querySelectorAll('.cmap-edge').forEach(function (p) {
        var on = p.getAttribute('data-a') === id || p.getAttribute('data-b') === id;
        p.classList.toggle('hl', on); p.classList.toggle('dim', !on);
      });
      var nbrs = new Set([id]);
      E.forEach(function (e) {
        if (e[0] === id) nbrs.add(e[1]); if (e[1] === id) nbrs.add(e[0]);
      });
      svg.querySelectorAll('.cmap-node').forEach(function (g) {
        var gid = g.getAttribute('data-id');
        g.classList.toggle('hl', gid === id);
        g.classList.toggle('dim', !nbrs.has(gid));
      });
    }
    function clear() {
      svg.querySelectorAll('.cmap-edge,.cmap-node').forEach(function (el) {
        el.classList.remove('hl', 'dim');
      });
    }
    svg.querySelectorAll('.cmap-node').forEach(function (g) {
      var id = g.getAttribute('data-id');
      g.addEventListener('mouseenter', function () { highlight(id); });
      g.addEventListener('mouseleave', clear);
      g.addEventListener('click', function () {
        var sec = byId[id].s;
        var href = (window.__PAGES__ || {})[sec];
        info.innerHTML = "<strong>" + byId[id].t + ".</strong> " + RECAP[id] +
          (href ? " <a href='" + href + "'>Go to the section →</a>" : '');
      });
    });
    info.innerHTML = 'Hover a node to isolate its connections; click for a one-breath recap. ' +
      'The outlined nodes are the load-bearing hubs: if you can only keep four ideas, keep those.';
  })();
})();

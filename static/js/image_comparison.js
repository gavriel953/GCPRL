/**
 * Image Comparison Slider — v3 (bug-fixed).
 *
 * Root cause of "both images look the same":
 *   The afterImg was styled width:100% inside the clipping wrapper.
 *   When the wrapper clips to e.g. 30% width, the afterImg also shrinks
 *   to 30% of container — it scales DOWN rather than being clipped.
 *   So at every position, you see a squished version of the after image
 *   filling the left portion, indistinguishable from the before image.
 *
 * Fix: afterImg width is set to the CONTAINER's pixel width, not 100% of
 * the wrapper. This way clipping the wrapper reveals/hides the right part
 * of a full-width image — true comparison slider behaviour.
 *
 * Additional fixes:
 *   - Height is locked via ResizeObserver after both images load
 *   - Drag starts immediately on mousedown anywhere on container
 *   - RAF prevents duplicate position updates
 *   - Touch support preserved
 */
class ImageComparisonSlider {
  constructor(container, beforeB64, afterB64, labels = ['Original', 'Enhanced']) {
    this.container  = container;
    this.dragging   = false;
    this.position   = 50;
    this.rafPending = false;
    this._build(beforeB64, afterB64, labels);
    this._bindEvents();
  }

  _build(beforeB64, afterB64, labels) {
    this.container.innerHTML = '';
    Object.assign(this.container.style, {
      position:   'relative',
      overflow:   'hidden',
      cursor:     'col-resize',
      userSelect: 'none',
      display:    'block',
      background: 'var(--bg-0)',
      minHeight:  '200px',
    });

    // ── BEFORE image (bottom layer, full width always) ────────────────────
    this.beforeImg = this._img(beforeB64, 'Original');
    Object.assign(this.beforeImg.style, {
      display:  'block',
      width:    '100%',
      height:   'auto',
    });

    // ── AFTER wrapper (clips via overflow:hidden + explicit px width) ─────
    this.afterWrapper = document.createElement('div');
    Object.assign(this.afterWrapper.style, {
      position:   'absolute',
      top:        '0',
      left:       '0',
      height:     '100%',
      overflow:   'hidden',
      width:      '50%',      // will be updated by _setPosition
    });

    // ── AFTER image (inside wrapper, fixed to CONTAINER width in px) ──────
    // KEY FIX: width is set in px to match container, not '100%' of wrapper.
    this.afterImg = this._img(afterB64, 'Enhanced');
    Object.assign(this.afterImg.style, {
      position:   'absolute',
      top:        '0',
      left:       '0',
      display:    'block',
      height:     'auto',
      // width is set in px after layout — see _syncAfterWidth()
    });
    this.afterWrapper.appendChild(this.afterImg);

    // ── Divider ───────────────────────────────────────────────────────────
    this.divider = document.createElement('div');
    Object.assign(this.divider.style, {
      position:      'absolute',
      top:           '0',
      left:          '50%',
      width:         '3px',
      height:        '100%',
      background:    'var(--accent-primary)',
      transform:     'translateX(-50%)',
      zIndex:        '10',
      pointerEvents: 'none',
      boxShadow:     '0 0 10px rgba(0,184,212,0.7)',
    });

    // ── Handle ────────────────────────────────────────────────────────────
    this.handle = document.createElement('div');
    Object.assign(this.handle.style, {
      position:       'absolute',
      top:            '50%',
      left:           '50%',
      transform:      'translate(-50%, -50%)',
      width:          '40px',
      height:         '40px',
      borderRadius:   '50%',
      background:     'var(--accent-primary)',
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'center',
      color:          '#fff',
      fontSize:       '0.9rem',
      fontWeight:     '700',
      boxShadow:      '0 2px 12px rgba(0,0,0,0.5)',
      cursor:         'col-resize',
      zIndex:         '11',
    });
    this.handle.textContent = '⟺';
    this.divider.appendChild(this.handle);

    // ── Labels ────────────────────────────────────────────────────────────
    this.labelBefore = this._label(labels[0], 'left',  false);
    this.labelAfter  = this._label(labels[1], 'right', true);

    this.container.append(
      this.beforeImg, this.afterWrapper,
      this.divider, this.labelBefore, this.labelAfter,
    );

    // Lock height + sync after-image width once before image is ready
    const onLoad = () => {
      this._lockHeight();
      this._syncAfterWidth();
      this._setPosition(50);
    };
    if (this.beforeImg.complete && this.beforeImg.naturalWidth) {
      onLoad();
    } else {
      this.beforeImg.addEventListener('load', onLoad, { once: true });
    }
  }

  _img(b64, alt) {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${b64}`;
    img.alt = alt;
    img.draggable = false;
    return img;
  }

  _label(text, side, accent) {
    const el = document.createElement('div');
    el.textContent = text;
    Object.assign(el.style, {
      position:      'absolute',
      top:           '10px',
      [side]:        '12px',
      fontFamily:    'var(--font-mono, monospace)',
      fontSize:      '0.72rem',
      padding:       '3px 10px',
      borderRadius:  '4px',
      pointerEvents: 'none',
      zIndex:        '12',
      background:    accent ? 'rgba(0,184,212,0.18)' : 'rgba(8,13,20,0.75)',
      color:         accent ? 'var(--accent-primary, #00b8d4)' : 'rgba(200,210,220,0.9)',
      border:        accent ? '1px solid rgba(0,184,212,0.4)' : 'none',
    });
    return el;
  }

  _lockHeight() {
    const nw = this.beforeImg.naturalWidth;
    const nh = this.beforeImg.naturalHeight;
    if (!nw) return;
    const h = (nh / nw) * this.container.offsetWidth;
    this.container.style.height = `${Math.round(h)}px`;
  }

  /** Set afterImg width in px = container width, so it never squishes. */
  _syncAfterWidth() {
    const w = this.container.offsetWidth;
    if (w > 0) this.afterImg.style.width = `${w}px`;
  }

  _setPosition(pct) {
    this.position = Math.max(1, Math.min(99, pct));
    const p = this.position;
    this.afterWrapper.style.width = `${p}%`;
    this.divider.style.left       = `${p}%`;
  }

  _pctFromEvent(clientX) {
    const rect = this.container.getBoundingClientRect();
    return ((clientX - rect.left) / rect.width) * 100;
  }

  _bindEvents() {
    this.container.addEventListener('mousedown', (e) => {
      this.dragging = true;
      this._setPosition(this._pctFromEvent(e.clientX));
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.dragging || this.rafPending) return;
      this.rafPending = true;
      requestAnimationFrame(() => {
        this._setPosition(this._pctFromEvent(e.clientX));
        this.rafPending = false;
      });
    });
    window.addEventListener('mouseup', () => { this.dragging = false; });

    this.container.addEventListener('touchstart', (e) => {
      this.dragging = true;
      this._setPosition(this._pctFromEvent(e.touches[0].clientX));
    }, { passive: true });
    window.addEventListener('touchmove', (e) => {
      if (!this.dragging || this.rafPending) return;
      e.preventDefault();
      this.rafPending = true;
      requestAnimationFrame(() => {
        this._setPosition(this._pctFromEvent(e.touches[0].clientX));
        this.rafPending = false;
      });
    }, { passive: false });
    window.addEventListener('touchend', () => { this.dragging = false; });

    // Re-lock on resize
    if (window.ResizeObserver) {
      new ResizeObserver(() => {
        this._lockHeight();
        this._syncAfterWidth();
        this._setPosition(this.position);
      }).observe(this.container);
    }
  }

  update(beforeB64, afterB64) {
    this.beforeImg.src = `data:image/png;base64,${beforeB64}`;
    this.afterImg.src  = `data:image/png;base64,${afterB64}`;
    const onLoad = () => {
      this._lockHeight();
      this._syncAfterWidth();
      this._setPosition(50);
    };
    if (this.beforeImg.complete && this.beforeImg.naturalWidth) onLoad();
    else this.beforeImg.addEventListener('load', onLoad, { once: true });
  }
}

window.ImageComparisonSlider = ImageComparisonSlider;

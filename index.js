var linalgModule = require('linalg-asm');

module.exports = function pdipQuad(n, m) {
  'use strict';
  var uintBytes = 4,
      floatBytes = 8,
      QLength = n * n,
      ALength = n * m,
      bLength = m,
      cLength = n,
      xLength = n,
      yLength = m,
      zLength = n,
      xyLength = xLength + yLength,
      wLength = xLength + yLength + zLength,
      JLength = xyLength * xyLength,
      rLength = wLength,
      rXLength = xLength,
      rYLength = yLength,
      rZLength = zLength,
      floatLength = QLength + ALength + bLength + cLength + wLength + JLength + rLength,
      ipivLength = Math.ceil((xLength + yLength) / 2) * 2,
      heap = new ArrayBuffer(calcBufferSize(floatLength * floatBytes + ipivLength * uintBytes)),
      Q = new Float64Array(heap, 0, QLength),
      A = new Float64Array(heap, Q.byteOffset + Q.byteLength, ALength),
      b = new Float64Array(heap, A.byteOffset + A.byteLength, bLength),
      c = new Float64Array(heap, b.byteOffset + b.byteLength, cLength),
      w = new Float64Array(heap, c.byteOffset + c.byteLength, wLength),
      J = new Float64Array(heap, w.byteOffset + w.byteLength, JLength),
      r = new Float64Array(heap, J.byteOffset + J.byteLength, rLength),
      ipiv = new Uint32Array(heap, r.byteOffset + r.byteLength, ipivLength),
      x = new Float64Array(heap, w.byteOffset, xLength),
      y = new Float64Array(heap, x.byteOffset + x.byteLength, yLength),
      z = new Float64Array(heap, y.byteOffset + y.byteLength, zLength),
      rX = new Float64Array(heap, r.byteOffset, rXLength),
      rY = new Float64Array(heap, rX.byteOffset + rX.byteLength, rYLength),
      rZ = new Float64Array(heap, rY.byteOffset + rY.byteLength, rZLength),
      linalg = linalgModule(global, null, heap);

  return {
    q: function() {
      return Q;
    },
    a: function() {
      return A;
    },
    b: function() {
      return b;
    },
    c: function() {
      return c;
    },
    w: function() {
      return w;
    },
    solve: solve
  };

  function solve(options) {
    options = options || {};
    var i, j;
    var mu = options.mu || 0.8,
        muMin = options.muMin || 1e-10,
        gamma = options.gamma || 0.5,
        M = options.M || 2,
        rho = options.rho || 100,
        xi = options.xi || 0.9;

    for (i = 0; i < n; ++i) {
      x[i] = 1;
      y[i] = 0;
      z[i] = 1;
    }

    for (var loop = 0; mu > muMin; ++loop, mu *= gamma) {
      for (;;) {
        // r_x := A ^ t y + z - Q x - c
        linalg.dcopy(n, z.byteOffset, 1, rX.byteOffset, 1);
        linalg.daxpy(n, -1, c.byteOffset, 1, rX.byteOffset, 1);
        linalg.dgemv(0, n, n, -1, Q.byteOffset, n, x.byteOffset, 1, 1, rX.byteOffset, 1);
        linalg.dgemv(1, m, n, 1, A.byteOffset, n, y.byteOffset, 1, 1, rX.byteOffset, 1);

        // r_y := A x - b
        linalg.dcopy(n, b.byteOffset, 1, rY.byteOffset, 1);
        linalg.dgemv(0, m, n, 1, A.byteOffset, n, x.byteOffset, 1, -1, rY.byteOffset, 1);

        // r_z := mu e - X Z e
        for (i = 0; i < n; ++i) {
          rZ[i] = mu - x[i] * z[i];
        }

        if (linalg.ddot(r.length, r.byteOffset, 1, r.byteOffset, 1) < mu * M) {
          break;
        }

        // J := (  Q-X^(-1)Z  -A^t )
        //        -A           O
        // r_x := r_x - X ^ -1 r_z
        for (i = 0; i < xyLength; ++i) {
          for (j = 0; j < xyLength; ++j) {
            J[i * xyLength + j] = 0;
          }
        }
        for (i = 0; i < n; ++i) {
          for (j = 0; j < n; ++j) {
            J[i * xyLength + j] = Q[i * n + j];
          }
          for (j = 0; j < m; ++j) {
            J[(j + n) * xyLength + i] = J[i * xyLength + j + n] = -A[j * n + i];
          }
          J[i * xyLength + i] += z[i] / x[i];
          rX[i] += rZ[i] / x[i];
        }

        // solve J dw = r
        linalg.dgesv(xyLength, 1, J.byteOffset, xyLength, ipiv.byteOffset, r.byteOffset, 1);

        // dz = X^-1 (Z dx - r_z)
        for (i = 0; i < n; ++i) {
          rZ[i] = (rZ[i] - z[i] * rX[i]) / x[i];
        }

        // Find alpha_x and alpha_z
        var alpha = 1;
        for (i = 0; i < n; ++i) {
          if (rX[i] < 0) {
            alpha = Math.min(alpha, -x[i] / rX[i]);
          }
          if (rZ[i] < 0) {
            alpha = Math.min(alpha, -z[i] / rZ[i]);
          }
        }
        alpha *= 0.99;

        // Armijo condition
        var dpx = dp(mu, rho);
        if (dpx > 0) {
          break;
        }
        var px = p(mu, rho);
        linalg.daxpy(n, alpha, rX.byteOffset, 1, x.byteOffset, 1);
        for (var k = 1; !(p(mu, rho) <= px + xi * alpha * Math.pow(gamma, k) * dpx) && k < 100; ++k) {
          linalg.daxpy(n, alpha * Math.pow(gamma, k) * (gamma - 1), rX.byteOffset, 1, x.byteOffset, 1);
        }
        linalg.daxpy(m, alpha * Math.pow(gamma, k - 1), rY.byteOffset, 1, y.byteOffset, 1);
        linalg.daxpy(n, alpha * Math.pow(gamma, k - 1), rZ.byteOffset, 1, z.byteOffset, 1);
      }
    }

    return {
      x: x
    };
  }

  function f() {
    var i, j, val = 0;
    for (i = 0; i < n; ++i) {
      for (j = 0; j < n; ++j) {
        val += x[i] * x[j] * Q[i * n + j];
      }
    }
    val /= 2;
    for (i = 0; i < n; ++i) {
      val += x[i] * c[i];
    }
    return val;
  }

  function g(i) {
    var j, val = -b[i];
    for (j = 0; j < n; ++j) {
      val += x[j] * A[i * n + j];
    }
    return val;
  }

  function p(mu, rho) {
    var i, val;
    val = f();
    for (i = 0; i < n; ++i) {
      val -= mu * Math.log(x[i]);
    }
    for (i = 0; i < m; ++i) {
      val += rho * Math.abs(g(i));
    }
    return val;
  }

  function dp(mu, rho) {
    var i, j, val = 0, qx, gjx, ax;
    for (i = 0; i < n; ++i) {
      qx = c[i];
      for (j = 0; j < n; ++j) {
        qx += Q[i * n + j] * x[j];
      }
      val += qx * rX[i];
    }
    for (i = 0; i < n; ++i) {
      val -= mu * rX[i] / x[i];
    }
    for (i = 0; i < m; ++i) {
      gjx = g(i);
      ax = 0;
      for (j = 0; j < n; ++j) {
        ax += A[i * n + j] * rX[j];
      }
      val += rho * (Math.abs(gjx + ax) - Math.abs(gjx));
    }
    return val;
  }

  function calcBufferSize(s) {
    var l = 1;
    while (l < s) {
      l <<= 1;
    }
    return l;
  }
};

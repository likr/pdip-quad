var linalgModule = require('linalg-asm');

module.exports = function pdipQuad(n, m) {
  'use strict';

  var nm = n + m;
  var memory = allocate({
    Float64: {
      Q: n * n,
      A: n * m,
      b: m,
      c: n,
      w: {
        x: n,
        y: m,
        z: n
      },
      J: nm * nm,
      K: m * n,
      L: n * n,
      r: {
        rX: n,
        rY: m,
        rZ: n
      },
      work: n
    },
    Uint32: {
      ipiv: m
    }
  });
  var linalg = linalgModule(global, null, memory.heap);

  (function() {
    var i;
    for (i = 0; i < n; ++i) {
      memory.views.x[i] = 1;
      memory.views.y[i] = 0;
      memory.views.z[i] = 1;
    }
  })();

  return {
    q: function() {
      return memory.views.Q;
    },
    a: function() {
      return memory.views.A;
    },
    b: function() {
      return memory.views.b;
    },
    c: function() {
      return memory.views.c;
    },
    w: function() {
      return memory.views.w;
    },
    solve: solve
  };

  function solve(options) {
    options = options || {};
    var i, j,
        xFeasible = false,
        yFeasible = false;
    var A = memory.views.A,
        b = memory.views.b,
        c = memory.views.c,
        Q = memory.views.Q,
        J = memory.views.J,
        K = memory.views.K,
        L = memory.views.L,
        x = memory.views.x,
        y = memory.views.y,
        z = memory.views.z,
        r = memory.views.r,
        rX = memory.views.rX,
        rY = memory.views.rY,
        rZ = memory.views.rZ,
        work = memory.views.work,
        ipiv = memory.views.ipiv;
    var mu = options.mu || 0.8,
        muMin = options.muMin || 1e-10,
        gamma = options.gamma || 0.5,
        M = options.M || 2,
        rho = options.rho || 100,
        tau = options.tau || 0.8,
        xi = options.xi || 0.9,
        err = options.err || 1e-12;
    var reduce = options.reduce === undefined ? false : options.reduce;

    for (var loop = 0; mu > muMin; ++loop, mu *= gamma) {
      for (;;) {
        // r_z := mu e - X Z e
        for (i = 0; i < n; ++i) {
          rZ[i] = mu - x[i] * z[i];
        }

        if (xFeasible) {
          // r_x := O
          for (i = 0; i < n; ++i) {
            rX[i] = 0;
          }
        } else {
          // r_x := A ^ t y + z - Q x - c
          linalg.dcopy(n, z.byteOffset, 1, rX.byteOffset, 1);
          linalg.daxpy(n, -1, c.byteOffset, 1, rX.byteOffset, 1);
          linalg.dgemv(0, n, n, -1, Q.byteOffset, n, x.byteOffset, 1, 1, rX.byteOffset, 1);
          linalg.dgemv(1, m, n, 1, A.byteOffset, n, y.byteOffset, 1, 1, rX.byteOffset, 1);

          if (linalg.ddot(rX.length, rX.byteOffset, 1, rX.byteOffset, 1) < err) {
            xFeasible = true;
          }
        }

        if (yFeasible) {
          // r_y := O
          for (i = 0; i < m; ++i) {
            rY[i] = 0;
          }
        } else {
          // r_y := A x - b
          linalg.dcopy(m, b.byteOffset, 1, rY.byteOffset, 1);
          linalg.dgemv(0, m, n, 1, A.byteOffset, n, x.byteOffset, 1, -1, rY.byteOffset, 1);

          if (linalg.ddot(rY.length, rY.byteOffset, 1, rY.byteOffset, 1) < err) {
            yFeasible = true;
          }
        }

        if (linalg.ddot(r.length, r.byteOffset, 1, r.byteOffset, 1) < mu * M) {
          break;
        }

        if (reduce) {
          // L := (Q + X^(-1) Z)^-1
          for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
              L[i * n + j] = Q[i * n + j];
            }
            L[i * n + i] += z[i] / x[i];
          }
          linalg.dgetrf(n, n, L.byteOffset, n, ipiv.byteOffset, n);
          linalg.dgetri(n, L.byteOffset, n, ipiv.byteOffset, work.byteOffset, n);

          // K := A L
          linalg.dgemm(0, 0, m, n, n, 1, A.byteOffset, n, L.byteOffset, n, 0, K.byteOffset, n);

          // J := K A^t
          linalg.dgemm(0, 1, m, m, n, 1, K.byteOffset, n, A.byteOffset, n, 0, J.byteOffset, m);

          // r_y := - r_y - K (r_x + X^(-1) r_z)
          for (i = 0; i < n; ++i) {
            work[i] = rX[i] + rZ[i] / x[i];
          }
          linalg.dgemv(0, m, n, -1, K.byteOffset, n, work.byteOffset, 1, -1, rY.byteOffset, 1);

          // solve J dw = r
          linalg.dgesv(m, 1, J.byteOffset, m, ipiv.byteOffset, rY.byteOffset, 1);

          // dx := L (r_x + X^-1 r_z + A^t d_y)
          linalg.dgemv(1, m, n, 1, A.byteOffset, n, rY.byteOffset, 1, 1, work.byteOffset, 1);
          linalg.dgemv(0, n, n, 1, L.byteOffset, n, work.byteOffset, 1, 0, rX.byteOffset, 1);
        } else {
          // J := (  Q-X^(-1)Z  -A^t )
          //        -A           O
          // r_x := r_x - X ^ -1 r_z
          for (i = 0; i < nm; ++i) {
            for (j = 0; j < nm; ++j) {
              J[i * nm + j] = 0;
            }
          }
          for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
              J[i * nm + j] = Q[i * n + j];
            }
            for (j = 0; j < m; ++j) {
              J[(j + n) * nm + i] = J[i * nm + j + n] = -A[j * n + i];
            }
            J[i * nm + i] += z[i] / x[i];
            rX[i] += rZ[i] / x[i];
          }

          // solve J dw = r
          linalg.dgesv(nm, 1, J.byteOffset, nm, ipiv.byteOffset, r.byteOffset, 1);
        }

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
        var dpx = dp(n, m, A, b, c, Q, x, rX, mu, rho);
        if (dpx > 0) {
          break;
        }
        var px = p(n, m, A, b, c, Q, x, mu, rho);
        linalg.daxpy(n, alpha, rX.byteOffset, 1, x.byteOffset, 1);
        for (var k = 1; !(p(n, m, A, b, c, Q, x, mu, rho) <= px + xi * alpha * Math.pow(tau, k) * dpx) && k < 100; ++k) {
          linalg.daxpy(n, alpha * Math.pow(tau, k) * (tau - 1), rX.byteOffset, 1, x.byteOffset, 1);
        }
        linalg.daxpy(m, alpha * Math.pow(tau, k - 1), rY.byteOffset, 1, y.byteOffset, 1);
        linalg.daxpy(n, alpha * Math.pow(tau, k - 1), rZ.byteOffset, 1, z.byteOffset, 1);
      }
    }

    return {
      x: x
    };
  }
};

function f(n, m, Q, c, x) {
  'use strict';
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

function g(n, m, A, b, x, i) {
  'use strict';
  var j, val = -b[i];
  for (j = 0; j < n; ++j) {
    val += x[j] * A[i * n + j];
  }
  return val;
}

function p(n, m, A, b, c, Q, x, mu, rho) {
  'use strict';
  var i, val;
  val = f(n, m, Q, c, x);
  for (i = 0; i < n; ++i) {
    val -= mu * Math.log(x[i]);
  }
  for (i = 0; i < m; ++i) {
    val += rho * Math.abs(g(n, m, A, b, x, i));
  }
  return val;
}

function dp(n, m, A, b, c, Q, x, rX, mu, rho) {
  'use strict';
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
    gjx = g(n, m, A, b, x, i);
    ax = 0;
    for (j = 0; j < n; ++j) {
      ax += A[i * n + j] * rX[j];
    }
    val += rho * (Math.abs(gjx + ax) - Math.abs(gjx));
  }
  return val;
}

function allocate(arg) {
  'use strict';
  var totalSize = 0;

  totalSize += 8 * countLength(arg.Float64);
  totalSize += 4 * countLength(arg.Uint32);

  var heap = new ArrayBuffer(calcBufferSize(totalSize)),
      views = {};
  allocViews(arg.Uint32, 8 * allocViews(arg.Float64, 0, Float64Array), Uint32Array);

  return {
    heap: heap,
    views: views
  };

  function countLength(obj) {
    var length = 0;
    for (var name in obj) {
      if (typeof obj[name] === 'number') {
        length += obj[name];
      } else {
        length += countLength(obj[name]);
      }
    }
    return length;
  }

  function allocViews(obj, offset, TArray) {
    var length, totalLength = 0;
    for (var name in obj) {
      if (typeof obj[name] === 'number') {
        length = obj[name];
      } else {
        length = allocViews(obj[name], offset, TArray);
      }
      views[name] = new TArray(heap, offset, length);
      offset += views[name].byteLength;
      totalLength += length;
    }
    return totalLength;
  }

  function calcBufferSize(s) {
    var l = 0x1000;
    while (l < s) {
      l <<= 1;
    }
    return l;
  }
}

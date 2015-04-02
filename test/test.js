var expect = require('expect.js'),
    pdip = require('../index');

describe('pdip', function() {
  'use strict';
  it('solves max. x^t Q x / 2 + c^t x s.t. A x = b, x >= 0', function() {
    var n = 5, m = 3;

    var solver = pdip(n, m);
    solver.q().set([2, 0, 0, 0, 0,
                    0, 2, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0]);
    solver.a().set([1, 1, -1, 0, 0,
                    1, -1, 0, -1, 0,
                    -3, 1, 0, 0, -1]);
    solver.b().set([-0.5, 1, -3]);
    solver.c().set([-2, -4, 0, 0, 0]);

    var result = solver.solve(1);

    expect(+result.x[0].toFixed(10)).to.be(1);
    expect(+result.x[1].toFixed(10)).to.be(0);
  });

  it('solves max. x^t Q x / 2 + c^t x s.t. A x = b, x >= 0', function() {
    var n = 5, m = 3;

    var solver = pdip(n, m);
    solver.q().set([1, -1, 0, 0, 0,
                    -1, 2, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0]);
    solver.a().set([1, 1, 1, 0, 0,
                    -1, 2, 0, 1, 0,
                    2, 1, 0, 0, 1]);
    solver.b().set([2, 2, 3]);
    solver.c().set([-2, -6, 0, 0, 0]);

    var result = solver.solve(1);

    expect(+result.x[0].toFixed(4)).to.be(0.6667);
    expect(+result.x[1].toFixed(4)).to.be(1.3333);
  });

  it('solves max. x^t Q x / 2 + c^t x s.t. A x = b, x >= 0', function() {
    var n = 2, m = 1;

    var solver = pdip(n, m);
    solver.q().set([2, 1, 1, 3]);
    solver.a().set([2, 3]);
    solver.b().set([10]);
    solver.c().set([-1, -2]);

    var result = solver.solve(1);

    expect(+result.x[0].toFixed(4)).to.be(1.5);
    expect(+result.x[1].toFixed(4)).to.be(2.3333);
  });

  it('solves max. x^t Q x / 2 + c^t x s.t. A x = b, x >= 0', function() {
    var n = 2, m = 2;

    var solver = pdip(n, m);
    solver.q().set([2, 1, 1, 3]);
    solver.a().set([2, 3, 3, 2.5]);
    solver.b().set([10, 12]);
    solver.c().set([-1, -2]);

    var result = solver.solve(1);

    expect(+result.x[0].toFixed(10)).to.be(2.75);
    expect(+result.x[1].toFixed(10)).to.be(1.5);
  });
});

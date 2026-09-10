import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../../docs/site/normals.js', import.meta.url), 'utf8');
const { smoothNormals } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function wedge(angle) {
  const s = Math.sin(angle), c = Math.cos(angle);
  return new Float32Array([
    0,0,0, 0,0,1, 1,0,0, 0,0,1, 0,1,0, 0,0,1,
    0,0,0, 0,s,c, 0,-c,s, 0,s,c, 1,0,0, 0,s,c,
  ]);
}

test('smooth joins share normals without moving any triangle', () => {
  const data = wedge(Math.PI / 9), original = data.slice();
  smoothNormals(data);
  for (let i = 0; i < data.length; i += 6) {
    assert.deepEqual(data.slice(i, i + 3), original.slice(i, i + 3));
    assert.ok(Math.abs(Math.hypot(...data.slice(i + 3, i + 6)) - 1) < 1e-6);
  }
  assert.deepEqual(data.slice(3, 6), data.slice(21, 24));
  assert.ok(data[4] > 0 && data[4] < Math.sin(Math.PI / 9));
});

test('right-angle creases retain their original normals', () => {
  const data = wedge(Math.PI / 2), original = data.slice();
  smoothNormals(data);
  assert.deepEqual(data, original);
});

test('degenerate triangles do not introduce NaNs', () => {
  const data = new Float32Array(18);
  smoothNormals(data);
  assert.ok(data.every(Number.isFinite));
});

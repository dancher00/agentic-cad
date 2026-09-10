// Lighting only: preserve every STL vertex and triangle, retaining sharp creases.
export function smoothNormals(data, creaseAngle = Math.PI / 6) {
  const groups = new Map();
  const weights = new Float32Array(data.length / 6);
  for (let i = 0; i < data.length; i += 6) {
    const key = `${data[i]},${data[i + 1]},${data[i + 2]}`;
    const group = groups.get(key);
    if (group) group.push(i); else groups.set(key, [i]);
    const triangle = Math.floor(i / 18) * 18;
    const corner = (i - triangle) / 6;
    const b = triangle + ((corner + 1) % 3) * 6;
    const c = triangle + ((corner + 2) % 3) * 6;
    const u = [data[b] - data[i], data[b + 1] - data[i + 1], data[b + 2] - data[i + 2]];
    const v = [data[c] - data[i], data[c + 1] - data[i + 1], data[c + 2] - data[i + 2]];
    const length = Math.hypot(...u) * Math.hypot(...v);
    weights[i / 6] = length ? Math.acos(Math.max(-1, Math.min(1,
      (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / length))) : 0;
  }
  const normals = new Float32Array(data.length / 2);
  const minimumDot = Math.cos(creaseAngle);
  for (const group of groups.values()) {
    for (const i of group) {
      let x = 0, y = 0, z = 0;
      for (const j of group) {
        const dot = data[i + 3] * data[j + 3] + data[i + 4] * data[j + 4] + data[i + 5] * data[j + 5];
        if (dot >= minimumDot) {
          const weight = weights[j / 6];
          x += data[j + 3] * weight;
          y += data[j + 4] * weight;
          z += data[j + 5] * weight;
        }
      }
      const length = Math.hypot(x, y, z);
      normals[i / 2] = length ? x / length : data[i + 3];
      normals[i / 2 + 1] = length ? y / length : data[i + 4];
      normals[i / 2 + 2] = length ? z / length : data[i + 5];
    }
  }
  for (let i = 0; i < data.length; i += 6) data.set(normals.subarray(i / 2, i / 2 + 3), i + 3);
}

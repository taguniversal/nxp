import * as d3 from "d3";
let TopologyHook = {
  mounted() {
    this.render(this.el.dataset.topology);

    this.handleEvent("update_topology", ({ topology }) => {
      this.render(topology);
    });
  },

  updated() {
    const topology = this.el.dataset.topology;
    d3.select(this.el).select("svg").remove();
    this.render(topology);
  },

  render(topology) {
    const width = 400;
    const height = 400;
    const svg = d3.select(this.el)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const nodeCount = 6;
    const radius = 150;
    const centerX = width / 2;
    const centerY = height / 2;

    const nodes = d3.range(nodeCount).map(i => {
      const angle = (2 * Math.PI / nodeCount) * i;
      return {
        id: i,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      };
    });

    // Draw nodes
    svg.selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", 12)
      .attr("fill", "#4f46e5");

    // Draw links based on topology
    const lines = [];

    switch (topology) {
      case "mesh":
        nodes.forEach((n1, i) => {
          nodes.forEach((n2, j) => {
            if (i < j && Math.abs(i - j) === 1) {
              lines.push([n1, n2]);
            }
          });
        });
        break;

      case "ring":
        for (let i = 0; i < nodeCount; i++) {
          lines.push([nodes[i], nodes[(i + 1) % nodeCount]]);
        }
        break;

      case "bus":
        nodes.forEach(n => {
          lines.push([{ x: centerX, y: centerY }, n]);
        });
        break;

      case "tree":
        for (let i = 1; i < nodeCount; i++) {
          const parent = Math.floor((i - 1) / 2);
          lines.push([nodes[i], nodes[parent]]);
        }
        break;

      case "crossbar":
        nodes.forEach(n1 => {
          nodes.forEach(n2 => {
            if (n1 !== n2) lines.push([n1, n2]);
          });
        });
        break;
    }

    // Draw lines
    svg.selectAll("line")
      .data(lines)
      .enter()
      .append("line")
      .attr("x1", d => d[0].x)
      .attr("y1", d => d[0].y)
      .attr("x2", d => d[1].x)
      .attr("y2", d => d[1].y)
      .attr("stroke", "#888")
      .attr("stroke-width", 2);
  }
};

export default TopologyHook;
